# HIDDEN-GEM

This repo contains the scripts used to run HIDDEN GEM

Generic code to carry out generation and similarity searching can be found in the `scripts` folder
as `generate.py` and `simsearch.py`. These scripts are standalone, and require no additional files to run, outside
of a python environment with the correct packages. Helper scripts are included as well. See below sections
for details on how to run them

HIDDEN GEM was designed and originally run on an HPC system that used a SLURM scheduler, as well as a compressed
database of the ENAMINE Real Space. The generic code as been altered such that it can run on any system using a libarry 
that is simply a file containing the SMILES and name of compounds in the library.
However, there are some example scripts in the slurm folder to assist with implementing HIDDEN GEM to run with SLURM
if that resource is available to you.

We do not provide code for running docking as this is highly dependent on the docking software you use, some of which 
are commercial and require a license. Code to run program like AutodockVina or SMINA is easily available in other repos

HIDDEN GEM is somewhat manual, requiring that you submit the Docking jobs, wait for it to finish, submit the generation 
job using the previous output wait for it to finish, then submit the simsearch job using the previous output. A job 
scheduler could be used to link them (a pet project of mine) but for now its is up to the user to keep track of when a 
specific step finishes and the next can be submitted.

## Setup

install a python environment with the correct packages using the `environment.yaml` file provided

## Generation

Use `python3 generate.py --help` (using the python interpreter installed above) to see a descscription of all the
options and parameters for the generation model.

As a minimum, `generate.py` requires: 
 - --inpath: A file with SMILES used for fine-tuning. This will normally be a file with SMILES and docking scores. `generate.py` will take care of selecting the top scoring SMILES for you

The SMILES you pass can be of two formats:
 1. it can have SMILES and scores (docking scores are used in the paper). This is the default assumption `generate.py` makes.
The script will then you the `--score_quantile` parameter to pick out the top compounds for fine-tuning the generation model.
2. it can be just the set of SMILES that is to be used for fine-tuning. In this case `--bias_set` must be used, otherwise `generate.py` will crash.
NOTE: using this format, the filtering model will be turned off, as there are no scores to make a filtering model (see below)

`generate.py` will also handle making the Filter model for you. By default, it will create a filter model, however this can be turned off using `--no_filter`.
The filter model will assign the compounds to the positive and negative classes based on the `--score_quantile`, and will allow compounds to pass only if the confidence in prediction for the positive class is above `--conf_threshold`.

By default, `generate.py` will save generated compounds and information on the fly (using `--prefix` to handle naming):
 - `{prefix}_bias_set.csv` is the set of smiles used for fine-tuning
 - `{prefix}_denovo_hits.smi` is the SMILES of the generated hits
 - `{prefix}_num_hits.txt` track the total number of denovo compounds that passed the filter at each batch
 - `{prefix}_status.txt` tells you some quick status about the model
 - `{prefix}_RF.joblib` is the file for the RandomForestClassifier (SKLearn) used for filtering
 - `{prefix}_tuned.pt` is the fine-tuned generative model parameters

Various setting for the fine-tuning can be changes. They have good defaults, but tweaking things like the `--fine_tune_lr`
or `--fine_tune_epochs` can adjust how biased the model becomes.

`generate.py` supports both cuda and CPU

Last, while a pretrained model is provided, the option to pass a different trained model is available. 
Same for the filter model. This also allows you to reuse previous models if you desire (and in that case you should use `--warm_start`)

## Similarity Searching

`simsearch.py` is the code used for similarity search. It is a brute force algorithm, but we are able to search all 37 
billion against 100 queries (so 3.7 trillion comparisons) in about 4 hours on 800 cores (using slurm of course). 
As a rule of thumb, for every core you can do 4.5 billion comparisons in 4ish hours, so keep that in mind when picking 
a computer and reference/query set.
Any similarity commercial or open source software of your choice can be substituted (like SpaceLight from BioSolveIT). 
This script is offered as an easy implementation if no other options are available for you. 

This Similarity Search works by assigning the smallest distance (in chemical feature space) to any query compound to a 
given reference compound. So, for example, with 100 queries, each reference is compared to all 100, and the closest 
distance is saved as its "similarity" of the reference to the query. The corresponding query that was most similar is also saved.

This will then create a type of file call `.dist`. This file type will have a row for each reference which includes the name and smiles of the reference (if name is provided, otherwise the index in the file), along with the distance to the closest query, and the name and index of that closet query. 
By default, only the nearest query is saved, but by adjusting `--n_neighbors` you can increase the nubmer is saves. Be wary though, about 3.0 TB of data is created when run on 37 billion compounds, keeping 1 query. every additional query kept will increase that by 1TB. The program will crash if it runs out of memory to write to.

To use `simsearch.py` you need to provide:
 - --query_file: the file with SMILES you want to act as the query. This file is completely loaded at the start, so try and keep it small to avoid eating lots of memory.
 - --reference_file: the file with SMILES you want as the reference. The reference file is read on the fly, so it can be as small or large as you want, it will not cause memory issues.

These input files should have headers, and you can tell the program which column header hold the SMILES or Names with `--query_smi_col`, `--ref_smi_col`, `--query_name_col`, or `--ref_name_col`.
if no name col is given, it will default to assign the name as the line number that SMILES was on in the respective file. 

This similarity search is embarrassingly parallel and can be parallelized easily but splitting the reference file into N files and `simsearch.py` on all N files.
This is how we parallelized on SLURM (see the `slurm` folder). Bash utils like `split` can be helpful for this. A util called `dist_to_csv.py` is also included to convert the `.dist` file to a csv for easy input into most docking programs