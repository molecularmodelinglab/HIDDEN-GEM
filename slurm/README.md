# SLURM

Every slurm set up is unique, so the code provided here will likely not just run without any changes.
It is provided more as a guide to show how we parallelized the similarity search on out HPC cluster using slurm
It could also be parallelized on the Cloud, on a large workstation.

First, the reference file was split up into 800 separate files of equal number of smiles. Then a single core was used
run `simseach.py` on each split reference file. `submit_similairty.sh` was used for this (sorry about the awful piping).

After all these jobs finished, 3 TB of similarity data is created, but to extract the N most similar compound, we used
`collect_dist.py`. Since loading TB onto RAM is unlikely, this is batched into many mini jobs and covers 3 steps. You cal alter the setting to pick how many compounds you want to find (must be teh same in all 3 collection shell scripts).

At the end, you will then have `.dist` file with the N most similar compounds to the quires. You can convert with `dist_to_csv.py`