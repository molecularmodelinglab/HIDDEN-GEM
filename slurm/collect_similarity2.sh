#!/bin/sh

# get all the .dist file
ls $PWD/*.dist > file_list

thing=""
i=1
res_name=1

cat file_list|while read each_file
do
  PYTHON_ENV="PATH_TO_PYTHON"
  SCRIPT_LOC="PATH_TO_SCRIPT"
  thing=$thing" "$each_file
  # echo $each_file
  if [ $((i%120)) == 0 ]; then
          sbatch -N 1 -n 4 --job-name=sim_search --mem=40g -t 00-12:00:00 --constraint="[rhel7|rhel8]" --wrap="$PATH_TO_PYTHON $PATH_TO_SCRIPT --dist_files $thing --num_total 200000 --max_hits_per_query 2000 --max_dist 15.0 --outdir $PWD"
          #echo $thing
          thing=""
          res_name=$((res_name+1))
  fi
  i=$((i+1))
done
rm file_list