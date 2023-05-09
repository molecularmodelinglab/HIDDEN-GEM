#!/bin/sh

ls PATHS_TO_ALL_REFERENCE_FILES > file_list

cat file_list|while read each_file
do

PYTHON_ENV="PATH_TO_PYTHON"
SCRIPT_LOC="PATH_TO_SCRIPT"
QUERY_LOC="PATH_TO_QUERY_FILE"

sbatch -N 1 -n 1 --job-name=sim_search --mem=1g -t 4-12:00:00 --constraint="[rhel7|rhel8]" --wrap="$PYTHON_ENV $SCRIPT_LOC --reference_file $each_file --query_file $QUERY_LOC --out_dir $PWD"

done

rm file_list