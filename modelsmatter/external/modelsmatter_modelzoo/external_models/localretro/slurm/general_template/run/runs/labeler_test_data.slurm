#!/bin/bash
#SBATCH --job-name=LocalRetro_extract_templates
#SBATCH --partition=core
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --array=0-9

ROW_INDEX=$((SLURM_ARRAY_TASK_ID))


train_template_python_script=<LOCALRETRO_FOLDER>distributed_preprocessing/preprocessing/localretro_template_extraction.py
raw_data=<EXPERIMENT_FOLDER>/run/csv_batches/test/raw_test_batch_${ROW_INDEX}.csv
output_directory=<EXPERIMENT_FOLDER>/run/labeling_data/test/test_labeling_data_${ROW_INDEX}.pkl

python ${train_template_python_script} -dp ${raw_data} -out ${output_directory} --use_labeler True