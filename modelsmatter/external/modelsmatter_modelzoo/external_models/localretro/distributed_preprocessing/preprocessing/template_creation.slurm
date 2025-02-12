#!/bin/bash
#SBATCH --job-name=LocalRetro_extract_templates
#SBATCH --error=%x_%j.err
#SBATCH --partition=core
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB
#SBATCH --array=0-19

ROW_INDEX=$((SLURM_ARRAY_TASK_ID))


train_template_python_script=<Path>/localretro/preprocessing/localretro_template_extraction.py
raw_training_data=<Path>/experiments/ssbenchmark/localretro/raw/raw_train_batch_${ROW_INDEX}.csv
output_directory=<Path>/experiments/ssbenchmark/localretro/templates/raw_train_batch_template_${ROW_INDEX}.pkl

python ${train_template_python_script} -dp ${raw_training_data} -out ${output_directory}