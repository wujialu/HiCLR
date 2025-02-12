#!/bin/bash
#SBATCH --job-name=LocalRetro_graph_creation
#SBATCH --partition=core
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --array=0-78

ROW_INDEX=$((SLURM_ARRAY_TASK_ID))

localretro_scripts_path=<LOCALRETRO_FOLDER>scripts/
model_file_path=<EXPERIMENT_FOLDER>/model/<EXPERIMENT_NAME>_model.bin
config_path=<EXPERIMENT_FOLDER>/model/training_config.json
template_path=<EXPERIMENT_FOLDER>/data/
labeled_data_file_path=<EXPERIMENT_FOLDER>/run/graph_creation/<EXPERIMENT_NAME>_batch_${ROW_INDEX}/labeled_data.csv
graph_file_path=<EXPERIMENT_FOLDER>/run/graph_creation/<EXPERIMENT_NAME>_batch_${ROW_INDEX}/<EXPERIMENT_NAME>_graphs_batch_${ROW_INDEX}.bin

cd ${localretro_scripts_path} && python Train.py --graph_creation_only --model_file_path ${model_file_path} --config_path ${config_path} --template_path ${template_path} --labeled_data_path ${labeled_data_file_path} --graph_file_path ${graph_file_path}

