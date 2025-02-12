#!/bin/bash
#SBATCH --job-name=localretro_18M                              # job name
#SBATCH --time=90-00:00:00                                                 # wall time dd-hh:mm:ss (dd = days, hh = hours, mm = minutes, ss = seconds)
#SBATCH --partition=gpu                                               # -p, set the cluster partition to gpu
#SBATCH --nodes=1                                                       # -N, node count
#SBATCH --cpus-per-task=4                                           # -c, cpu core count
#SBATCH --gres=gpu:1                                                 # specify use of a generic resource (single gpu)
#SBATCH --mem=350g                                                     # RAM requirement

source activate ssbenchmark
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<Path>/.conda/envs/single_step_benchmark/lib/

localretro_scripts_path=<LOCALRETRO_FOLDER>scripts/
model_file_path=<EXPERIMENT_FOLDER>/model/<EXPERIMENT_NAME>_model.bin
config_path=<EXPERIMENT_FOLDER>/model/training_config.json
template_path=<EXPERIMENT_FOLDER>/data/
labeled_data_file_path=<EXPERIMENT_FOLDER>/data/labeled_data.csv
graph_file_path=<EXPERIMENT_FOLDER>/data/<EXPERIMENT_NAME>_dglgraph.bin

batch_size=32
max_epochs=50
early_stopping_patience=3

cd ${localretro_scripts_path} && python Train.py --num-epochs ${max_epochs} --patience ${early_stopping_patience} --batch-size ${batch_size} --model_file_path ${model_file_path} --config_path ${config_path} --template_path ${template_path} --labeled_data_path ${labeled_data_file_path} --graph_file_path ${graph_file_path}