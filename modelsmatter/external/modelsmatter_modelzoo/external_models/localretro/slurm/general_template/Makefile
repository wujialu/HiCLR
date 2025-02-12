############################################# LocalRetro Preprocessing #######################################
#srun -p core -c 4 --mem=120g -N 1 -t 0-24:00:00 --pty /bin/bash

############################################# Process Holdout Splits #######################################
holdout_python_script = <LOCALRETRO_FOLDER>distributed_preprocessing/data/holdout_splitter.py
target_data_folder = <RAW_DATA_PATH>
raw_data = <RAW_DATA_FILE_PATH>


# given the raw data, split into holdout splits
create_holdout_splits:
	python $(holdout_python_script) --dataset $(raw_data) --target_folder $(target_data_folder)

############################################# Create Slurm Data Splits #######################################

split_python = <LOCALRETRO_FOLDER>distributed_preprocessing/data/slurm_csv_splitter.py
data_split_folder = $(target_data_folder)
batch_path = <EXPERIMENT_FOLDER>/run/csv_batches/

split_data:
	python $(split_python) --dataset $(data_split_folder)raw_train.csv --chunk_size 10000 --target_folder  $(batch_path)train/
	python $(split_python) --dataset $(data_split_folder)raw_val.csv --chunk_size 10000 --target_folder  $(batch_path)val/
	python $(split_python) --dataset $(data_split_folder)raw_test.csv --chunk_size 10000 --target_folder  $(batch_path)test/

#########################################################################################################
####################################### CREATE TRAINING TEMPLATES #######################################

slurm_path=<EXPERIMENT_FOLDER>/run/runs/

delete_slurm:
	cd $(slurm_path) && rm *.err && rm *.out

# we need to create the atom, bond, and reaction templates for the training data
############ SLURM
# one wants to run that in parallel instead of sequentiallz
# extract_train_template_python_script=<LOCALRETRO_FOLDER>distributed_preprocessing/preprocessing/localretro_template_extraction.py
# raw_training_data=<RAW_DATA_PATH>raw_train.csv
# output_directory=<EXPERIMENT_FOLDER>/run/training_templates/raw_train_batch_template.pkl
#
# extract_training_templates:
#	 python $(extract_train_template_python_script) -dp $(raw_training_data) -out $(output_directory)
############ End SLURM

####################################### join training_template SLURM results #############################
extract_train_template_python_script=<LOCALRETRO_FOLDER>distributed_preprocessing/preprocessing/localretro_template_extraction.py
training_template_pkl_folder=<EXPERIMENT_FOLDER>/run/training_templates/
training_template_result_path =<EXPERIMENT_FOLDER>/data/
training_template_output_directory=$(training_template_result_path)raw_train_batch_templates.pkl

join_training_pkl_files:
	python $(extract_train_template_python_script) -dp $(training_template_pkl_folder) -out $(training_template_output_directory) --join_pkl True > run.txt

process_train_template_python_script = <LOCALRETRO_FOLDER>distributed_preprocessing/preprocessing/localretro_preprocessing.py

# define a minimum template occurence for the training templates
min_template_frequency = 3

process_training_templates:
	python $(process_train_template_python_script) --template_path $(training_template_output_directory) -out $(training_template_result_path) --min_template_freq $(min_template_frequency) > run.txt

############################################### LABELING ###############################################

####################################### Labeling Preoprocessing #######################################

# create the reaction templates necessary for the labeling process. Again, this should be done in parallel
############ SLURM
# one wants to run that in parallel instead of sequential
#extract_train_template_python_script=<LOCALRETRO_FOLDER>distributed_preprocessing/preprocessing/localretro_template_extraction.py
#raw_training_data=<RAW_DATA_PATH>/raw_train.csv
#output_directory=<EXPERIMENT_FOLDER>/labeling_data/train_labeling_data.pkl

#extract_label_templates:
#	python $(extract_train_template_python_script) -dp $(raw_training_data) -out $(output_directory) --use_labeler
############ End SLURM

label: join_labeler_pkl_files label_dataset finalize_labeling

label_template_folder=<EXPERIMENT_FOLDER>/run/labeling_data
join_labeler_pkl_files:
	python $(extract_train_template_python_script) -dp $(label_template_folder)/train/ -out $(label_template_folder)/labeler_train_data.pkl --join_pkl True --use_labeler True > run.txt
	python $(extract_train_template_python_script) -dp $(label_template_folder)/val/ -out $(label_template_folder)/labeler_val_data.pkl --join_pkl True --use_labeler True >> run.txt
	python $(extract_train_template_python_script) -dp $(label_template_folder)/test/ -out $(label_template_folder)/labeler_test_data.pkl --join_pkl True --use_labeler True >> run.txt


labeler_python_script = <LOCALRETRO_FOLDER>distributed_preprocessing/labeling/localretro_labeler.py
training_atom_bond_templates_path=<EXPERIMENT_FOLDER>/results
label_data_path=<EXPERIMENT_FOLDER>/run/labeling_data
min_template_occurence=3

label_dataset:
	python $(labeler_python_script) --min-template-n $(min_template_occurence) -tp $(training_atom_bond_templates_path) --label_data_path $(label_data_path)/labeler_train_data.pkl -o $(training_atom_bond_templates_path) -s train > run.txt
	python $(labeler_python_script) -tp $(training_atom_bond_templates_path) --label_data_path $(label_data_path)/labeler_val_data.pkl -o $(training_atom_bond_templates_path) -s val >> run.txt
	python $(labeler_python_script) -tp $(training_atom_bond_templates_path) --label_data_path $(label_data_path)/labeler_test_data.pkl -o $(training_atom_bond_templates_path) -s test >> run.txt


finalize_labeling:
	python $(labeler_python_script) --min-template-n $(min_template_occurence) -tp $(training_atom_bond_templates_path) --label_data_path $(label_data_path)/labeler_train_data.pkl -o $(training_atom_bond_templates_path) --join True > finalize_labeling.txt


################################################# Graph Creation before training #####################################################



split_labeled_python = <LOCALRETRO_FOLDER>distributed_preprocessing/data/slurm_csv_splitter.py
labeled_data_path = <EXPERIMENT_FOLDER>/data/labeled_data.csv
graph_batch_path = <EXPERIMENT_FOLDER>/run/graph_creation
output_folder_name = <EXPERIMENT_NAME>

split_labeled_data:
	python $(split_labeled_python) --dataset $(labeled_data_path) --chunk_size 10000 --target_folder $(graph_batch_path) --localretro_graph_creation --output_folder_name $(output_folder_name)

############################## SLURM THIS #### SLURM THIS #### SLURM THIS
#
# ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found solution:
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<Path>/.conda/envs/single_step_benchmark/lib/
#
#localretro_scripts_path = <LOCALRETRO_FOLDER>scripts/
#labeled_batch_folder =<EXPERIMENT_FOLDER>/run/graph_creation/raw/<EXPERIMENT_NAME>_batch_0/
#dataset_batch_name = <EXPERIMENT_NAME>_batch_0
#reporting_graph_creation_file = $(labeled_batch_folder)graph_creation_$(dataset_batch_name).txt
#
#
#graph_create_localretro:
#	cd $(localretro_scripts_path) && python Train.py -d $(dataset_batch_name) --graph_creation_only --graph_creation_csv_folder $(labeled_batch_folder) > $(reporting_graph_creation_file)
#
#
############################################ END SLURM THIS #### END SLURM THIS #### END SLURM THIS

join_graphs_python_path=<LOCALRETRO_FOLDER>distributed_preprocessing/graph_extraction/localretro_graph_extraction.py
graph_output_file_name=<EXPERIMENT_NAME>_dglgraph.bin
graph_output_file = <EXPERIMENT_FOLDER>/data/$(graph_output_file_name)

join_localretro_graphs:
	python $(join_graphs_python_path) --graph_batch_folder_path $(graph_batch_path) --output_file_path $(graph_output_file) > $(graph_output_file)_creation.txt




################################################# Train NEW INTERFACE LocalRetro #####################################################
# note, you have to create a data folder in the localretro/data folder
# srun -p gpu -c 4 --mem=50g -N 1 --gres=gpu:1 -t 7-00:00:00 --pty /bin/bash
# ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found solution:
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<Path>/.conda/envs/single_step_benchmark/lib/

localretro_scripts_path = <LOCALRETRO_FOLDER>scripts/
model_file_path = <EXPERIMENT_FOLDER>/model/<EXPERIMENT_NAME>_model.bin
config_path = <EXPERIMENT_FOLDER>/model/training_config.json
template_path = <EXPERIMENT_FOLDER>/data/
labeled_data_file_path = <EXPERIMENT_FOLDER>/data/labeled_data.csv
graph_file_path = <EXPERIMENT_FOLDER>/data/<EXPERIMENT_NAME>_dglgraph.bin
reporting_output_file = $(model_file_path)_training.txt
batch_size = 32
max_epochs = 50
early_stopping_patience = 3


train_localretro:
	cd $(localretro_scripts_path) && python Train.py --num-epochs $(max_epochs) --patience $(early_stopping_patience) --batch-size $(batch_size) --model_file_path $(model_file_path) --config_path $(config_path) --template_path $(template_path) --labeled_data_path $(labeled_data_file_path) --graph_file_path $(graph_file_path) > $(reporting_output_file)





