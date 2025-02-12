# This makefile is used to create experiments for the AIDD LocalRetro project.

# the dataset name
experiment_name = paroutes

# the path of the localretro installation
localretro_folder = <Path>/aidd_localretro/
raw_data_path = <Path>/experiments/datasets/singlestep/paroutes
raw_data_file_path = <Path>/experiments/datasets/singlestep/paroutes/rawdata_paroutes.csv
experiment_folder_path = <Path>/experiments/singlestep/$(experiment_name)

# change these folders if you dont want to use the localretro slurm folder of localretro
#raw_data_file_name = rawdata_az_1M.csv
#raw_data_path = $(PWD)/raw_data/$(experiment_name)/
#raw_data_file_path = $(raw_data_path)$(raw_data_file_name)
#experiment_folder_path = $(PWD)/results/$(experiment_name)

create_experiment: create_folder replace_strings_in_makefile replace_strings_in_slurm_scripts replace_strings_in_sbatch_scripts

create_folder:
	cp -r general_template $(experiment_folder_path)
	mkdir -p $(raw_data_path)

replace_strings_in_makefile:
	find $(experiment_folder_path) -name 'Makefile' -exec sed -i -e 's?<EXPERIMENT_NAME>?$(experiment_name)?' {} \;
	find $(experiment_folder_path) -name 'Makefile' -exec sed -i -e 's?<EXPERIMENT_FOLDER>?$(experiment_folder_path)?' {} \;
	find $(experiment_folder_path) -name 'Makefile' -exec sed -i -e 's?<RAW_DATA_PATH>?$(raw_data_path)?' {} \;
	find $(experiment_folder_path) -name 'Makefile' -exec sed -i -e 's?<RAW_DATA_FILE_PATH>?$(raw_data_file_path)?' {} \;
	find $(experiment_folder_path) -name 'Makefile' -exec sed -i -e 's?<LOCALRETRO_FOLDER>?$(localretro_folder)?' {} \;
	
replace_strings_in_slurm_scripts:
	find $(experiment_folder_path)/run/runs -name '*.slurm' -exec sed -i -e 's?<LOCALRETRO_FOLDER>?$(localretro_folder)?' {} \;
	find $(experiment_folder_path)/run/runs -name '*.slurm' -exec sed -i -e 's?<EXPERIMENT_FOLDER>?$(experiment_folder_path)?' {} \;
	find $(experiment_folder_path)/run/runs -name '*.slurm' -exec sed -i -e 's?<EXPERIMENT_NAME>?$(experiment_name)?' {} \;
	find $(experiment_folder_path)/run/runs -name '*.slurm' -exec sed -i -e 's?<EXPERIMENT_NAME>?$(experiment_name)?' {} \;
	# twice EXPERIMENT_NAME because two occurences

replace_strings_in_sbatch_scripts:
	find $(experiment_folder_path)/model/ -name '*.slurm' -exec sed -i -e 's?<LOCALRETRO_FOLDER>?$(localretro_folder)?' {} \;
	find $(experiment_folder_path)/model/ -name '*.slurm' -exec sed -i -e 's?<EXPERIMENT_FOLDER>?$(experiment_folder_path)?' {} \;
