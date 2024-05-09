from omegaconf import OmegaConf
import yaml
import os

wandb_project = "grid_search"
os.makedirs(f"./config/moleculenet/{wandb_project}", exist_ok=True)

tasks = [
    "BBBP",
    "BACE",
    "HIV",
    "SIDER",
    "ClinTox",
    "Tox21",
]


config = yaml.load(open("./config/moleculenet/config_finetune.yaml", "r"), Loader=yaml.FullLoader)

config_files = {}

for i, t in enumerate(tasks):
    oc = OmegaConf.structured(config)
    oc.task_name = t
    oc.gpu = f"cuda:0"
    for j, finetune in enumerate(["adapter", "full_finetune", "last_layer"]):
        oc.finetune = finetune
        for pool in ["mean", "cls"]: 
            oc.model.pool = pool
            for lr_group in [[0.001, 0.001], [0.001, 0.0001], [0.0001, 0.0001]]:
                oc.init_lr = lr_group[0]
                oc.init_base_lr = lr_group[1]
                for dropout in [0.3, 0.4, 0.5]:
                    oc.model.drop_ratio = dropout 
                    param_name = f"{t}_{finetune}_{pool}_{lr_group[0]}_{lr_group[1]}_{dropout}"
                    filename = f"./config/moleculenet/{wandb_project}/config_{param_name}.yaml"
                    config_files[param_name] = filename
                    OmegaConf.save(config=oc, f=filename)

# base_cmd = "python finetune_moleculenet_molformer.py --config {}"
# list_file = f"./config/{wandb_project}/benchmark_sweep_list.txt"
# with open(list_file, "w+") as f:
#     for pname, fi in config_files.items():
#         cmd = base_cmd.format(fi, pname)
#         f.write(f"{cmd}\n")