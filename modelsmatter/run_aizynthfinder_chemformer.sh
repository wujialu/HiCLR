export PYTHONPATH=$(git rev-parse --show-toplevel):$PYTHONPATH
export PYTHONPATH="$(git rev-parse --show-toplevel)/external/modelsmatter_modelzoo/external_models/Chemformer:$PYTHONPATH"
export PYTHONPATH="$(git rev-parse --show-toplevel)/external/modelsmatter_modelzoo/external_models/HiCLR:$PYTHONPATH"
export PYTHONPATH="$(git rev-parse --show-toplevel)/external/modelsmatter_modelzoo:$PYTHONPATH"
echo $PYTHONPATH

expansion_policy=chemformer
# target_smiles=paroutes_n1_random_100
target_smiles=caspyrus_random_100
nproc=8
gpu_ids='0 1 2 3 4 5 6 7'

for target_smiles in caspyrus_random_100 paroutes_n1_random_100
do
    mkdir ./experiments/output/${expansion_policy}_paroutes/${target_smiles}
    python ./aizynthfinder/interfaces/aizynthcli.py \
        --config ./configs/${expansion_policy}_default_config.yml \
        --smiles ./experiments/data/${target_smiles}.txt \
        --policy ${expansion_policy} \
        --output ./experiments/output/${expansion_policy}_paroutes/${target_smiles}/output.hdf5 \
        --nproc ${nproc} \
        --gpu_ids ${gpu_ids}
done