import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
import pickle
import re
import torch
from torch.utils.data import DataLoader
from args_parse import args_parser
from utils.build_utils import build_model, load_checkpoint_downstream, load_vocab


def load_pretrained_transformer(args, vocab_itos_src, vocab_itos_tgt, checkpoint_path, vocab_itos_ec=None):
    model = build_model(args, vocab_itos_src, vocab_itos_tgt, vocab_itos_ec)
    model = load_checkpoint_downstream(checkpoint_path, model)
    return model


def canonical_smiles(smi):
    """Canonicalize a SMILES without atom mapping"""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    else:
        canonical_smi = Chem.MolToSmiles(mol)
        # print('>>', canonical_smi)
        if '.' in canonical_smi:
            canonical_smi_list = canonical_smi.split('.')
            canonical_smi_list = sorted(canonical_smi_list, key=lambda x: (len(x), x))
            canonical_smi = '.'.join(canonical_smi_list)
        return canonical_smi
    

def smi_tokenizer(smi):  
    """Tokenize a SMILES sequence or reaction"""
    pattern = "(\[[^\]]+]|Bi|Br?|Ge|Te|Mo|K|Ti|Zr|Y|Na|125I|Al|Ce|Cr|Cl?|Ni?|O|S|Pd?|Fe?|I|b|c|Mn|n|o|s|<unk>|>>|Li|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    if smi != ''.join(tokens):
        print('ERROR:', smi, ''.join(tokens))
    assert smi == ''.join(tokens)
    return tokens


def collate_fn(data, src_pad=1, tgt_pad=1):
    src, tgt = zip(*data)
    max_src_length = max([len(s) for s in src])
    max_tgt_length = max([len(t) for t in tgt])

    anchor = torch.zeros([])

    # Pad_sequence
    new_src = anchor.new_full((max_src_length, len(data)), src_pad, dtype=torch.long)
    new_tgt = anchor.new_full((max_tgt_length, len(data)), tgt_pad, dtype=torch.long)

    for i in range(len(data)):
        new_src[:, i][:len(src[i])] = torch.LongTensor(src[i])
        new_tgt[:, i][:len(tgt[i])] = torch.LongTensor(tgt[i])
    
    return new_src, new_tgt

if __name__ == "__main__":
    args = args_parser()
    model_type = "hiclr"
    args.checkpoint = f"./checkpoint/supcon_hierar/model_pretrain_best_mAP.pt"
    args.vocab_file = "./data/vocab_share.pk"
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    src_itos, src_stoi, tgt_itos, tgt_stoi = load_vocab(args.vocab_file)
    model = load_pretrained_transformer(args, src_itos, tgt_itos, args.checkpoint)
    model.to(args.device)
    model.eval()

    data_dir = "./data/schneider_classification/"
    dataset_name = "raw_test"
    data = pd.read_csv(os.path.join(data_dir, f"{dataset_name}.csv"))
    reactants = data["reactant_smiles"].tolist()
    product = data["product_smiles"].tolist()

    rxn_dataset = []
    for (reacts, prod) in tqdm(zip(reactants, product)):
        src_tokens = ["<UNK>"] + smi_tokenizer(canonical_smiles(prod))
        tgt_tokens =  ['<sos>'] + smi_tokenizer(canonical_smiles(reacts)) + ['<eos>'] #! w/o atom-mapping

        src_token_ids = [src_stoi.get(t, src_stoi['<unk>']) for t in src_tokens]
        tgt_token_ids = [tgt_stoi.get(t, tgt_stoi['<unk>']) for t in tgt_tokens]

        rxn_dataset.append([src_token_ids, tgt_token_ids])

    loader = DataLoader(rxn_dataset, collate_fn=collate_fn, batch_size=32, num_workers=12)

    react_fps = []
    prod_fps = []
    for batch_data in tqdm(loader):
        src_token_ids, tgt_token_ids = batch_data
        src_token_ids, tgt_token_ids = src_token_ids.to(args.device), tgt_token_ids.to(args.device)
        src_reps, tgt_reps = model._reaction_fp(src=src_token_ids, tgt=tgt_token_ids)
        prod_fps.append(src_reps.detach().cpu())
        react_fps.append(tgt_reps.detach().cpu())
            

    react_fps = torch.cat(react_fps, dim=0).numpy()
    prod_fps = torch.cat(prod_fps, dim=0).numpy()
    np.savez_compressed(os.path.join(data_dir, f'{model_type}_prod_fps'), fps=prod_fps)
    np.savez_compressed(os.path.join(data_dir, f'{model_type}_rxn_fps'), fps=react_fps)