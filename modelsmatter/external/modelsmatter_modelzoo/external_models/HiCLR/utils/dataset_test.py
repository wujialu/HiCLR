"Ref: MolCLR"

import os
import csv
import math
import time
import random
import numpy as np
from typing import List
import pickle

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger                                                                                                                                                               
RDLogger.DisableLog('rdApp.*')  
from utils.smiles_utils import canonical_smiles, smi_tokenizer, randomize_smiles_with_am, clear_map_number
from collections import defaultdict


ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = len(dataset)
    print(data_len)

    print("About to generate scaffolds")
    for ind, smiles in enumerate(dataset.smiles_data):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        #! change
        scaffold = _generate_scaffold(smiles, include_chirality=True)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets


def scaffold_split(dataset, valid_size, test_size, seed=None, log_every_n=1000):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds

def read_smiles(data_path, target, task):
    smiles_data, labels, masks = [], [], []
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i != 0:
                smiles = row['smiles']
                label = [row[t] for t in target]
                mol = Chem.MolFromSmiles(smiles)
                if mol != None and label != []:
                    smiles_data.append(smiles)
                    if task == 'classification':
                        labels.append([-1 if y=="" else int(y) for y in label])
                        masks.append([0 if y=="" else 1 for y in label])
                        # labels.append([int(y) for y in label])
                    elif task == 'regression':
                        labels.append([float(y) for y in label])
                        masks.append([0 if y=="" else 1 for y in label])
                    else:
                        ValueError('task must be either regression or classification')
    print(len(smiles_data))
    return smiles_data, labels, masks


def read_reactions(data_path, target, task):
    smiles_data, labels = [], []
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i != 0:
                smiles = row['smiles']
                label = row[target]
                mol = Chem.MolFromSmiles(smiles)
                if mol != None and label != '':
                    smiles_data.append(smiles)
                    if task == 'classification':
                        labels.append(int(label))
                    elif task == 'regression':
                        labels.append(float(label))
                    else:
                        ValueError('task must be either regression or classification')
    print(len(smiles_data))
    return smiles_data, labels


class MolTestDataset(Dataset):
    def __init__(self, data_path, target, task, num_smiles_aug):
        super(Dataset, self).__init__()
        self.smiles_data, self.labels, self.masks = read_smiles(data_path, target, task)  # list
        self.task = task
        self.num_smiles_aug = num_smiles_aug

        self.conversion = 1
        # if 'qm9' in data_path and target in ['homo', 'lumo', 'gap', 'zpve', 'u0']:
        #     self.conversion = 27.211386246
        #     print(target, 'Unit conversion needed!')
        
        # convert smiles to tokens (self.src_tokens, self.src_mask)
        vocab_file = "./data/vocab_share.pk"
        with open(vocab_file, 'rb') as f:
            self.src_itos, self.tgt_itos = pickle.load(f)
        self.src_stoi = {self.src_itos[i]: i for i in range(len(self.src_itos))}
        self.tgt_stoi = {self.tgt_itos[i]: i for i in range(len(self.tgt_itos))}
        
        smiles_tokens = []
        max_len = 0
        augment_labels = []
        augment_masks = []
        for index, smi in enumerate(self.smiles_data):
            #! data augmentation by smiles enumeration? 
            cano_smi = canonical_smiles(smi)
            augment_smiles = [cano_smi]
            src_token = ["<UNK>"] + smi_tokenizer(cano_smi) 
            src_token = [self.src_stoi.get(st, self.src_stoi['<unk>']) for st in src_token]
            smiles_tokens.append(src_token)
            if len(src_token) > max_len:
                max_len = len(src_token)
            augment_labels.append(self.labels[index])
            augment_masks.append(self.masks[index])
            
            num_success = 0
            for aug_time in range(100):
                if "." not in smi:
                    random_smi = randomize_smiles_with_am(smi)
                else:
                    reacts_list = smi.split('.')
                    np.random.shuffle(reacts_list)
                    reacts_list_random = [randomize_smiles_with_am(react) for react in reacts_list]
                    random_smi = '.'.join(reacts_list_random)

                if random_smi not in augment_smiles:
                    augment_smiles.append(random_smi)
                    src_token = ["<UNK>"] + smi_tokenizer(random_smi) 
                    src_token = [self.src_stoi.get(st, self.src_stoi['<unk>']) for st in src_token]
                    smiles_tokens.append(src_token)
                    if len(src_token) > max_len:
                        max_len = len(src_token)
                    augment_labels.append(self.labels[index])
                    augment_masks.append(self.masks[index])
                    num_success += 1
                
                if num_success >= self.num_smiles_aug:
                    break

        pad_token_idx = self.src_stoi["<pad>"] 
        padding_smiles_tokens = np.full((len(smiles_tokens), max_len), pad_token_idx, dtype=np.int64)
        for i, token in enumerate(smiles_tokens):
            padding_smiles_tokens[i, :len(token)] = token
        self.smiles_tokens = padding_smiles_tokens
        self.labels = augment_labels
        self.masks = augment_masks

    def __getitem__(self, index):
        x = self.smiles_tokens[index]
        
        if self.task == 'classification':
            # cross-entropy loss
            # y = torch.tensor(self.labels[index], dtype=torch.long).view(1,-1)
            # BCE loss
            y = torch.tensor(self.labels[index], dtype=torch.float).view(1,-1)
        elif self.task == 'regression':
            y = torch.tensor(self.labels[index] * self.conversion, dtype=torch.float).view(1,-1)

        mask = torch.tensor(self.masks[index], dtype=torch.float).view(-1)

        return x, y, mask

    def __len__(self):
        return len(self.smiles_tokens)


class MolTestDatasetAug(Dataset):
    def __init__(self, data_path, target, task, num_smiles_aug):
        super(Dataset, self).__init__()
        self.smiles_data, self.labels, self.masks = read_smiles(data_path, target, task)  # list
        self.task = task
        self.num_smiles_aug = num_smiles_aug

        self.conversion = 1
        # if 'qm9' in data_path and target in ['homo', 'lumo', 'gap', 'zpve', 'u0']:
        #     self.conversion = 27.211386246
        #     print(target, 'Unit conversion needed!')
        
        # convert smiles to tokens (self.src_tokens, self.src_mask)
        vocab_file = "./data/vocab_share.pk"
        with open(vocab_file, 'rb') as f:
            self.src_itos, self.tgt_itos = pickle.load(f)
        self.src_stoi = {self.src_itos[i]: i for i in range(len(self.src_itos))}
        self.tgt_stoi = {self.tgt_itos[i]: i for i in range(len(self.tgt_itos))}
        
        smiles_tokens_dict = {}
        pad_token_idx = self.src_stoi["<pad>"] 
        for index, smi in enumerate(self.smiles_data):
            max_len = 0
            smiles_tokens = []
            #! data augmentation by smiles enumeration? 
            cano_smi = canonical_smiles(smi)
            augment_smiles = [cano_smi]
            src_token = ["<UNK>"] + smi_tokenizer(cano_smi) 
            src_token = [self.src_stoi.get(st, self.src_stoi['<unk>']) for st in src_token]
            smiles_tokens.append(src_token)
            if len(src_token) > max_len:
                max_len = len(src_token)

            num_success = 0
            for aug_time in range(100):
                if "." not in smi:
                    random_smi = randomize_smiles_with_am(smi)
                else:
                    reacts_list = smi.split('.')
                    np.random.shuffle(reacts_list)
                    reacts_list_random = [randomize_smiles_with_am(react) for react in reacts_list]
                    random_smi = '.'.join(reacts_list_random)

                if random_smi not in augment_smiles:
                    augment_smiles.append(random_smi)
                    src_token = ["<UNK>"] + smi_tokenizer(random_smi) 
                    src_token = [self.src_stoi.get(st, self.src_stoi['<unk>']) for st in src_token]
                    smiles_tokens.append(src_token)
                    if len(src_token) > max_len:
                        max_len = len(src_token)
                    num_success += 1
                
                if num_success >= self.num_smiles_aug:
                    break
                
            padding_smiles_tokens = np.full((len(smiles_tokens), max_len), pad_token_idx, dtype=np.int64)
            for i, token in enumerate(smiles_tokens):
                padding_smiles_tokens[i, :len(token)] = token
            smiles_tokens_dict[index] = padding_smiles_tokens

        self.smiles_tokens = smiles_tokens_dict

    def __getitem__(self, index):
        x = self.smiles_tokens[index]
        
        if self.task == 'classification':
            # cross-entropy loss
            # y = torch.tensor(self.labels[index], dtype=torch.long).view(1,-1)
            # BCE loss
            y = torch.tensor(self.labels[index], dtype=torch.float).view(1,-1)
        elif self.task == 'regression':
            y = torch.tensor(self.labels[index] * self.conversion, dtype=torch.float).view(1,-1)

        mask = torch.tensor(self.masks[index], dtype=torch.float).view(-1)

        return x, y, mask

    def __len__(self):
        return len(self.smiles_tokens)


class MolTestDatasetWrapper(object):
    
    def __init__(self, 
        batch_size, num_workers, valid_size, test_size, 
        data_path, target, task, splitting, num_smiles_aug,
    ):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.target = target
        self.task = task
        self.num_smiles_aug = num_smiles_aug
        self.splitting = splitting
        assert splitting in ['random', 'scaffold', 'pre-defined']

    def get_data_loaders(self):
        if self.splitting == "pre-defined":
            train_dataset = MolTestDataset(data_path=os.path.join(self.data_path, "train.csv"), target=self.target, task=self.task, num_smiles_aug=self.num_smiles_aug)
            valid_dataset = MolTestDataset(data_path=os.path.join(self.data_path, "valid.csv"), target=self.target, task=self.task, num_smiles_aug=self.num_smiles_aug)
            #! test_loader: sample all augmentations of a molecule as a batch
            test_dataset = MolTestDatasetAug(data_path=os.path.join(self.data_path, "test.csv"), target=self.target, task=self.task, num_smiles_aug=self.num_smiles_aug)
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True,
                num_workers=self.num_workers, drop_last=True
            )
            valid_loader = DataLoader(
                valid_dataset, batch_size=self.batch_size,
                num_workers=self.num_workers, drop_last=False
            )
            test_loader = DataLoader(
                test_dataset, batch_size=1, #! change
                num_workers=self.num_workers, drop_last=False
            )
        else:
            train_dataset = MolTestDataset(data_path=self.data_path, target=self.target, task=self.task)
            train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader, test_loader, train_dataset

    def get_train_validation_data_loaders(self, train_dataset):
        if self.splitting == 'random':
            # obtain training indices that will be used for validation
            num_train = len(train_dataset)
            indices = list(range(num_train))
            np.random.shuffle(indices)

            split = int(np.floor(self.valid_size * num_train))
            split2 = int(np.floor(self.test_size * num_train))
            valid_idx, test_idx, train_idx = indices[:split], indices[split:split+split2], indices[split+split2:]
        
        elif self.splitting == 'scaffold': # fixed, no need for random seed
            train_idx, valid_idx, test_idx = scaffold_split(train_dataset, self.valid_size, self.test_size)
 
        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        valid_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        test_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=test_sampler,
            num_workers=self.num_workers, drop_last=False
        )

        return train_loader, valid_loader, test_loader
    

class ReactionDataset(Dataset):
    def __init__(self, reactants, products, vocab_file, task="backward_prediction"):
        super(Dataset, self).__init__()

        self.reactants = reactants
        self.products = products
        self.task = task

        with open(vocab_file, 'rb') as f:
            self.src_itos, self.tgt_itos = pickle.load(f)
        self.src_stoi = {self.src_itos[i]: i for i in range(len(self.src_itos))}
        self.tgt_stoi = {self.tgt_itos[i]: i for i in range(len(self.tgt_itos))}

        self.src_begin_token = "<UNK>"
        self.tgt_begin_token = "<sos>"
        self.tgt_end_token = "<eos>"
        self.mask_token = "<mask>"
        self.pad_token = "<pad>"

        self.src_pad = self.src_stoi[self.pad_token]
        self.tgt_pad = self.tgt_stoi[self.pad_token]
        self.remove_src_token_ids = [self.src_stoi['<UNK>'], self.src_stoi['<pad>']] 
        self.remove_tgt_token_ids = [self.tgt_stoi['<sos>'], self.tgt_stoi['<eos>'], self.tgt_stoi['<pad>']] 
        
    def __len__(self):
        return len(self.reactants)
    
    def __getitem__(self, index):
        reacts = self.reactants[index]
        prods = self.products[index]

        if self.task == "forward_prediction":
            src_token = [self.src_begin_token] + smi_tokenizer(clear_map_number(reacts))
            tgt_token = [self.tgt_begin_token] + smi_tokenizer(clear_map_number(prods)) + [self.tgt_end_token]
        elif self.task == "backward_prediction":
            src_token = [self.src_begin_token] + smi_tokenizer(clear_map_number(prods))
            tgt_token = [self.tgt_begin_token] + smi_tokenizer(clear_map_number(reacts)) + [self.tgt_end_token]  
        src_token = [self.src_stoi.get(st, self.src_stoi['<unk>']) for st in src_token]
        tgt_token = [self.tgt_stoi.get(tt, self.tgt_stoi['<unk>']) for tt in tgt_token]

        return src_token, tgt_token
    
    def _collate_fn(self, data):
        src, tgt = zip(*data)
        max_src_length = max([len(s) for s in src])
        max_tgt_length = max([len(t) for t in tgt])
        batch_size = len(src)
        
        new_src = torch.full((max_src_length, batch_size), self.src_pad, dtype=torch.long)
        new_tgt = torch.full((max_tgt_length, batch_size), self.tgt_pad, dtype=torch.long)
        for i in range(batch_size):
            new_src[:, i][:len(src[i])] = torch.LongTensor(src[i])
            new_tgt[:, i][:len(tgt[i])] = torch.LongTensor(tgt[i])

        return new_src, new_tgt

    def reconstruct_smi(self, tokens, src=True, raw=False):
        if src:
            if raw:
                return [self.src_itos[t] for t in tokens]
            else:
                return [self.src_itos[t] for t in tokens if t not in self.remove_src_token_ids]
        else:
            if raw:
                return [self.tgt_itos[t] for t in tokens]
            else:
                return [self.tgt_itos[t] for t in tokens if t not in self.remove_tgt_token_ids]