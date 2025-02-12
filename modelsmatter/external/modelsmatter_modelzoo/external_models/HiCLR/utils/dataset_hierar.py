import os
import pickle
import lmdb
import pandas as pd
import numpy as np
import math
import random
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from multiprocessing import Pool

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
# from scipy.optimize import curve_fit

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from utils.smiles_utils import smi_tokenizer, clear_map_number, SmilesGraph
from utils.smiles_utils import canonical_smiles, canonical_smiles_with_am, remove_am_without_canonical, \
    extract_relative_mapping, get_nonreactive_mask, randomize_smiles_with_am, get_template_mask


def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp(- (x - mean) ** 2 / (2 * standard_deviation ** 2))


# Dataset --> RetroDataset --> PCLdataset
# change: __init__, parse_smi_wrapper, parse_smi, __getitem__


class RetroDataset(Dataset):
    def __init__(self, mode, data_folder='./data', vocab_folder='./data',
                 known_class=False, shared_vocab=False, augment=False, sample=False, data_file=None,
                 sample_per_class=8, random_state=0):
        self.data_folder = data_folder
        self.data_file = data_file

        assert mode in ['train', 'test', 'val', 'train_val']
        self.BondTypes = ['NONE', 'AROMATIC', 'DOUBLE', 'SINGLE', 'TRIPLE']
        self.bondtoi = {bond: i for i, bond in enumerate(self.BondTypes)}

        self.mode = mode
        self.augment = augment
        self.known_class = known_class
        self.shared_vocab = shared_vocab
        print('Building {} data from: {}'.format(mode, data_folder))
        if shared_vocab:
            vocab_file = 'vocab_share.pk'
        else:
            vocab_file = 'vocab.pk'

        if mode in ["val", "test"]:
            assert vocab_file in os.listdir(vocab_folder)
            with open(os.path.join(vocab_folder, vocab_file), 'rb') as f:
                self.src_itos, self.tgt_itos = pickle.load(f)
            self.src_stoi = {self.src_itos[i]: i for i in range(len(self.src_itos))}
            self.tgt_stoi = {self.tgt_itos[i]: i for i in range(len(self.tgt_itos))}
            if data_file is None:
                self.data = pd.read_csv(os.path.join(data_folder, f'raw_{mode}.csv'))
            else:
                self.data = pd.read_csv(os.path.join(data_folder, f'{data_file}.csv'))
            if sample:
                self.data = self.data.sample(n=200, random_state=0)
                self.data.reset_index(inplace=True, drop=True)
        else:
            if mode == "train":
                train_data = pd.read_csv(os.path.join(data_folder, f'raw_train.csv'))
                val_data = pd.read_csv(os.path.join(data_folder, f'raw_val.csv'))
                raw_data = pd.concat([val_data, train_data])
                raw_data.reset_index(inplace=True, drop=True)
            elif mode == "train_val":
                train_val_data = pd.read_csv(os.path.join(data_folder, 'raw_train_val.csv'))
                raw_data = train_val_data
            # if sample:
            #     train_data = train_data.sample(n=1000, random_state=0)
            #     train_data.reset_index(inplace=True, drop=True)
            #     val_data = val_data.sample(n=200, random_state=0)
            #     val_data.reset_index(inplace=True, drop=True)
            if sample:
                reaction_class = train_data["class"].unique()
                sample_index = np.array([])
                for class_id in reaction_class:
                    class_index = train_data[train_data["class"]==class_id].sample(n=sample_per_class, random_state=random_state).index.values
                    sample_index = np.concatenate([sample_index, class_index])
                train_data = train_data.iloc[sample_index, :].reset_index(drop=True)

            if vocab_file not in os.listdir(vocab_folder):
                print('Building vocab...')

                prods, reacts = self.build_vocab_from_raw_data(raw_data)
                if self.shared_vocab:  # Shared src and tgt vocab
                    itos = set()
                    for i in range(len(prods)):
                        itos.update(smi_tokenizer(prods[i]))
                        itos.update(smi_tokenizer(reacts[i]))
                    # vocab for reaction_class token 
                    itos.update(['<RX_{}>'.format(i) for i in range(1, 11)])
                    itos.add('<UNK>')
                    itos = ['<unused{}'.format(i) for i in range(1, 11)] + sorted(list(itos)) 
                    itos = ['<unk>', '<pad>', '<sos>', '<eos>', '<mask>'] + itos
                    self.src_itos, self.tgt_itos = itos, itos
                else:  # Non-shared src and tgt vocab
                    self.src_itos, self.tgt_itos = set(), set()
                    for i in range(len(prods)):
                        self.src_itos.update(smi_tokenizer(reacts[i]))
                        self.tgt_itos.update(smi_tokenizer(prods[i]))
                    self.src_itos.update(['<RX_{}>'.format(i) for i in range(1, 11)])
                    self.src_itos.add('<UNK>')
                    self.src_itos = ['<unused{}'.format(i) for i in range(1, 11)] + sorted(list(self.src_itos)) 
                    self.src_itos = ['<unk>', '<pad>', '<sos>', '<eos>', '<mask>'] + self.src_itos

                    self.tgt_itos = ['<unused{}'.format(i) for i in range(1, 11)] + sorted(list(self.tgt_itos)) 
                    self.tgt_itos = ['<unk>', '<pad>', '<sos>', '<eos>'] + self.tgt_itos
                    
                self.src_stoi = {self.src_itos[i]: i for i in range(len(self.src_itos))}
                self.tgt_stoi = {self.tgt_itos[i]: i for i in range(len(self.tgt_itos))}

                with open(os.path.join(vocab_folder, vocab_file), 'wb') as f:
                    pickle.dump([self.src_itos, self.tgt_itos], f)
            else:
                with open(os.path.join(vocab_folder, vocab_file), 'rb') as f:
                    self.src_itos, self.tgt_itos = pickle.load(f)

                self.src_stoi = {self.src_itos[i]: i for i in range(len(self.src_itos))}
                self.tgt_stoi = {self.tgt_itos[i]: i for i in range(len(self.tgt_itos))}

            self.data = eval('{}_data'.format(mode))

        # Build and load processed data into lmdb
        sample_suffix = f"_{sample_per_class}" if sample else ""
        self.cooked_file = f'cooked_{self.mode}{sample_suffix}.lmdb' if self.data_file is None else f'cooked_{self.data_file}.lmdb'
        if self.cooked_file not in os.listdir(self.data_folder):
            self.build_processed_data(self.data)
        self.env = lmdb.open(os.path.join(self.data_folder, self.cooked_file),
                             max_readers=1, readonly=True,
                             lock=False, readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            self.product_keys = list(txn.cursor().iternext(values=False))

        self.factor_func = lambda x: (1 + gaussian(x, 5.55391565, 0.27170542, 1.20071279)) 

    def build_vocab_from_raw_data(self, raw_data):
        reactions = raw_data['reactants>reagents>production'].to_list()
        prods, reacts = [], []
        for i in tqdm(range(len(reactions))):
            rxn = reactions[i]
            r, reagents, p = rxn.split('>')
            if not r or not p:
                continue

            src, tgt = self.parse_smi(p, r, '<UNK>', build_vocab=True)
            if Chem.MolFromSmiles(src) is None or Chem.MolFromSmiles(tgt) is None:
                continue
            prods.append(src)
            reacts.append(tgt)
        return prods, reacts

    def build_processed_data(self, raw_data):
        raw_data.reset_index(inplace=True, drop=True)
        reactions = raw_data['reactants>reagents>production'].to_list()

        env = lmdb.open(os.path.join(self.data_folder, self.cooked_file),
                        map_size=1099511627776)

        with env.begin(write=True) as txn:
            for i in tqdm(range(len(reactions))):
                rxn = reactions[i]
                r, reagents, p = rxn.split('>')
                rt = '<RX_{}>'.format(raw_data['class'][i]) if 'class' in raw_data else '<UNK>'
                result = self.parse_smi_wrapper((p, r, rt))
                if result is not None:
                    src, src_graph, tgt, context_align, nonreact_mask = result
                    graph_contents = src_graph.adjacency_matrix, src_graph.bond_type_dict, src_graph.bond_attributes

                    p_key = '{} {}'.format(i, clear_map_number(p))
                    processed = {
                        'src': src,
                        'graph_contents': graph_contents,
                        'tgt': tgt,
                        'context_align': context_align,
                        'nonreact_mask': nonreact_mask,  
                        'raw_product': p,  
                        'raw_reactants': r, 
                        'reaction_class': rt
                    }
                    try:
                        txn.put(p_key.encode(), pickle.dumps(processed))
                    except:
                        print('Error processing index {} and product {}'.format(i, p_key))
                        continue
                else:
                    print('Warning. Process Failed.')

        return

    def parse_smi_wrapper(self, task):
        prod, reacts, react_class = task
        if not prod or not reacts:
            return None
        return self.parse_smi(prod, reacts, react_class, build_vocab=False, randomize=False)

    def parse_smi(self, prod, reacts, react_class, build_vocab=False, randomize=False):
        ''' Process raw atom-mapped product and reactants into model-ready inputs
        :param prod: atom-mapped product
        :param reacts: atom-mapped reactants
        :param react_class: reaction class
        :param build_vocab: whether it is the stage of vocab building
        :param randomize: whether do random permutation of the reaction smiles
        :return:
        '''
        # Process raw prod and reacts:
        cano_prod_am = canonical_smiles_with_am(prod)
        cano_reacts_am = canonical_smiles_with_am(reacts) #! [Au]-->[Au:0]

        cano_prod = clear_map_number(prod)
        cano_reacts = remove_am_without_canonical(cano_reacts_am)  #! [Au:0]-->Au

        if build_vocab:
            return cano_prod, cano_reacts

        if Chem.MolFromSmiles(cano_reacts) is None:
            cano_reacts = clear_map_number(reacts)

        if Chem.MolFromSmiles(cano_prod) is None or Chem.MolFromSmiles(cano_reacts) is None:
            return None

        if randomize:
            # print('permute product')
            cano_prod_am = randomize_smiles_with_am(prod)
            cano_prod = remove_am_without_canonical(cano_prod_am)
            if np.random.rand() > 0.5:
                # print('permute reacts')
                cano_reacts_am = '.'.join(cano_reacts_am.split('.')[::-1])
                cano_reacts = remove_am_without_canonical(cano_reacts_am)

        # Get the smiles graph
        smiles_graph = SmilesGraph(cano_prod)
        # Get the nonreactive masking based on atom-mapping
        gt_nonreactive_mask = get_nonreactive_mask(cano_prod_am, prod, reacts, radius=1)  #! reactive mask for product (changed_atom and its one-hop neighbor)

        # Get the context alignment based on atom-mapping
        position_mapping_list = extract_relative_mapping(cano_prod_am, cano_reacts_am)

        # Note: gt_context_attn.size(0) = tgt.size(0) - 1; attention for token that need to predict
        gt_context_attn = torch.zeros(
            (len(smi_tokenizer(cano_reacts_am)) + 1, len(smi_tokenizer(cano_prod_am)) + 1)).long()
        for i, j in position_mapping_list:
            gt_context_attn[i][j + 1] = 1

        # Prepare model inputs
        src_token = smi_tokenizer(cano_prod)
        tgt_token = ['<sos>'] + smi_tokenizer(cano_reacts) + ['<eos>']
        if self.known_class:
            src_token = [react_class] + src_token
        else:
            src_token = ['<UNK>'] + src_token
        gt_nonreactive_mask = [True] + gt_nonreactive_mask

        src_token = [self.src_stoi.get(st, self.src_stoi['<unk>']) for st in src_token]
        tgt_token = [self.tgt_stoi.get(tt, self.tgt_stoi['<unk>']) for tt in tgt_token]

        return src_token, smiles_graph, tgt_token, gt_context_attn, gt_nonreactive_mask

    def reconstruct_smi(self, tokens, src=True, raw=False):
        if src:
            if raw:
                return [self.src_itos[t] for t in tokens]
            else:
                return [self.src_itos[t] for t in tokens if t != 1]
        else:
            if raw:
                return [self.tgt_itos[t] for t in tokens]
            else:
                return [self.tgt_itos[t] for t in tokens if t not in [1, 2, 3]]

    def __len__(self):
        return len(self.product_keys)

    def __getitem__(self, idx):
        p_key = self.product_keys[idx]
        with self.env.begin(write=False) as txn:
            processed = pickle.loads(txn.get(p_key))
        reaction_id, p_key = p_key.decode().split(' ')  # prod smiles

        p = np.random.rand()
        if self.mode == 'train' and p > 0.5 and self.augment:
            prod = processed['raw_product']
            react = processed['raw_reactants']
            rt = processed['reaction_class']
            try:
                src, src_graph, tgt, context_alignment, nonreact_mask = \
                    self.parse_smi(prod, react, rt, randomize=True)
            except:
                src, graph_contents, tgt, context_alignment, nonreact_mask = \
                    processed['src'], processed['graph_contents'], processed['tgt'], \
                    processed['context_align'], processed['nonreact_mask']
                src_graph = SmilesGraph(p_key, existing=graph_contents)
        else:
            src, graph_contents, tgt, context_alignment, nonreact_mask = \
                processed['src'], processed['graph_contents'], processed['tgt'], \
                processed['context_align'], processed['nonreact_mask']
            src_graph = SmilesGraph(p_key, existing=graph_contents)

        # Make sure the reaction class is known/unknown
        if self.known_class:
            src[0] = self.src_stoi[processed['reaction_class']]
        else:
            src[0] = self.src_stoi['<UNK>']

        try:
            reaction_class = eval(processed['reaction_class'].split("_")[-1][:-1]) 
        except:
            reaction_class = 0.
        
        return src, src_graph, tgt, context_alignment, nonreact_mask, reaction_class, eval(reaction_id)


class SimpleRetroDataset(Dataset):
    def __init__(self, mode, data_folder='./data', vocab_folder='./data',
                 known_class=False, shared_vocab=False, augment=False, sample=False, data_file=None,
                 sample_per_class=8, random_state=0, canonicalize=True):
        self.data_folder = data_folder
        self.data_file = data_file

        assert mode in ['train', 'test', 'val', 'train_val']
        self.BondTypes = ['NONE', 'AROMATIC', 'DOUBLE', 'SINGLE', 'TRIPLE']
        self.bondtoi = {bond: i for i, bond in enumerate(self.BondTypes)}

        self.mode = mode
        self.augment = augment
        self.canonicalize = canonicalize
        self.known_class = known_class
        self.shared_vocab = shared_vocab
        print('Building {} data from: {}'.format(mode, data_folder))

        if shared_vocab:
            vocab_file = 'vocab_share.pk'
        else:
            vocab_file = 'vocab.pk'

        if mode in ["val", "test"]:
            assert vocab_file in os.listdir(vocab_folder)
            with open(os.path.join(vocab_folder, vocab_file), 'rb') as f:
                self.src_itos, self.tgt_itos = pickle.load(f)
            self.src_stoi = {self.src_itos[i]: i for i in range(len(self.src_itos))}
            self.tgt_stoi = {self.tgt_itos[i]: i for i in range(len(self.tgt_itos))}
            if data_file is None:
                self.data = pd.read_csv(os.path.join(data_folder, f'raw_{mode}.csv'))
            else:
                self.data = pd.read_csv(os.path.join(data_folder, f'{data_file}.csv'))
            if sample:
                self.data = self.data.sample(n=200, random_state=0)
                self.data.reset_index(inplace=True, drop=True)
        else:
            if mode == "train":
                train_data = pd.read_csv(os.path.join(data_folder, f'raw_train.csv'))
                val_data = pd.read_csv(os.path.join(data_folder, f'raw_val.csv'))
                raw_data = pd.concat([val_data, train_data])
                raw_data.reset_index(inplace=True, drop=True) # raw_data for building vocab
            elif mode == "train_val":
                train_val_data = pd.read_csv(os.path.join(data_folder, 'raw_train_val.csv'))
                raw_data = train_val_data
            if sample:
                reaction_class = train_data["class"].unique()
                sample_index = np.array([])
                for class_id in reaction_class:
                    class_index = train_data[train_data["class"]==class_id].sample(n=sample_per_class, random_state=random_state).index.values
                    sample_index = np.concatenate([sample_index, class_index])
                train_data = train_data.iloc[sample_index, :].reset_index(drop=True)

            if vocab_file not in os.listdir(vocab_folder):
                print('Building vocab...')

                prods, reacts = self.build_vocab_from_raw_data(raw_data)
                if self.shared_vocab:  # Shared src and tgt vocab
                    itos = set()
                    for i in range(len(prods)):
                        itos.update(smi_tokenizer(prods[i]))
                        itos.update(smi_tokenizer(reacts[i]))
                    # vocab for reaction_class token 
                    itos.update(['<RX_{}>'.format(i) for i in range(1, 11)])
                    itos.add('<UNK>')
                    itos = ['<unused{}'.format(i) for i in range(1, 11)] + sorted(list(itos)) 
                    itos = ['<unk>', '<pad>', '<sos>', '<eos>', '<mask>'] + itos
                    self.src_itos, self.tgt_itos = itos, itos
                else:  # Non-shared src and tgt vocab
                    self.src_itos, self.tgt_itos = set(), set()
                    for i in range(len(prods)):
                        self.src_itos.update(smi_tokenizer(reacts[i]))
                        self.tgt_itos.update(smi_tokenizer(prods[i]))
                    self.src_itos.update(['<RX_{}>'.format(i) for i in range(1, 11)])
                    self.src_itos.add('<UNK>')
                    self.src_itos = ['<unused{}'.format(i) for i in range(1, 11)] + sorted(list(self.src_itos)) 
                    self.src_itos = ['<unk>', '<pad>', '<sos>', '<eos>', '<mask>'] + self.src_itos

                    self.tgt_itos = ['<unused{}'.format(i) for i in range(1, 11)] + sorted(list(self.tgt_itos)) 
                    self.tgt_itos = ['<unk>', '<pad>', '<sos>', '<eos>'] + self.tgt_itos
                    
                self.src_stoi = {self.src_itos[i]: i for i in range(len(self.src_itos))}
                self.tgt_stoi = {self.tgt_itos[i]: i for i in range(len(self.tgt_itos))}

                with open(os.path.join(vocab_folder, vocab_file), 'wb') as f:
                    pickle.dump([self.src_itos, self.tgt_itos], f)
            else:
                with open(os.path.join(vocab_folder, vocab_file), 'rb') as f:
                    self.src_itos, self.tgt_itos = pickle.load(f)

                self.src_stoi = {self.src_itos[i]: i for i in range(len(self.src_itos))}
                self.tgt_stoi = {self.tgt_itos[i]: i for i in range(len(self.tgt_itos))}

            self.data = eval('{}_data'.format(mode))

        if "class" in self.data:
            self.y_mean = self.data["class"].mean()
            self.y_std = self.data["class"].std()
        else:
            self.y_mean = 0.
            self.y_std = 0.

        # Build and load processed data into lmdb
        sample_suffix = f"_{sample_per_class}" if sample else ""
        self.cooked_file = f'cooked_{self.mode}{sample_suffix}.lmdb' if self.data_file is None else f'cooked_{self.data_file}.lmdb'
        if self.cooked_file not in os.listdir(self.data_folder):
            self.build_processed_data(self.data)
        self.env = lmdb.open(os.path.join(self.data_folder, self.cooked_file),
                             max_readers=1, readonly=True,
                             lock=False, readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            self.product_keys = list(txn.cursor().iternext(values=False))

        self.factor_func = lambda x: (1 + gaussian(x, 5.55391565, 0.27170542, 1.20071279)) 

    def build_vocab_from_raw_data(self, raw_data):
        reactions = raw_data['reactants>reagents>production'].to_list()
        prods, reacts = [], []
        for i in tqdm(range(len(reactions))):
            rxn = reactions[i]
            r, reagents, p = rxn.split('>')
            if not r or not p:
                continue

            src, tgt = self.parse_smi(p, r, '<UNK>', build_vocab=True)
            if Chem.MolFromSmiles(src) is None or Chem.MolFromSmiles(tgt) is None:
                continue
            prods.append(src)
            reacts.append(tgt)
        return prods, reacts

    def build_processed_data(self, raw_data):
        raw_data.reset_index(inplace=True, drop=True)
        reactions = raw_data['reactants>reagents>production'].to_list()

        env = lmdb.open(os.path.join(self.data_folder, self.cooked_file),
                        map_size=1099511627776)

        with env.begin(write=True) as txn:
            for i in tqdm(range(len(reactions))):
                rxn = reactions[i]
                r, reagents, p = rxn.split('>')
                rt = '<RX_{}>'.format(raw_data['class'][i]) if 'class' in raw_data else '<UNK>'
                result = self.parse_smi_wrapper((p, r, rt))
                if result is not None:
                    src, tgt, context_align = result

                    p_key = '{} {}'.format(i, clear_map_number(p))
                    processed = {
                        'src': src,
                        'tgt': tgt,
                        'raw_product': p,  
                        'raw_reactants': r, 
                        'reaction_class': rt,
                        'reagents': reagents,
                        'context_align': context_align, 
                    }
                    try:
                        txn.put(p_key.encode(), pickle.dumps(processed))
                    except:
                        print('Error processing index {} and product {}'.format(i, p_key))
                        continue
                else:
                    print('Warning. Process Failed.')

        return

    def parse_smi_wrapper(self, task):
        prod, reacts, react_class = task
        if not prod or not reacts:
            return None
        return self.parse_smi(prod, reacts, react_class, build_vocab=False, randomize=False)

    def parse_smi(self, prod, reacts, react_class, build_vocab=False, 
                  canonicalize=True, randomize=False, react_permute_mode="order"):
        ''' Process raw atom-mapped product and reactants into model-ready inputs
        :param prod: atom-mapped product
        :param reacts: atom-mapped reactants
        :param react_class: reaction class
        :param build_vocab: whether it is the stage of vocab building
        :param randomize: whether do random permutation of the reaction smiles
        :return:
        '''
        # Process raw prod and reacts
        #??? Question: the atom order in cano_prod_am&cano_prod, cano_reacts_am&cano_reacts must be the same
        cano_prod_am = canonical_smiles_with_am(prod)
        cano_reacts_am = canonical_smiles_with_am(reacts) #! [Au]-->[Au:0]

        cano_prod = clear_map_number(prod) #! same as remove_am_without_canonical(cano_prod_am) 
        cano_reacts = remove_am_without_canonical(cano_reacts_am)  #! [Au:0]-->Au

        if build_vocab:
            return cano_prod, cano_reacts

        if Chem.MolFromSmiles(cano_reacts) is None:
            cano_reacts = clear_map_number(reacts)

        if Chem.MolFromSmiles(cano_prod) is None or Chem.MolFromSmiles(cano_reacts) is None:
            return None

        #! data augmentation on-the-fly
        if randomize:
            # print('permute product')
            cano_prod_am = randomize_smiles_with_am(prod)
            cano_prod = remove_am_without_canonical(cano_prod_am)
            
            if np.random.rand() > 0.5:
                # print('permute reactants')
                if react_permute_mode == "all":
                    reacts_list = reacts.split('.')
                    np.random.shuffle(reacts_list)
                    reacts_list_random = [randomize_smiles_with_am(react) for react in reacts_list]
                    cano_reacts_am = '.'.join(reacts_list_random)
                    cano_reacts = remove_am_without_canonical(cano_reacts_am)
                elif react_permute_mode == "order":
                    cano_reacts_am = '.'.join(cano_reacts_am.split('.')[::-1]) #* randomize order of reacts
                    cano_reacts = remove_am_without_canonical(cano_reacts_am)
        
        elif not canonicalize:
            cano_reacts_am = reacts
            cano_reacts = remove_am_without_canonical(cano_reacts_am)
            cano_prod_am = prod
            cano_prod = remove_am_without_canonical(cano_prod_am)

        # Prepare model inputs
        src_token = smi_tokenizer(cano_prod)
        tgt_token = ['<sos>'] + smi_tokenizer(cano_reacts) + ['<eos>']
        if self.known_class:
            src_token = [react_class] + src_token
        else:
            src_token = ['<UNK>'] + src_token

        src_token = [self.src_stoi.get(st, self.src_stoi['<unk>']) for st in src_token]
        tgt_token = [self.tgt_stoi.get(tt, self.tgt_stoi['<unk>']) for tt in tgt_token]

        assert len(smi_tokenizer(cano_prod_am)) == len(smi_tokenizer(cano_prod))
        assert len(smi_tokenizer(cano_reacts_am)) == len(smi_tokenizer(cano_reacts))

        # Get the context alignment based on atom-mapping
        position_mapping_list = extract_relative_mapping(cano_prod_am, cano_reacts_am)

        # Note: gt_context_attn.size(0) = tgt.size(0) - 1, only has attention for tokens that need to be predicted (exclude the start token <sos>)
        gt_context_attn = torch.zeros(
            (len(smi_tokenizer(cano_reacts_am)) + 1, len(smi_tokenizer(cano_prod_am)) + 1)).long()
        for i, j in position_mapping_list:
            gt_context_attn[i][j + 1] = 1

        return src_token, tgt_token, gt_context_attn

    def reconstruct_smi(self, tokens, src=True, raw=False):
        if src:
            if raw:
                return [self.src_itos[t] for t in tokens]
            else:
                return [self.src_itos[t] for t in tokens if t != 1]
        else:
            if raw:
                return [self.tgt_itos[t] for t in tokens]
            else:
                return [self.tgt_itos[t] for t in tokens if t not in [1, 2, 3]]

    def __len__(self):
        return len(self.product_keys)

    def __getitem__(self, idx):
        p_key = self.product_keys[idx]
        with self.env.begin(write=False) as txn:
            processed = pickle.loads(txn.get(p_key))
        reaction_id, p_key = p_key.decode().split(' ')

        p = np.random.rand()
        #! training set augmentation is not sufficient?
        if self.mode == 'train' and p > 0.5 and self.augment: 
            prod = processed['raw_product']
            react = processed['raw_reactants']
            rt = processed['reaction_class']
            try:
                src, tgt, context_align = self.parse_smi(prod, react, rt, randomize=True, react_permute_mode="all")
            except:
                src, tgt = processed['src'], processed['tgt']
                context_align = processed['context_align']
        elif not self.canonicalize:
            prod = processed['raw_product']
            react = processed['raw_reactants']
            rt = processed['reaction_class']
            src, tgt, context_align = self.parse_smi(prod, react, rt, canonicalize=False, randomize=False, react_permute_mode="all")
        else:
            src, tgt = processed['src'], processed['tgt']
            context_align = processed['context_align']

        # Make sure the reaction class is known/unknown
        if self.known_class:
            src[0] = self.src_stoi[processed['reaction_class']]
        else:
            src[0] = self.src_stoi['<UNK>']

        try:
            reaction_class = eval(processed['reaction_class'].split("_")[-1][:-1]) 
        except:
            reaction_class = 0.

        reagents = processed.get('reagents', "")
        if reagents != "":
            #TODO: get fingerprint for each condition
            # reagents = reagents.split(".")
            # reagents_fp = [list(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi),2,512).ToBitString()) for smi in reagents]
            reagents_fp = list(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(reagents),2,512).ToBitString())
            reagents_fp = np.array(reagents_fp, dtype=float).reshape(1, -1)

            # reagents_token = smi_tokenizer(reagents)
            # reagents_token_id = [self.tgt_stoi.get(tt, self.tgt_stoi['<unk>']) for tt in reagents_token]
            # tgt = tgt[:-1] + reagents_token_id + [tgt[-1]] #! add <sep> token

            return src, tgt, reaction_class, eval(reaction_id), torch.tensor(reagents_fp, dtype=torch.float), context_align
        else:
            return src, tgt, reaction_class, eval(reaction_id), reagents, context_align


class PCLRetroDataset(RetroDataset):
    def __init__(self, mode, data_folder='./data',
                 known_class=False, shared_vocab=False, augment=False, sample=False, data_file=None,
                 sample_per_class=8, random_state=0):
        super().__init__(mode, data_folder, known_class, shared_vocab, augment, sample, data_file, sample_per_class, random_state)
        self.labels = {}
        self.data.reset_index(inplace=True, drop=True)
        print("----------Start building hierarchical label trees----------")
        for i in tqdm(range(len(self.data))):
            rxn = self.data["reactants>reagents>production"][i]  # with atom-mapping
            rxn_superclass = self.data["rxn_superclass"][i]
            template_label = self.data["template_label"][i]
            rxn_index = self.data["index"][i]
            if rxn_superclass not in self.labels:
                self.labels[rxn_superclass] = {}
            if template_label not in self.labels[rxn_superclass]:
                self.labels[rxn_superclass][template_label] = {}
            self.labels[rxn_superclass][template_label][rxn_index] = i

    def build_processed_data(self, raw_data):
        raw_data.reset_index(inplace=True, drop=True)
        reactions = raw_data['reactants>reagents>production'].to_list()

        env = lmdb.open(os.path.join(self.data_folder, 'cooked_{}.lmdb'.format(self.mode)),
                        map_size=1099511627776)

        with env.begin(write=True) as txn:
            for i in tqdm(range(len(reactions))):
                rxn = reactions[i]
                r, reagents, p = rxn.split('>')
                rt = '<RX_{}>'.format(raw_data['class'][i]) if 'class' in raw_data else '<UNK>'
                # retro_template = raw_data["retro_template"][i]
                rxn_superclass = int(raw_data["rxn_superclass"][i])
                template_label = int(raw_data["template_label"][i])
                rxn_index = int(raw_data["index"][i])
                result = self.parse_smi_wrapper((p, r, rt))
                if result is not None:
                    src, tgt, context_align = result

                    # p_key = '{}'.format(i)
                    # p_key = '{}'.format(clear_map_number(p))
                    p_key = "{:06d}".format(i)  # 按照数据顺序存储
                    processed = {
                        'src': src,
                        'tgt': tgt,
                        'raw_product': p,  
                        'raw_reactants': r, 
                        'class_token': rt,   # str
                        # "retro_template": retro_template,  
                        "rxn_superclass": rxn_superclass, 
                        "template_label": template_label, 
                        "rxn_index": rxn_index,
                        'context_align': context_align,
                    }
                    try:
                        txn.put(p_key.encode(), pickle.dumps(processed))
                    except:
                        print('Error processing index {} and product {}'.format(i, p))
                        continue
                else:
                    print('Warning. Process Failed.')

        return

    def parse_smi_wrapper(self, task):
        prod, reacts, react_class = task
        if not prod or not reacts:
            return None
        return self.parse_smi(prod, reacts, react_class, build_vocab=False, randomize=False)

    def parse_smi(self, prod, reacts, react_class, build_vocab=False, randomize=False, react_permute_mode="order"): 
        ''' Process raw atom-mapped product and reactants into model-ready inputs
        :param prod: atom-mapped product
        :param reacts: atom-mapped reactants
        :param react_class: reaction class
        :param build_vocab: whether it is the stage of vocab building
        :param randomize: whether do random permutation of the reaction smiles
        :return:
        '''
        # Process raw prod and reacts:
        cano_prod_am = canonical_smiles_with_am(prod)
        cano_reacts_am = canonical_smiles_with_am(reacts)

        cano_prod = clear_map_number(prod)
        cano_reacts = remove_am_without_canonical(cano_reacts_am)

        if build_vocab:
            return cano_prod, cano_reacts

        if Chem.MolFromSmiles(cano_reacts) is None:
            cano_reacts = clear_map_number(reacts)

        if Chem.MolFromSmiles(cano_prod) is None or Chem.MolFromSmiles(cano_reacts) is None:
            return None

        if randomize:
            # print('permute product')
            cano_prod_am = randomize_smiles_with_am(prod)   #! bug: permute后"()"的数量增加-->token length is different
            cano_prod = remove_am_without_canonical(cano_prod_am)

            if np.random.rand() > 0.5:
                # print('permute reacts')
                if react_permute_mode == "all":
                    reacts_list = reacts.split('.')
                    np.random.shuffle(reacts_list)
                    reacts_list_random = [randomize_smiles_with_am(react) for react in reacts_list]
                    cano_reacts_am = '.'.join(reacts_list_random)
                    cano_reacts = remove_am_without_canonical(cano_reacts_am)
                elif react_permute_mode == "order":
                    cano_reacts_am = '.'.join(cano_reacts_am.split('.')[::-1]) #* randomize order of reacts
                    cano_reacts = remove_am_without_canonical(cano_reacts_am)

        #TODO: extract reaction center from retro_template (src_template_mask, tgt_template_mask)
        # prod_template_mask, reacts_template_mask = get_template_mask(cano_prod_am, cano_reacts_am, retro_template) #! rely on atom-mapping number
        # src_template_mask = [False] + prod_template_mask
        # tgt_template_mask = [False] + reacts_template_mask + [False]

        # Prepare model inputs
        src_token = smi_tokenizer(cano_prod)
        tgt_token = ['<sos>'] + smi_tokenizer(cano_reacts) + ['<eos>']
        if self.known_class:
            src_token = [react_class] + src_token
        else:
            src_token = ['<UNK>'] + src_token

        src_token = [self.src_stoi.get(st, self.src_stoi['<unk>']) for st in src_token]
        tgt_token = [self.tgt_stoi.get(tt, self.tgt_stoi['<unk>']) for tt in tgt_token]

        # Get the context alignment based on atom-mapping
        position_mapping_list = extract_relative_mapping(cano_prod_am, cano_reacts_am)

        # Note: gt_context_attn.size(0) = tgt.size(0) - 1; attention for token that need to predict
        gt_context_attn = torch.zeros(
            (len(smi_tokenizer(cano_reacts_am)) + 1, len(smi_tokenizer(cano_prod_am)) + 1)).long()
        for i, j in position_mapping_list:
            gt_context_attn[i][j + 1] = 1

        return src_token, tgt_token, gt_context_attn

    def __getitem__(self, idxs): 
        src_ls, tgt_ls = [], []
        src_permute_1_ls, tgt_permute_1_ls, src_template_mask_1_ls, tgt_template_mask_1_ls = [], [], [], []
        src_permute_2_ls, tgt_permute_2_ls, src_template_mask_2_ls, tgt_template_mask_2_ls = [], [], [], []
        context_align_ls = []
        reaction_class_ls = []

        if not isinstance(idxs, list):
            idxs = [idxs]

        for idx in idxs:
            p_key = self.product_keys[idx]
            with self.env.begin(write=False) as txn:
                processed = pickle.loads(txn.get(p_key))

            src, tgt = processed['src'], processed['tgt']
            context_align = processed['context_align']
            prod = processed['raw_product']
            react = processed['raw_reactants']
            rt = processed['class_token']
            retro_template = processed['retro_template']
            # src_permute_1, tgt_permute_1, src_template_mask_1, tgt_template_mask_1 = \
            #     self.parse_smi(prod, react, rt, retro_template, randomize=True, react_permute_mode="all")
            # src_permute_2, tgt_permute_2, src_template_mask_2, tgt_template_mask_2 = \
            #     self.parse_smi(prod, react, rt, retro_template, randomize=True, react_permute_mode="all")

            src_permute_1, tgt_permute_1, _ = \
                self.parse_smi(prod, react, rt, retro_template, randomize=True, react_permute_mode="all")
            src_permute_2, tgt_permute_2, _ = \
                self.parse_smi(prod, react, rt, retro_template, randomize=True, react_permute_mode="all")

            # TODO: reverse products and precursors (contain reagents)

            # Make sure the reaction class is known/unknown
            if self.known_class:
                src[0] = self.src_stoi[processed['class_token']]
                src_permute_1[0] = self.src_stoi[processed['class_token']]
                src_permute_2[0] = self.src_stoi[processed['class_token']]
            else:
                src[0] = self.src_stoi['<UNK>']
                src_permute_1[0] = self.src_stoi['<UNK>']
                src_permute_2[0] = self.src_stoi['<UNK>']
            
            # reaction_class = eval(processed['class_token'].split("_")[-1][:-1])  # int
            reaction_class = list(self.get_label_by_index(idx))

            if len(idxs) > 1:
                src_ls.append(src)
                tgt_ls.append(tgt)
                context_align_ls.append(context_align)
                src_permute_1_ls.append(src_permute_1)
                tgt_permute_1_ls.append(tgt_permute_1)
                src_permute_2_ls.append(src_permute_2)
                tgt_permute_2_ls.append(tgt_permute_2)
                # src_template_mask_1_ls.append(src_template_mask_1)
                # tgt_template_mask_1_ls.append(tgt_template_mask_1)
                # src_template_mask_2_ls.append(src_template_mask_2)
                # tgt_template_mask_2_ls.append(tgt_template_mask_2)
                reaction_class_ls.append(reaction_class)
            else:
                # return src, tgt, src_permute_1, tgt_permute_1, src_template_mask_1, tgt_template_mask_1, src_permute_2, tgt_permute_2, src_template_mask_2, tgt_template_mask_2, reaction_class
                return src, tgt, context_align, src_permute_1, tgt_permute_1, src_permute_2, tgt_permute_2, reaction_class
        return src_ls, tgt_ls, context_align_ls, src_permute_1_ls, tgt_permute_1_ls, src_permute_2_ls, tgt_permute_2_ls, reaction_class_ls

    def random_sample(self, label, label_dict):
        curr_dict = label_dict
        top_level = True
        #all sub trees end with an int index (reaction_id)
        while type(curr_dict) is not int:
            if top_level:
                random_label = label #! top level need to be different
                if len(curr_dict.keys()) != 1:  # num_item in curr_dict 
                    while (random_label == label):
                        random_label = random.sample(curr_dict.keys(), 1)[0]
            else:
                random_label = random.sample(curr_dict.keys(), 1)[0]
            curr_dict = curr_dict[random_label]
            top_level = False
        return curr_dict  # int (sample index)

    def get_label_by_index(self, idx):
        p_key = self.product_keys[idx]
        with self.env.begin(write=False) as txn:
            processed = pickle.loads(txn.get(p_key))
        
        return processed["rxn_superclass"], processed["template_label"], processed["rxn_index"]  # all labels must be the same type (int)


class ForwardDataset(RetroDataset):
    def __init__(self, mode, data_folder='./data', vocab_folder='./data', data_file=None,
                 known_class=False, shared_vocab=False, augment=False, sample=False, 
                 sample_per_class=8, random_state=0):
        self.data_folder = data_folder
        self.data_file = data_file

        assert mode in ['train', 'test', 'val', 'train_val']
        self.BondTypes = ['NONE', 'AROMATIC', 'DOUBLE', 'SINGLE', 'TRIPLE']
        self.bondtoi = {bond: i for i, bond in enumerate(self.BondTypes)}

        self.mode = mode
        self.augment = augment
        self.known_class = known_class
        self.shared_vocab = shared_vocab
        print('Building {} data from: {}'.format(mode, data_folder))

        if shared_vocab:
            vocab_file = 'vocab_share.pk'
        else:
            vocab_file = 'vocab.pk'

        if mode in ["val", "test"]:
            assert vocab_file in os.listdir(vocab_folder)
            with open(os.path.join(vocab_folder, vocab_file), 'rb') as f:
                self.src_itos, self.tgt_itos = pickle.load(f)

            self.src_stoi = {self.src_itos[i]: i for i in range(len(self.src_itos))}
            self.tgt_stoi = {self.tgt_itos[i]: i for i in range(len(self.tgt_itos))}
            if data_file is None:
                self.data = pd.read_csv(os.path.join(data_folder, 'raw_{}.csv'.format(mode)))
            else:
                self.data = pd.read_csv(os.path.join(data_folder, f'{data_file}.csv'))
            if sample:
                self.data = self.data.sample(n=200, random_state=0)
                self.data.reset_index(inplace=True, drop=True)
        else:
            if mode == "train":
                train_data = pd.read_csv(os.path.join(data_folder, 'raw_train.csv'))
                val_data = pd.read_csv(os.path.join(data_folder, 'raw_val.csv'))
                raw_data = pd.concat([val_data, train_data])
                raw_data.reset_index(inplace=True, drop=True)
            elif mode == "train_val":
                train_val_data = pd.read_csv(os.path.join(data_folder, 'raw_train_val.csv'))
                raw_data = train_val_data
            self.raw_data = raw_data
            if sample:
                reaction_class = train_data["class"].unique()
                sample_index = np.array([])
                for class_id in reaction_class:
                    class_index = train_data[train_data["class"]==class_id].sample(n=sample_per_class, random_state=random_state).index.values
                    sample_index = np.concatenate([sample_index, class_index])
                train_data = train_data.iloc[sample_index, :].reset_index(drop=True)
                
                # train_data = train_data.sample(n=1000, random_state=0)
                # train_data.reset_index(inplace=True, drop=True)
                # val_data = val_data.sample(n=200, random_state=0)
                # val_data.reset_index(inplace=True, drop=True)
                
            if vocab_file not in os.listdir(vocab_folder):
                print('Building vocab...')

                prods, reacts = self.build_vocab_from_raw_data(raw_data)
                if self.shared_vocab:  # Shared src and tgt vocab
                    itos = set()
                    for i in range(len(prods)):
                        itos.update(smi_tokenizer(prods[i]))
                        itos.update(smi_tokenizer(reacts[i]))
                    # vocab for reaction_class token 
                    itos.update(['<RX_{}>'.format(i) for i in range(1, 11)])
                    itos.add('<UNK>')
                    itos = ['<unused{}'.format(i) for i in range(1, 11)] + sorted(list(itos)) 
                    itos = ['<unk>', '<pad>', '<sos>', '<eos>', '<mask>'] + itos
                    self.src_itos, self.tgt_itos = itos, itos
                else:  # Non-shared src and tgt vocab
                    self.src_itos, self.tgt_itos = set(), set()
                    for i in range(len(prods)):
                        self.src_itos.update(smi_tokenizer(reacts[i]))
                        self.tgt_itos.update(smi_tokenizer(prods[i]))
                    self.src_itos.update(['<RX_{}>'.format(i) for i in range(1, 11)])
                    self.src_itos.add('<UNK>')
                    self.src_itos = ['<unused{}'.format(i) for i in range(1, 11)] + sorted(list(self.src_itos)) 
                    self.src_itos = ['<unk>', '<pad>', '<sos>', '<eos>', '<mask>'] + self.src_itos

                    self.tgt_itos = ['<unused{}'.format(i) for i in range(1, 11)] + sorted(list(self.tgt_itos)) 
                    self.tgt_itos = ['<unk>', '<pad>', '<sos>', '<eos>'] + self.tgt_itos

                self.src_stoi = {self.src_itos[i]: i for i in range(len(self.src_itos))}
                self.tgt_stoi = {self.tgt_itos[i]: i for i in range(len(self.tgt_itos))}

                with open(os.path.join(vocab_folder, vocab_file), 'wb') as f:
                    pickle.dump([self.src_itos, self.tgt_itos], f)
            else:
                with open(os.path.join(vocab_folder, vocab_file), 'rb') as f:
                    self.src_itos, self.tgt_itos = pickle.load(f)  # list

                self.src_stoi = {self.src_itos[i]: i for i in range(len(self.src_itos))}
                self.tgt_stoi = {self.tgt_itos[i]: i for i in range(len(self.tgt_itos))}

            self.data = eval('{}_data'.format(mode))
            
        self.y_mean = self.data["class"].mean()
        self.y_std = self.data["class"].std()

        # Build and load processed data into lmdb
        sample_suffix = f"_{sample_per_class}" if sample else ""
        self.cooked_file = f'cooked_{self.mode}{sample_suffix}.lmdb' if self.data_file is None else f'cooked_{self.data_file}.lmdb'
        if self.cooked_file not in os.listdir(self.data_folder):
            self.build_processed_data(self.data)
        self.env = lmdb.open(os.path.join(self.data_folder, self.cooked_file),
                             max_readers=1, readonly=True,
                             lock=False, readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            self.product_keys = list(txn.cursor().iternext(values=False))

        self.factor_func = lambda x: (1 + gaussian(x, 5.55391565, 0.27170542, 1.20071279)) 
    
    def build_vocab_from_raw_data(self, raw_data):
        reactions = raw_data['reactants>reagents>production'].to_list()
        prods, reacts = [], []
        for i in tqdm(range(len(reactions))):
            rxn = reactions[i]
            r, reagents, p = rxn.split('>')
            if not r or not p:
                continue
            
            if len(reagents) > 0:
                r = ".".join([r, reagents])

            src, tgt = self.parse_smi(p, r, '<UNK>', build_vocab=True) #* remove atom-mapping number in raw_data
            if Chem.MolFromSmiles(src) is None or Chem.MolFromSmiles(tgt) is None:
                continue
            prods.append(src)
            reacts.append(tgt)
        return prods, reacts

    def build_processed_data(self, raw_data):
        raw_data.reset_index(inplace=True, drop=True)
        reactions = raw_data['reactants>reagents>production'].to_list()

        env = lmdb.open(os.path.join(self.data_folder, self.cooked_file),
                        map_size=1099511627776)

        with env.begin(write=True) as txn:
            for i in tqdm(range(len(reactions))):
                rxn = reactions[i]
                r, reagents, p = rxn.split('>')
                if len(reagents) > 0:
                    r = ".".join([r, reagents])
                rt = '<RX_{}>'.format(raw_data['class'][i]) if 'class' in raw_data else '<UNK>'
                result = self.parse_smi_wrapper((p, r, rt))
                if result is not None:
                    src, src_graph, tgt, context_align, nonreact_mask = result
                    graph_contents = src_graph.adjacency_matrix, src_graph.bond_type_dict, src_graph.bond_attributes

                    p_key = '{} {}'.format(i, clear_map_number(p))
                    processed = {
                        'src': src,
                        'graph_contents': graph_contents,
                        'tgt': tgt,
                        'context_align': context_align,
                        'nonreact_mask': nonreact_mask,  
                        'raw_product': p,  
                        'raw_reactants': r, 
                        'reaction_class': rt,
                    }
                    try:
                        txn.put(p_key.encode(), pickle.dumps(processed))
                    except:
                        print('Error processing index {} and product {}'.format(i, p_key))
                        continue
                else:
                    print('Warning. Process Failed.')

        return
    
    def parse_smi(self, prod, reacts, react_class, build_vocab=False, randomize=False):  
        ''' Process raw atom-mapped product and reactants into model-ready inputs
        :param prod: atom-mapped product
        :param reacts: atom-mapped reactants
        :param react_class: reaction class
        :param build_vocab: whether it is the stage of vocab building
        :param randomize: whether do random permutation of the reaction smiles
        :return:
        '''
        # Process raw prod and reacts:
        cano_prod_am = canonical_smiles_with_am(prod)
        cano_reacts_am = canonical_smiles_with_am(reacts)

        cano_prod = clear_map_number(prod)
        cano_reacts = remove_am_without_canonical(cano_reacts_am)

        if build_vocab:
            return cano_prod, cano_reacts

        if Chem.MolFromSmiles(cano_reacts) is None:
            cano_reacts = clear_map_number(reacts)

        if Chem.MolFromSmiles(cano_prod) is None or Chem.MolFromSmiles(cano_reacts) is None:
            return None

        if randomize:
            # print('permute product')
            cano_prod_am = randomize_smiles_with_am(prod)
            cano_prod = remove_am_without_canonical(cano_prod_am)
            if np.random.rand() > 0.5:
                # print('permute reacts')
                cano_reacts_am = '.'.join(cano_reacts_am.split('.')[::-1])
                cano_reacts = remove_am_without_canonical(cano_reacts_am)

        # Get the smiles graph
        smiles_graph = SmilesGraph(cano_prod)
        # Get the nonreactive masking based on atom-mapping
        gt_nonreactive_mask = get_nonreactive_mask(cano_prod_am, prod, reacts, radius=1) 

        # Get the context alignment based on atom-mapping
        position_mapping_list = extract_relative_mapping(cano_prod_am, cano_reacts_am)

        # Note: gt_context_attn.size(0) = tgt.size(0) - 1; attention for token that need to predict
        gt_context_attn = torch.zeros(
            (len(smi_tokenizer(cano_reacts_am)) + 1, len(smi_tokenizer(cano_prod_am)) + 1)).long()
        for i, j in position_mapping_list:
            gt_context_attn[i][j + 1] = 1

        # Prepare model inputs
        src_token = smi_tokenizer(cano_reacts)
        tgt_token = ['<sos>'] + smi_tokenizer(cano_prod) + ['<eos>']

        if self.known_class:
            src_token = [react_class] + src_token
        else:
            src_token = ['<UNK>'] + src_token
        gt_nonreactive_mask = [True] + gt_nonreactive_mask

        src_token = [self.src_stoi.get(st, self.src_stoi['<unk>']) for st in src_token]
        tgt_token = [self.tgt_stoi.get(tt, self.tgt_stoi['<unk>']) for tt in tgt_token]

        return src_token, smiles_graph, tgt_token, gt_context_attn, gt_nonreactive_mask
    

class PCLForwardDataset(PCLRetroDataset):
    def build_vocab_from_raw_data(self, raw_data):
        reactions = raw_data['reactants>reagents>production'].to_list()
        prods, reacts = [], []
        for i in tqdm(range(len(reactions))):
            rxn = reactions[i]
            r, reagents, p = rxn.split('>')
            if not r or not p:
                continue
            
            if len(reagents) > 0:
                r = ".".join([r, reagents])

            src, tgt = self.parse_smi(p, r, '<UNK>', build_vocab=True)
            if Chem.MolFromSmiles(src) is None or Chem.MolFromSmiles(tgt) is None:
                continue
            prods.append(src)
            reacts.append(tgt)
        return prods, reacts
    
    def parse_smi(self, prod, reacts, react_class, retro_template=None, build_vocab=False, randomize=False): 
        ''' Process raw atom-mapped product and reactants into model-ready inputs
        :param prod: atom-mapped product
        :param reacts: atom-mapped reactants
        :param react_class: reaction class
        :param build_vocab: whether it is the stage of vocab building
        :param randomize: whether do random permutation of the reaction smiles
        :return:
        '''
        # Process raw prod and reacts:
        cano_prod_am = canonical_smiles_with_am(prod)
        cano_reacts_am = canonical_smiles_with_am(reacts)  # atom-mapping number from small to large

        cano_prod = clear_map_number(prod)
        cano_reacts = remove_am_without_canonical(cano_reacts_am)

        if build_vocab:
            return cano_prod, cano_reacts

        if Chem.MolFromSmiles(cano_reacts) is None:
            cano_reacts = clear_map_number(reacts)

        if Chem.MolFromSmiles(cano_prod) is None or Chem.MolFromSmiles(cano_reacts) is None:
            return None

        if randomize:
            # print('permute product')
            cano_prod_am = randomize_smiles_with_am(prod)   #! bug: permute后"()"的数量增加-->token length is different
            cano_prod = remove_am_without_canonical(cano_prod_am)
            if np.random.rand() > 0.5:
                # print('permute reacts')
                # cano_reacts_am = '.'.join(cano_reacts_am.split('.')[::-1])
                reacts_list = reacts.split('.')
                np.random.shuffle(reacts_list)
                reacts_list_random = [randomize_smiles_with_am(react) for react in reacts_list]
                cano_reacts_am = '.'.join(reacts_list_random)

                cano_reacts = remove_am_without_canonical(cano_reacts_am)

        #TODO: extract reaction center from retro_template (src_template_mask, tgt_template_mask)
        prod_template_mask, reacts_template_mask = get_template_mask(cano_prod_am, cano_reacts_am, retro_template)
        src_template_mask = [False] + reacts_template_mask
        tgt_template_mask = [False] + prod_template_mask + [False]

        # Prepare model inputs
        src_token = smi_tokenizer(cano_reacts)
        tgt_token = ['<sos>'] + smi_tokenizer(cano_prod) + ['<eos>']

        if self.known_class:
            src_token = [react_class] + src_token
        else:
            src_token = ['<UNK>'] + src_token

        src_token = [self.src_stoi.get(st, self.src_stoi['<unk>']) for st in src_token]
        tgt_token = [self.tgt_stoi.get(tt, self.tgt_stoi['<unk>']) for tt in tgt_token]

        return src_token, tgt_token, src_template_mask, tgt_template_mask
    
    def build_processed_data(self, raw_data):
        raw_data.reset_index(inplace=True, drop=True)
        reactions = raw_data['reactants>reagents>production'].to_list()

        env = lmdb.open(os.path.join(self.data_folder, 'cooked_{}.lmdb'.format(self.mode)),
                        map_size=1099511627776)

        with env.begin(write=True) as txn:
            for i in tqdm(range(len(reactions))):
                rxn = reactions[i]
                r, reagents, p = rxn.split('>')
                if len(reagents) > 0:
                    r = ".".join([r, reagents])
                rt = '<RX_{}>'.format(raw_data['class'][i]) if 'class' in raw_data else '<UNK>'
                retro_template = raw_data["retro_template"][i]
                result = self.parse_smi_wrapper((p, r, rt, retro_template))
                if result is not None:
                    src, tgt, src_template_mask, tgt_template_mask = result

                    p_key = '{}'.format(i)
                    processed = {
                        'src': src,
                        'tgt': tgt,
                        'raw_product': p,  
                        'raw_reactants': r, 
                        'reaction_class': rt, 
                        "retro_template": retro_template,  # for randomize smiles in training
                        "src_template_mask": src_template_mask,
                        "tgt_template_mask": tgt_template_mask
                    }
                    try:
                        txn.put(p_key.encode(), pickle.dumps(processed))
                    except:
                        print('Error processing index {} and product {}'.format(i, p))
                        continue
                else:
                    print('Warning. Process Failed.')

        return


class YieldDataset(SimpleRetroDataset):

    def build_processed_data(self, raw_data):
        raw_data.reset_index(inplace=True, drop=True)
        rxn_col = 'reactants>reagents>production'
        # cond_cols = ["ligand_smiles", "solvent_smiles", "base_smiles"]
        cond_cols = list(raw_data.drop([rxn_col, "class"], axis=1).columns)
        print("Columns of reaction conditions:", cond_cols)

        env = lmdb.open(os.path.join(self.data_folder, self.cooked_file),
                        map_size=1099511627776)

        with env.begin(write=True) as txn:
            for i, row in tqdm(raw_data.iterrows()):
                rxn = row[rxn_col]
                r, _, p = rxn.split('>')
                rt = '<RX_{}>'.format(raw_data['class'][i]) if 'class' in raw_data else '<UNK>'
                result = self.parse_smi_wrapper((p, r, rt))
                reagents = [row[c] for c in cond_cols]
                if result is not None:
                    src, tgt, context_align = result

                    p_key = '{} {}'.format(i, clear_map_number(p))
                    processed = {
                        'src': src,
                        'tgt': tgt,
                        'raw_product': p,  
                        'raw_reactants': r, 
                        'reaction_class': rt,
                        'reagents': reagents,
                        'context_align': context_align, 
                    }
                    try:
                        txn.put(p_key.encode(), pickle.dumps(processed))
                    except:
                        print('Error processing index {} and product {}'.format(i, p_key))
                        continue
                else:
                    print('Warning. Process Failed.')

        return

    def get_fingerprint(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 512).ToBitString())
            else:
                return [0] * 512
        except:
            return [0] * 512
        
    def __getitem__(self, idx):
        p_key = self.product_keys[idx]
        with self.env.begin(write=False) as txn:
            processed = pickle.loads(txn.get(p_key))
        reaction_id, p_key = p_key.decode().split(' ')

        p = np.random.rand()
        # data augmentation on the fly
        if p > 0.5 and self.augment:
            prod = processed['raw_product']
            react = processed['raw_reactants']
            rt = processed['reaction_class']
            try:
                src, tgt, context_align = self.parse_smi(prod, react, rt, randomize=True, react_permute_mode="all")
            except:
                # cano_smiles
                src, tgt = processed['src'], processed['tgt']
                context_align = processed['context_align']
        else:
            src, tgt = processed['src'], processed['tgt']
            context_align = processed['context_align']

        # Make sure the reaction class is known/unknown
        if self.known_class:
            src[0] = self.src_stoi[processed['reaction_class']]
        else:
            src[0] = self.src_stoi['<UNK>']

        try:
            reaction_class = eval(processed['reaction_class'].split("_")[-1][:-1]) 
        except:
            reaction_class = 0.

        reagents = processed.get('reagents', "")
        #! change the representation of reagents
        # ECFP4
        reagents_fp = [self.get_fingerprint(x) for x in reagents]
        # MACCS
        # reagents_fp = [list(rdMolDescriptors.GetMACCSKeysFingerprint(Chem.MolFromSmiles(x)))[1:] for x in reagents]
        # AP3
        # reagents_fp = [list(rdMolDescriptors.GetHashedAtomPairFingerprint(Chem.MolFromSmiles(x),2048,use2D=True)) for x in reagents]
        #TODO: use environment factors(temp, pka) and descriptors of reagents
        return src, tgt, reaction_class, eval(reaction_id), torch.tensor(np.array(reagents_fp, dtype=float), dtype=torch.float), context_align


class YieldDatasetAug(YieldDataset):
    
    def parse_smi(self, prod, reacts, react_class, build_vocab=False, randomize=False, react_permute_mode="order"):
        ''' Process raw atom-mapped product and reactants into model-ready inputs
        :param prod: atom-mapped product
        :param reacts: atom-mapped reactants
        :param react_class: reaction class
        :param build_vocab: whether it is the stage of vocab building
        :param randomize: whether do random permutation of the reaction smiles
        :return:
        '''
        # Process raw prod and reacts:
        cano_prod_am = canonical_smiles_with_am(prod)
        cano_reacts_am = canonical_smiles_with_am(reacts) #! [Au]-->[Au:0]

        cano_prod = clear_map_number(prod)
        cano_reacts = remove_am_without_canonical(cano_reacts_am)  #! [Au:0]-->Au

        if build_vocab:
            return cano_prod, cano_reacts

        if Chem.MolFromSmiles(cano_reacts) is None:
            cano_reacts = clear_map_number(reacts)

        if Chem.MolFromSmiles(cano_prod) is None or Chem.MolFromSmiles(cano_reacts) is None:
            return None

        if randomize:
            # print('permute product')
            cano_prod_am = randomize_smiles_with_am(prod)
            cano_prod = remove_am_without_canonical(cano_prod_am)

            # print('permute reacts')
            if react_permute_mode == "all":
                reacts_list = reacts.split('.')
                np.random.shuffle(reacts_list)
                reacts_list_random = [randomize_smiles_with_am(react) for react in reacts_list]
                cano_reacts_am = '.'.join(reacts_list_random)
                cano_reacts = remove_am_without_canonical(cano_reacts_am)
            elif react_permute_mode == "order":
                cano_reacts_am = '.'.join(cano_reacts_am.split('.')[::-1]) #* randomize order of reacts
                cano_reacts = remove_am_without_canonical(cano_reacts_am)

        # Prepare model inputs
        src_token = smi_tokenizer(cano_prod)
        tgt_token = ['<sos>'] + smi_tokenizer(cano_reacts) + ['<eos>']
        if self.known_class:
            src_token = [react_class] + src_token
        else:
            src_token = ['<UNK>'] + src_token

        src_token = [self.src_stoi.get(st, self.src_stoi['<unk>']) for st in src_token]
        tgt_token = [self.tgt_stoi.get(tt, self.tgt_stoi['<unk>']) for tt in tgt_token]

        # Get the context alignment based on atom-mapping
        position_mapping_list = extract_relative_mapping(cano_prod_am, cano_reacts_am)

        # Note: gt_context_attn.size(0) = tgt.size(0) - 1; attention for token that need to predict
        gt_context_attn = torch.zeros(
            (len(smi_tokenizer(cano_reacts_am)) + 1, len(smi_tokenizer(cano_prod_am)) + 1)).long()
        for i, j in position_mapping_list:
            gt_context_attn[i][j + 1] = 1

        return src_token, tgt_token, gt_context_attn
    
    def _get_item_aug(self, prod, react, rt):
        src, tgt, context_align = self.parse_smi(prod, react, rt, randomize=True, react_permute_mode="all")
        assert src[0] == self.src_stoi['<UNK>']
        
        return src, tgt, context_align
    
    def __getitem__(self, idx):
        p_key = self.product_keys[idx]
        with self.env.begin(write=False) as txn:
            processed = pickle.loads(txn.get(p_key))
        reaction_id, p_key = p_key.decode().split(' ')

        prod = processed['raw_product']
        react = processed['raw_reactants']
        rt = processed['reaction_class']
        
        parse_result = [self._get_item_aug(prod, react, rt) for _ in range(5)]
        parse_result.append((processed['src'], processed['tgt'], processed['context_align']))
        src, tgt, context_align = zip(*parse_result)

        try:
            reaction_class = eval(processed['reaction_class'].split("_")[-1][:-1]) 
        except:
            reaction_class = 0.

        reagents = processed.get('reagents', "")
        reagents_fp = [list(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x),2,512).ToBitString()) for x in reagents]
        return src, tgt, reaction_class, eval(reaction_id), torch.tensor(np.array(reagents_fp, dtype=float), dtype=torch.float), context_align
        

class ECDataset(ForwardDataset):
    def __init__(self, mode, data_folder='./data', data_file=None,
                 known_class=False, shared_vocab=False, augment=False):
        self.data_folder = data_folder
        self.data_file = data_file

        assert mode in ['train', 'test', 'val', 'train_val']
        self.BondTypes = ['NONE', 'AROMATIC', 'DOUBLE', 'SINGLE', 'TRIPLE']
        self.bondtoi = {bond: i for i, bond in enumerate(self.BondTypes)}

        self.mode = mode
        self.augment = augment
        self.known_class = known_class
        self.shared_vocab = shared_vocab
        print('Building {} data from: {}'.format(mode, data_folder))
        vocab_file = ''
        if 'full' in self.data_folder:
            vocab_file = 'full_'
        if shared_vocab:
            vocab_file += 'vocab_share.pk'
        else:
            vocab_file += 'vocab.pk'
        ec_vocab_file = "ec_vocab.pk"

        if mode != "train":
            assert vocab_file in os.listdir(data_folder)
            assert ec_vocab_file in os.listdir(data_folder)
            with open(os.path.join(data_folder, vocab_file), 'rb') as f:
                self.src_itos, self.tgt_itos = pickle.load(f)
            self.src_stoi = {self.src_itos[i]: i for i in range(len(self.src_itos))}
            self.tgt_stoi = {self.tgt_itos[i]: i for i in range(len(self.tgt_itos))}
            with open(os.path.join(data_folder, ec_vocab_file), 'rb') as f:
                self.ec_itos = pickle.load(f)
            self.ec_stoi = {self.ec_itos[i]: i for i in range(len(self.ec_itos))}
            if data_file is None:
                self.data = pd.read_csv(os.path.join(data_folder, 'raw_{}.csv'.format(mode)))
            else:
                self.data = pd.read_csv(os.path.join(data_folder, f'{data_file}.csv'))
        else:
            train_data = pd.read_csv(os.path.join(data_folder, 'raw_train.csv'))
            val_data = pd.read_csv(os.path.join(data_folder, 'raw_val.csv'))
            raw_data = pd.concat([val_data, train_data])
            raw_data.reset_index(inplace=True, drop=True)

            if vocab_file not in os.listdir(data_folder):
                print('Building vocab...')

                prods, reacts = self.build_vocab_from_raw_data(raw_data)
                if self.shared_vocab:  # Shared src and tgt vocab
                    itos = set()
                    for i in range(len(prods)):
                        itos.update(smi_tokenizer(prods[i]))
                        itos.update(smi_tokenizer(reacts[i]))
                    # vocab for reaction_class token 
                    itos.update(['<RX_{}>'.format(i) for i in range(1, 11)])
                    itos.add('<UNK>')
                    itos = ['<unused{}'.format(i) for i in range(1, 11)] + sorted(list(itos)) 
                    itos = ['<unk>', '<pad>', '<sos>', '<eos>', '<mask>'] + itos
                    self.src_itos, self.tgt_itos = itos, itos
                else:  # Non-shared src and tgt vocab
                    self.src_itos, self.tgt_itos = set(), set()
                    for i in range(len(prods)):
                        self.src_itos.update(smi_tokenizer(reacts[i]))
                        self.tgt_itos.update(smi_tokenizer(prods[i]))
                    self.src_itos.update(['<RX_{}>'.format(i) for i in range(1, 11)])
                    self.src_itos.add('<UNK>')
                    self.src_itos = ['<unused{}'.format(i) for i in range(1, 11)] + sorted(list(self.src_itos)) 
                    self.src_itos = ['<unk>', '<pad>', '<sos>', '<eos>', '<mask>'] + self.src_itos

                    self.tgt_itos = ['<unused{}'.format(i) for i in range(1, 11)] + sorted(list(self.tgt_itos)) 
                    self.tgt_itos = ['<unk>', '<pad>', '<sos>', '<eos>'] + self.tgt_itos

                self.src_stoi = {self.src_itos[i]: i for i in range(len(self.src_itos))}
                self.tgt_stoi = {self.tgt_itos[i]: i for i in range(len(self.tgt_itos))}

                with open(os.path.join(data_folder, vocab_file), 'wb') as f:
                    pickle.dump([self.src_itos, self.tgt_itos], f)
            else:
                with open(os.path.join(data_folder, vocab_file), 'rb') as f:
                    self.src_itos, self.tgt_itos = pickle.load(f)

                self.src_stoi = {self.src_itos[i]: i for i in range(len(self.src_itos))}
                self.tgt_stoi = {self.tgt_itos[i]: i for i in range(len(self.tgt_itos))}

            if ec_vocab_file not in os.listdir(data_folder):
                print('Building ec vocab...')
                itos = set()
                ec = raw_data["ec"].tolist()
                for i in range(len(ec)):
                    itos.update(smi_tokenizer(ec[i]))
                self.ec_itos = ['<unk>'] + sorted(list(itos))
                self.ec_stoi = {self.ec_itos[i]: i for i in range(len(self.ec_itos))}
                with open(os.path.join(data_folder, ec_vocab_file), 'wb') as f:
                    pickle.dump(self.ec_itos, f)
            else:
                with open(os.path.join(data_folder, ec_vocab_file), 'rb') as f:
                    self.ec_itos = pickle.load(f)
                self.ec_stoi = {self.ec_itos[i]: i for i in range(len(self.ec_itos))}
                
            self.data = eval('{}_data'.format(mode))

        # Build and load processed data into lmdb
        self.cooked_file = f'cooked_{self.mode}.lmdb' if self.data_file is None else f'cooked_{self.data_file}.lmdb'
        if self.cooked_file not in os.listdir(self.data_folder):
            self.build_processed_data(self.data)
        self.env = lmdb.open(os.path.join(self.data_folder, self.cooked_file),
                             max_readers=1, readonly=True,
                             lock=False, readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            self.product_keys = list(txn.cursor().iternext(values=False))

        self.factor_func = lambda x: (1 + gaussian(x, 5.55391565, 0.27170542, 1.20071279))      
    
    def build_processed_data(self, raw_data):
        raw_data.reset_index(inplace=True, drop=True)
        reactions = raw_data['reactants>reagents>production'].to_list()

        env = lmdb.open(os.path.join(self.data_folder, self.cooked_file),
                        map_size=1099511627776)

        with env.begin(write=True) as txn:
            for i in tqdm(range(len(reactions))):
                rxn = reactions[i]
                r, reagents, p = rxn.split('>')
                if len(reagents) > 0:
                    r = ".".join([r, reagents])
                rt = '<RX_{}>'.format(raw_data['class'][i]) if 'class' in raw_data else '<UNK>'
                ec = raw_data["ec"][i]
                result = self.parse_smi_wrapper((p, r, rt, ec))
                if result is not None:
                    src, tgt, ec_token = result

                    p_key = '{}'.format(i)
                    processed = {
                        'src': src,
                        'tgt': tgt,
                        'raw_product': p,  
                        'raw_reactants': r, 
                        'reaction_class': rt,
                        'raw_ec': ec,
                        'ec': ec_token
                    }
                    try:
                        txn.put(p_key.encode(), pickle.dumps(processed))
                    except:
                        print('Error processing index {} and product {}'.format(i, p))
                        continue
                else:
                    print('Warning. Process Failed.')

        return
    
    def parse_smi(self, prod, reacts, react_class, ec=None, build_vocab=False, randomize=False):  
        ''' 
        return: src_token(reacts), tgt_token(prods), ec_token
        '''
        # Process raw prod and reacts:
        cano_prod_am = canonical_smiles_with_am(prod)
        cano_reacts_am = canonical_smiles_with_am(reacts)

        cano_prod = clear_map_number(prod)
        cano_reacts = remove_am_without_canonical(cano_reacts_am)

        if build_vocab:
            return cano_prod, cano_reacts

        if Chem.MolFromSmiles(cano_reacts) is None:
            cano_reacts = clear_map_number(reacts)

        if Chem.MolFromSmiles(cano_prod) is None or Chem.MolFromSmiles(cano_reacts) is None:
            return None

        if randomize:
            # print('permute product')
            cano_prod_am = randomize_smiles_with_am(prod)
            cano_prod = remove_am_without_canonical(cano_prod_am)
            if np.random.rand() > 0.5:
                # print('permute reacts')
                cano_reacts_am = '.'.join(cano_reacts_am.split('.')[::-1])
                cano_reacts = remove_am_without_canonical(cano_reacts_am)

        # Prepare model inputs
        src_token = smi_tokenizer(cano_reacts)
        tgt_token = ['<sos>'] + smi_tokenizer(cano_prod) + ['<eos>']
        ec_token = smi_tokenizer(ec)

        if self.known_class:
            src_token = [react_class] + src_token
        else:
            src_token = ['<UNK>'] + src_token

        src_token = [self.src_stoi.get(st, self.src_stoi['<unk>']) for st in src_token]
        tgt_token = [self.tgt_stoi.get(tt, self.tgt_stoi['<unk>']) for tt in tgt_token]
        
        ec_token = [self.ec_stoi.get(ec, self.ec_stoi['<unk>']) for ec in ec_token]

        return src_token, tgt_token, ec_token
    
    def parse_smi_wrapper(self, task):
        prod, reacts, react_class, ec = task
        if not prod or not reacts:
            return None
        return self.parse_smi(prod, reacts, react_class, ec, build_vocab=False, randomize=False)
    
    def __getitem__(self, idx):
        p_key = self.product_keys[idx]
        with self.env.begin(write=False) as txn:
            processed = pickle.loads(txn.get(p_key))
        reaction_id = p_key.decode()

        p = np.random.rand()
        if self.mode == 'train' and p > 0.5 and self.augment:
            prod = processed['raw_product']
            react = processed['raw_reactants']
            rt = processed['reaction_class']
            ec = processed["raw_ec"]
            try:
                src, tgt, ec = self.parse_smi(prod, react, rt, ec, randomize=True)
            except:
                src, tgt, ec = processed['src'], processed['tgt'], processed['ec']
        else:
            src, tgt, ec = processed['src'], processed['tgt'], processed['ec']

        return src, tgt, ec



class HierarchicalBatchSampler(Sampler):
    def __init__(self, batch_size, drop_last, dataset, num_replicas=1, rank=0):

        super().__init__(dataset)
        self.batch_size = batch_size
        self.dataset = dataset  # PCLRetroDataset
        self.epoch=0

        self.num_replicas = num_replicas
        self.rank = rank  # rank: [0, num_replicas)
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.dataset) - self.num_replicas) / \
                self.num_replicas  # type: ignore
            )
        else:
            self.num_samples = math.ceil(
                len(self.dataset) / self.num_replicas)  # type: ignore
        self.total_size = self.num_samples * self.num_replicas
        print(self.total_size, self.num_replicas, self.batch_size,
              self.num_samples, len(self.dataset), self.rank) 

    def random_unvisited_sample(self, label, label_dict, visited, indices, remaining, num_attempt=10):
        attempt = 0
        while attempt < num_attempt:
            idx = self.dataset.random_sample(
                label, label_dict)
            if idx not in visited and idx in indices:
                visited.add(idx)
                return idx
            attempt += 1
        idx = remaining[torch.randint(len(remaining), (1,))]
        visited.add(idx)
        return idx

    def __iter__(self):
        "Yield a batch of data"
        g = torch.Generator() 
        g.manual_seed(self.epoch)
        batch = []
        visited = set()
        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]

        assert len(indices) == self.total_size

        # subsample (for multi-gpu training)
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        remaining = list(set(indices).difference(visited))  # unvisited indices
        while len(remaining) > self.batch_size:
            # 1. random sample an anchor and get the label hierarchy
            idx = indices[torch.randint(len(indices), (1,))] #! why not sample from remaining?
            batch.append(idx)
            visited.add(idx)
            superclass, template, rxn_index = self.dataset.get_label_by_index(idx)
            
            # 2. random sample hierar positive samples 
            rxn_index = self.random_unvisited_sample(rxn_index, self.dataset.labels[superclass][template], visited, indices, remaining) #! rxn with same template
            template_index = self.random_unvisited_sample(template, self.dataset.labels[superclass], visited, indices, remaining) #! rxn with same superclass
            
            # 3. random sample negative sample
            superclass_index = self.random_unvisited_sample(superclass, self.dataset.labels, visited, indices, remaining) #! rxn with different superclass
                
            batch.extend([rxn_index, template_index, superclass_index])  # list
            visited.update([rxn_index, template_index, superclass_index])  # set  
            remaining = list(set(indices).difference(visited))
            if len(batch) >= self.batch_size: 
                yield batch
                batch = []
            # remaining = list(set(indices).difference(visited))

        # if (len(remaining) > self.batch_size) and not self.drop_last:
        if (len(remaining) <= self.batch_size) and not self.drop_last:
            # batch.update(list(remaining))
            batch.extend(remaining)
            yield batch

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self) -> int:
        # return self.num_samples // self.batch_size  # num_iters
        return math.ceil(self.num_samples / self.batch_size)