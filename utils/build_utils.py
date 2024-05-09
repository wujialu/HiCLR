
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


from functools import partial
import os
import numpy as np
import pandas as pd
import torch
import random
import pickle
from torch.utils.data import DataLoader
from dataset_hierar import RetroDataset, ForwardDataset, PCLRetroDataset, PCLForwardDataset, DualDataset, ECDataset
from models.model import MolecularTransformer

def load_vocab(vocab_file="vocab_share.pk"):
    with open(vocab_file, 'rb') as f:
        src_itos, tgt_itos = pickle.load(f)
    src_stoi = {src_itos[i]: i for i in range(len(src_itos))}
    tgt_stoi = {tgt_itos[i]: i for i in range(len(tgt_itos))}
    return src_itos, src_stoi, tgt_itos, tgt_stoi

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_checkpoint(checkpoint_path, model, proj_src=None, proj_tgt=None):
    # checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint)
    print('Loading checkpoint from {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # model.load_state_dict(checkpoint['model'], strict=False)
    
    state_dict = checkpoint['model']
    neglect_key = [k for k in state_dict.keys() if "position" in k]  # change max length
    [state_dict.pop(k) for k in neglect_key]
    print(f"Don't load paramaters: {neglect_key}")
    model.load_state_dict(state_dict, strict=False)
    
    if proj_src is not None and proj_tgt is not None:
        proj_src.load_state_dict(checkpoint['proj_src'])
        proj_tgt.load_state_dict(checkpoint['proj_tgt'])
    optimizer = checkpoint['optim']
    step = checkpoint['step']

    # step += 1
    return step, optimizer, model, proj_src, proj_tgt


def load_checkpoint_single_proj(checkpoint_path, model, proj=None):
    # checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint)
    print('Loading checkpoint from {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint['model'], strict=False)
    if proj is not None:
        proj.load_state_dict(checkpoint['proj'])
    optimizer = checkpoint['optim']
    step = checkpoint['step']
    # step += 1
    return step, optimizer, model, proj


def load_checkpoint_downstream(checkpoint_path, model, model_type=None):
    print('Loading checkpoint from {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if model_type is None:
        state_dict = checkpoint['model']
        neglect_key = [k for k in state_dict.keys() if "position" in k]  # change max length
        [state_dict.pop(k) for k in neglect_key]
        print(f"Don't load paramaters: {neglect_key}")
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint[model_type], strict=False)

    return model

def load_checkpoint_new_vocab(checkpoint_path, model, model_type=None):
    print('Loading checkpoint from {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if model_type is None:
        state_dict = checkpoint['model']
        neglect_key = [k for k in state_dict.keys() if "embedding" in k or "generator" in k]
        [state_dict.pop(k) for k in neglect_key]
        print(f"Don't load paramaters: {neglect_key}")
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint[model_type], strict=False)

    return model

def build_model(args, vocab_itos_src, vocab_itos_tgt, vocab_itos_ec=None):
    src_pad_idx = np.argwhere(np.array(vocab_itos_src) == '<pad>')[0][0]
    tgt_pad_idx = np.argwhere(np.array(vocab_itos_tgt) == '<pad>')[0][0]
    vocab_size_ec = len(vocab_itos_ec) if vocab_itos_ec is not None else None
    
    model = MolecularTransformer(
        args,
        encoder_num_layers=args.encoder_num_layers,
        decoder_num_layers=args.decoder_num_layers,
        d_model=args.d_model, heads=args.heads, d_ff=args.d_ff, dropout=args.dropout,
        vocab_size_src=len(vocab_itos_src), vocab_size_tgt=len(vocab_itos_tgt),
        shared_vocab=args.shared_vocab, shared_encoder=args.shared_encoder,
        src_pad_idx=src_pad_idx, tgt_pad_idx=tgt_pad_idx, vocab_size_prompt=vocab_size_ec)

    return model.to(args.device)


def build_forward_iterator(args, mode="train", sample=False, augment=False, sample_per_class=8, random_state=0):
    if mode == "train":
        dataset = ForwardDataset(mode='train', data_folder=args.data_dir,
                               known_class=args.known_class,
                               shared_vocab=args.shared_vocab, sample=sample, augment=augment,
                               sample_per_class=sample_per_class, random_state=random_state)
        dataset_val = ForwardDataset(mode='val', data_folder=args.data_dir,
                                   known_class=args.known_class,
                                   shared_vocab=args.shared_vocab)
        src_pad, tgt_pad = dataset.src_stoi['<pad>'], dataset.tgt_stoi['<pad>']
        train_iter = DataLoader(dataset, batch_size=args.batch_size_trn, shuffle=True, 
                                collate_fn=partial(collate_fn, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device))
        val_iter = DataLoader(dataset_val, batch_size=args.batch_size_val, shuffle=False, 
                              collate_fn=partial(collate_fn, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device))
        return train_iter, val_iter, dataset.src_itos, dataset.tgt_itos, dataset.y_mean, dataset.y_std

    elif mode == "test":
        dataset = ForwardDataset(mode='test', data_folder=args.data_dir,
                               known_class=args.known_class,
                               shared_vocab=args.shared_vocab, data_file=args.data_file) 
        src_pad, tgt_pad = dataset.src_stoi['<pad>'], dataset.tgt_stoi['<pad>']
        test_iter = DataLoader(dataset, batch_size=args.batch_size_val, shuffle=False, 
                               collate_fn=partial(collate_fn, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device))
        return test_iter, dataset

    elif mode == "pretrain":
        dataset = PCLForwardDataset(mode="train_val", data_folder=args.data_dir,
                            known_class=args.known_class,
                            shared_vocab=args.shared_vocab, sample=sample, augment=augment)
        src_pad, tgt_pad = dataset.src_stoi['<pad>'], dataset.tgt_stoi['<pad>']
        train_iter = DataLoader(dataset, batch_size=args.batch_size_trn, shuffle=not sample, 
                                collate_fn=partial(collate_fn_pretrain, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device))
        return train_iter, dataset.src_itos, dataset.tgt_itos
    
    else:
        print("Please select a valid mode from pretrain/train/test")


def build_retro_iterator(args, mode="train", sample=False, augment=False):
    if mode == "train":
        dataset = RetroDataset(mode='train', data_folder=args.data_dir,
                               known_class=args.known_class,
                               shared_vocab=args.shared_vocab, sample=sample, augment=augment)
        dataset_val = RetroDataset(mode='val', data_folder=args.data_dir,
                                   known_class=args.known_class,
                                   shared_vocab=args.shared_vocab, sample=sample)
        src_pad, tgt_pad = dataset.src_stoi['<pad>'], dataset.tgt_stoi['<pad>']
        train_iter = DataLoader(dataset, batch_size=args.batch_size_trn, shuffle=not sample, 
                                collate_fn=partial(collate_fn, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device))
        val_iter = DataLoader(dataset_val, batch_size=args.batch_size_val, shuffle=False, 
                              collate_fn=partial(collate_fn, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device))
        return train_iter, val_iter, dataset.src_itos, dataset.tgt_itos

    elif mode == "test":
        dataset = RetroDataset(mode='test', data_folder=args.data_dir,
                               known_class=args.known_class,
                               shared_vocab=args.shared_vocab, data_file=args.data_file)  #! no need to build vocab
        src_pad, tgt_pad = dataset.src_stoi['<pad>'], dataset.tgt_stoi['<pad>']
        test_iter = DataLoader(dataset, batch_size=args.batch_size_val, shuffle=False, num_workers=0,
                               collate_fn=partial(collate_fn, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device))
        return test_iter, dataset

    elif mode == "pretrain":
        dataset = PCLRetroDataset(mode="train_val", data_folder=args.data_dir,
                            known_class=args.known_class,
                            shared_vocab=args.shared_vocab, sample=sample, augment=augment)
        src_pad, tgt_pad = dataset.src_stoi['<pad>'], dataset.tgt_stoi['<pad>']
        train_iter = DataLoader(dataset, batch_size=args.batch_size_trn, shuffle=not sample,  # num_workers=8,
                                collate_fn=partial(collate_fn_pretrain, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device))
        return train_iter, dataset.src_itos, dataset.tgt_itos
    
    else:
        print("Please select a valid mode from pretrain/train/test")


def collate_fn_retroformer(data, src_pad, tgt_pad, device='cuda'):
    """Build mini-batch tensors:
    :param sep: (int) index of src seperator
    :param pads: (tuple) index of src and tgt padding
    """
    # Sort a data list by caption length
    # data.sort(key=lambda x: len(x[0]), reverse=True)
    src, src_graph, tgt, alignment, nonreactive_mask = zip(*data)
    max_src_length = max([len(s) for s in src])
    max_tgt_length = max([len(t) for t in tgt])

    anchor = torch.zeros([], device=device)

    # Graph structure with edge attributes
    new_bond_matrix = anchor.new_zeros((len(data), max_src_length, max_src_length, 7), dtype=torch.long)

    # Pad_sequence
    new_src = anchor.new_full((max_src_length, len(data)), src_pad, dtype=torch.long)  #! fill_value=src_pad
    new_tgt = anchor.new_full((max_tgt_length, len(data)), tgt_pad, dtype=torch.long)
    new_alignment = anchor.new_zeros((len(data), max_tgt_length - 1, max_src_length), dtype=torch.float)
    new_nonreactive_mask = anchor.new_ones((max_src_length, len(data)), dtype=torch.bool)

    for i in range(len(data)):
        new_src[:, i][:len(src[i])] = torch.LongTensor(src[i])
        new_nonreactive_mask[:, i][:len(nonreactive_mask[i])] = torch.BoolTensor(nonreactive_mask[i])
        new_tgt[:, i][:len(tgt[i])] = torch.LongTensor(tgt[i])
        new_alignment[i, :alignment[i].shape[0], :alignment[i].shape[1]] = alignment[i].float()

        full_adj_matrix = torch.from_numpy(src_graph[i].full_adjacency_tensor)
        new_bond_matrix[i, 1:full_adj_matrix.shape[0]+1, 1:full_adj_matrix.shape[1]+1] = full_adj_matrix

    return new_src, new_tgt, new_alignment, new_nonreactive_mask, (new_bond_matrix, src_graph)


def accumulate_batch_retroformer(true_batch, src_pad=1, tgt_pad=1):
    src_max_length, tgt_max_length, entry_count = 0, 0, 0
    batch_size = true_batch[0][0].shape[1]
    for batch in true_batch:
        src, tgt, _, _, _ = batch
        src_max_length = max(src.shape[0], src_max_length)
        tgt_max_length = max(tgt.shape[0], tgt_max_length)
        entry_count += tgt.shape[1]

    new_src = torch.zeros((src_max_length, entry_count)).fill_(src_pad).long()
    new_tgt = torch.zeros((tgt_max_length, entry_count)).fill_(tgt_pad).long()

    new_context_alignment = torch.zeros((entry_count, tgt_max_length - 1, src_max_length)).float()
    new_nonreactive_mask = torch.ones((src_max_length, entry_count)).bool()

    # Graph packs:
    new_bond_matrix = torch.zeros((entry_count, src_max_length, src_max_length, 7)).long()
    new_src_graph_list = []

    for i in range(len(true_batch)):
        src, tgt, context_alignment, nonreactive_mask, graph_packs = true_batch[i]
        bond, src_graph = graph_packs
        new_src[:, batch_size * i: batch_size * (i + 1)][:src.shape[0]] = src
        new_nonreactive_mask[:, batch_size * i: batch_size * (i + 1)][:nonreactive_mask.shape[0]] = nonreactive_mask
        new_tgt[:, batch_size * i: batch_size * (i + 1)][:tgt.shape[0]] = tgt
        new_context_alignment[batch_size * i: batch_size * (i + 1), :context_alignment.shape[1], :context_alignment.shape[2]] = context_alignment

        new_bond_matrix[batch_size * i: batch_size * (i + 1), :bond.shape[1], :bond.shape[2]] = bond
        new_src_graph_list += src_graph

    return new_src, new_tgt, new_context_alignment, new_nonreactive_mask, \
           (new_bond_matrix, new_src_graph_list)


def collate_fn_pretrain(data, src_pad, tgt_pad, device='cuda'):
    """Build mini-batch tensors:
    :param sep: (int) index of src seperator
    :param pads: (tuple) index of src and tgt padding
    """
    # Sort a data list by caption length
    # data.sort(key=lambda x: len(x[0]), reverse=True)
    src, tgt, src_template_mask, tgt_template_mask, src_permute, tgt_permute, reaction_class = zip(*data)
    max_src_length = max([len(s) for s in src])
    max_tgt_length = max([len(t) for t in tgt])
    max_src_permute_length = max([len(s) for s in src_permute])
    max_tgt_permute_length = max([len(t) for t in tgt_permute])

    anchor = torch.zeros([], device=device)

    # Pad_sequence
    new_src = anchor.new_full((max_src_length, len(data)), src_pad, dtype=torch.long)
    new_tgt = anchor.new_full((max_tgt_length, len(data)), tgt_pad, dtype=torch.long)
    new_src_permute = anchor.new_full((max_src_permute_length, len(data)), src_pad, dtype=torch.long)
    new_tgt_permute = anchor.new_full((max_tgt_permute_length, len(data)), tgt_pad, dtype=torch.long)
    #! template_mask=1 means the token is in template-matched substructure
    new_src_template_mask = anchor.new_zeros((max_src_length, len(data)), dtype=torch.bool)
    new_tgt_template_mask = anchor.new_zeros((max_tgt_length, len(data)), dtype=torch.bool)

    for i in range(len(data)):
        new_src[:, i][:len(src[i])] = torch.LongTensor(src[i])
        new_src_permute[:, i][:len(src_permute[i])] = torch.LongTensor(src_permute[i])
        new_tgt[:, i][:len(tgt[i])] = torch.LongTensor(tgt[i])
        new_tgt_permute[:, i][:len(tgt_permute[i])] = torch.LongTensor(tgt_permute[i])
        new_src_template_mask[:, i][:len(src_template_mask[i])] = torch.BoolTensor(src_template_mask[i])  
        new_tgt_template_mask[:, i][:len(tgt_template_mask[i])] = torch.BoolTensor(tgt_template_mask[i])

    return new_src, new_tgt, new_src_permute, new_tgt_permute, new_src_template_mask, new_tgt_template_mask, torch.tensor(reaction_class)


def accumulate_batch_pretrain(true_batch, src_pad=1, tgt_pad=1):
    src_max_length, tgt_max_length, entry_count, src_permute_max_length, tgt_permute_max_length = 0, 0, 0, 0, 0
    batch_size = true_batch[0][0].shape[1]
    for batch in true_batch:
        src, tgt, src_permute, tgt_permute, _, _, _ = batch
        src_max_length = max(src.shape[0], src_max_length)
        tgt_max_length = max(tgt.shape[0], tgt_max_length)
        src_permute_max_length = max(src_permute.shape[0], src_permute_max_length)
        tgt_permute_max_length = max(tgt_permute.shape[0], tgt_permute_max_length)
        entry_count += tgt.shape[1]

    new_src = torch.zeros((src_max_length, entry_count)).fill_(src_pad).long()
    new_tgt = torch.zeros((tgt_max_length, entry_count)).fill_(tgt_pad).long()
    new_src_permute = torch.zeros((src_permute_max_length, entry_count)).fill_(src_pad).long()
    new_tgt_permute = torch.zeros((tgt_permute_max_length, entry_count)).fill_(tgt_pad).long()
    new_src_template_mask = torch.zeros((src_max_length, entry_count))
    new_tgt_template_mask = torch.zeros((tgt_max_length, entry_count))
    new_reaction_class = torch.zeros(entry_count)

    for i in range(len(true_batch)):
        src, tgt, src_permute, tgt_permute, src_template_mask, tgt_template_mask, reaction_class = true_batch[i]
        new_src[:, batch_size * i: batch_size * (i + 1)][:src.shape[0]] = src
        new_tgt[:, batch_size * i: batch_size * (i + 1)][:tgt.shape[0]] = tgt
        new_src_permute[:, batch_size * i: batch_size * (i + 1)][:src_permute.shape[0]] = src_permute
        new_tgt_permute[:, batch_size * i: batch_size * (i + 1)][:tgt_permute.shape[0]] = tgt_permute

        new_src_template_mask[:, batch_size * i: batch_size * (i + 1)][:src_template_mask.shape[0]] = src_template_mask
        new_tgt_template_mask[:, batch_size * i: batch_size * (i + 1)][:tgt_template_mask.shape[0]] = tgt_template_mask

        new_reaction_class[batch_size * i: batch_size * (i + 1)] =  reaction_class

    return new_src, new_tgt, new_src_permute, new_tgt_permute, new_src_template_mask, new_tgt_template_mask, new_reaction_class


# dataset.__getitem__ --> collate_fn --> accumulate_batch
def collate_fn(data, src_pad, tgt_pad, device='cuda'):
    """Build mini-batch tensors:
    :param sep: (int) index of src seperator
    :param pads: (tuple) index of src and tgt padding
    """
    # Sort a data list by caption length
    # data.sort(key=lambda x: len(x[0]), reverse=True)
    src, src_graph, tgt, context_alignment, nonreact_mask, rt, id, gt_proto = zip(*data)
    max_src_length = max([len(s) for s in src])
    max_tgt_length = max([len(t) for t in tgt])

    anchor = torch.zeros([], device=device)

    # Pad_sequence
    new_src = anchor.new_full((max_src_length, len(data)), src_pad, dtype=torch.long)
    new_tgt = anchor.new_full((max_tgt_length, len(data)), tgt_pad, dtype=torch.long)

    for i in range(len(data)):
        new_src[:, i][:len(src[i])] = torch.LongTensor(src[i])
        new_tgt[:, i][:len(tgt[i])] = torch.LongTensor(tgt[i])

    return new_src, new_tgt, torch.tensor(rt), torch.tensor(id), torch.tensor(gt_proto)


def accumulate_batch(true_batch, src_pad=1, tgt_pad=1):
    src_max_length, tgt_max_length, entry_count = 0, 0, 0
    batch_size = true_batch[0][0].shape[1]
    for batch in true_batch:
        src, tgt, rt, id, gt_proto = batch
        src_max_length = max(src.shape[0], src_max_length)
        tgt_max_length = max(tgt.shape[0], tgt_max_length)
        entry_count += tgt.shape[1]

    new_src = torch.zeros((src_max_length, entry_count)).fill_(src_pad).long()
    new_tgt = torch.zeros((tgt_max_length, entry_count)).fill_(tgt_pad).long()
    new_reaction_class = torch.zeros(entry_count)
    new_id = torch.zeros(entry_count)
    new_proto = torch.zeros(entry_count)

    for i in range(len(true_batch)):
        src, tgt, rt, id, gt_proto = true_batch[i]
        new_src[:, batch_size * i: batch_size * (i + 1)][:src.shape[0]] = src
        new_tgt[:, batch_size * i: batch_size * (i + 1)][:tgt.shape[0]] = tgt
        new_reaction_class[batch_size * i: batch_size * (i + 1)] =  rt
        new_id[batch_size * i: batch_size * (i + 1)] =  id
        new_proto[batch_size * i: batch_size * (i + 1)] =  gt_proto

    return new_src, new_tgt, new_reaction_class, new_id, new_proto


def build_iterator(args, mode="train", sample=False, sample_per_class=8, random_state=0):
    df = pd.read_csv(os.path.join(args.data_dir, f"raw_{mode}_fps.csv"))
    if sample:
        reaction_class = df["class"].unique()
        sample_index = np.array([])
        for class_id in reaction_class:
            class_index = df[df["class"]==class_id].sample(n=sample_per_class, random_state=random_state).index.values
            sample_index = np.concatenate([sample_index, class_index])
        df = df.iloc[sample_index, :].reset_index(drop=True)

    fps = list(map(eval, df[args.fp_col]))
    labels = df["class"].tolist()
    dataset = list(zip(fps, labels))
    len_fp = len(fps[0])

    if mode == "train":
        train_iter = DataLoader(dataset, batch_size=args.batch_size_trn, shuffle=True, 
                                collate_fn=collate_fps)
        return train_iter, len_fp

    else:
        test_iter = DataLoader(dataset, batch_size=args.batch_size_val, shuffle=False, 
                               collate_fn=collate_fps)
        return test_iter
    

def collate_fps(data):
    fps, labels = map(list, zip(*data))

    return torch.tensor(fps).float(), torch.tensor(labels)


def build_ecreact_iterator(args, mode="train", augment=False):
    if mode == "train":
        dataset = ECDataset(mode='train', data_folder=args.data_dir,
                            known_class=False,
                            shared_vocab=args.shared_vocab, augment=augment)
        dataset_val = ECDataset(mode='val', data_folder=args.data_dir,
                                known_class=False,
                                shared_vocab=args.shared_vocab)
        src_pad, tgt_pad = dataset.src_stoi['<pad>'], dataset.tgt_stoi['<pad>'] # pad_idx=1
        train_iter = DataLoader(dataset, batch_size=args.batch_size_trn, shuffle=True,
                                collate_fn=partial(collate_fn_ecreact, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device))
        val_iter = DataLoader(dataset_val, batch_size=args.batch_size_val, shuffle=False, 
                              collate_fn=partial(collate_fn_ecreact, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device))
        return train_iter, val_iter, dataset.src_itos, dataset.tgt_itos, dataset.ec_itos

    elif mode == "test":
        dataset = ECDataset(mode='test', data_folder=args.data_dir,
                            known_class=False,
                            shared_vocab=args.shared_vocab, data_file=args.data_file) 
        src_pad, tgt_pad = dataset.src_stoi['<pad>'], dataset.tgt_stoi['<pad>']
        test_iter = DataLoader(dataset, batch_size=args.batch_size_val, shuffle=False, num_workers=0,
                               collate_fn=partial(collate_fn_ecreact, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device))
        return test_iter, dataset
    
    
def collate_fn_ecreact(data, src_pad, tgt_pad, device='cuda'):
    """Build mini-batch tensors:
    :param sep: (int) index of src seperator
    :param pads: (tuple) index of src and tgt padding
    """
    # Sort a data list by caption length
    # data.sort(key=lambda x: len(x[0]), reverse=True)
    src, tgt, ec = zip(*data) 
    max_src_length = max([len(s) for s in src])  
    max_tgt_length = max([len(t) for t in tgt])  

    anchor = torch.zeros([], device=device)

    # Pad_sequence
    new_src = anchor.new_full((max_src_length, len(data)), src_pad, dtype=torch.long)
    new_tgt = anchor.new_full((max_tgt_length, len(data)), tgt_pad, dtype=torch.long)

    for i in range(len(data)):
        new_src[:, i][:len(src[i])] = torch.LongTensor(src[i])
        new_tgt[:, i][:len(tgt[i])] = torch.LongTensor(tgt[i])
    
    new_ec = torch.LongTensor(ec).permute(1,0).to(device)
    return new_src, new_tgt, new_ec


def accumulate_batch_ecreact(true_batch, src_pad=1, tgt_pad=1):
    src_max_length, tgt_max_length, entry_count = 0, 0, 0
    batch_size = true_batch[0][0].shape[1]
    len_ec = true_batch[0][-1].shape[0]
    for batch in true_batch:
        src, tgt, ec = batch
        src_max_length = max(src.shape[0], src_max_length)
        tgt_max_length = max(tgt.shape[0], tgt_max_length)
        entry_count += tgt.shape[1]

    new_src = torch.zeros((src_max_length, entry_count)).fill_(src_pad).long()
    new_tgt = torch.zeros((tgt_max_length, entry_count)).fill_(tgt_pad).long()
    new_ec = torch.zeros((len_ec, entry_count)).long()
    
    for i in range(len(true_batch)):
        src, tgt, ec = true_batch[i]
        new_src[:, batch_size * i: batch_size * (i + 1)][:src.shape[0]] = src
        new_tgt[:, batch_size * i: batch_size * (i + 1)][:tgt.shape[0]] = tgt
        new_ec[:, batch_size * i: batch_size * (i + 1)] = ec
        
    return new_src, new_tgt, new_ec


def build_dual_iterator(args, mode="train", sample=False, augment=False):
    if mode == "train":
        dataset = DualDataset(mode='train', data_folder=args.data_dir,
                               known_class=False,
                               shared_vocab=args.shared_vocab, sample=sample, augment=augment)
        dataset_val = DualDataset(mode='val', data_folder=args.data_dir,
                                   known_class=False,
                                   shared_vocab=args.shared_vocab, sample=sample)
        src_pad, tgt_pad = dataset.src_stoi["retro"]['<pad>'], dataset.tgt_stoi["retro"]['<pad>'] # pad_idx=1
        train_iter = DataLoader(dataset, batch_size=args.batch_size_trn, shuffle=not sample, 
                                collate_fn=partial(collate_fn_dual, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device))
        val_iter = DataLoader(dataset_val, batch_size=args.batch_size_val, shuffle=False, 
                              collate_fn=partial(collate_fn_dual, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device))
        return train_iter, val_iter, dataset.src_itos, dataset.tgt_itos, dataset.y_mean, dataset.y_std

    elif mode == "test":
        dataset = DualDataset(mode='test', data_folder=args.data_dir,
                               known_class=False,
                               shared_vocab=args.shared_vocab, data_file=args.data_file) 
        src_pad, tgt_pad = dataset.src_stoi["retro"]['<pad>'], dataset.tgt_stoi["retro"]['<pad>']
        test_iter = DataLoader(dataset, batch_size=args.batch_size_val, shuffle=False, num_workers=0,
                               collate_fn=partial(collate_fn_dual, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device))
        return test_iter, dataset
    
    
def collate_fn_dual(data, src_pad, tgt_pad, device='cuda'):
    """Build mini-batch tensors:
    :param sep: (int) index of src seperator
    :param pads: (tuple) index of src and tgt padding
    """
    # Sort a data list by caption length
    # data.sort(key=lambda x: len(x[0]), reverse=True)
    src_retro, tgt_retro, src_forward, tgt_forward, rt, id = zip(*data) 
    max_src_length = max([len(s) for s in src_forward])  # reacts+reagents
    max_tgt_length = max([len(t) for t in tgt_forward])  # prods

    anchor = torch.zeros([], device=device)

    # Pad_sequence
    new_src_retro = anchor.new_full((max_tgt_length, len(data)), src_pad, dtype=torch.long)
    new_tgt_retro = anchor.new_full((max_src_length+1, len(data)), tgt_pad, dtype=torch.long)
    new_src_forward = anchor.new_full((max_src_length, len(data)), src_pad, dtype=torch.long)
    new_tgt_forward = anchor.new_full((max_tgt_length, len(data)), tgt_pad, dtype=torch.long)

    for i in range(len(data)):
        new_src_retro[:, i][:len(src_retro[i])] = torch.LongTensor(src_retro[i])
        new_tgt_retro[:, i][:len(tgt_retro[i])] = torch.LongTensor(tgt_retro[i])
        new_src_forward[:, i][:len(src_forward[i])] = torch.LongTensor(src_forward[i])
        new_tgt_forward[:, i][:len(tgt_forward[i])] = torch.LongTensor(tgt_forward[i])

    return new_src_retro, new_tgt_retro, new_src_forward, new_tgt_forward, torch.tensor(rt), torch.tensor(id)


def accumulate_batch_dual(true_batch, src_pad=1, tgt_pad=1):
    src_max_length, tgt_max_length, entry_count = 0, 0, 0
    batch_size = true_batch[0][0].shape[1]
    for batch in true_batch:
        src_retro, tgt_retro, src_forward, tgt_forward, rt, id = batch
        src_max_length = max(src_forward.shape[0], src_max_length)
        tgt_max_length = max(tgt_forward.shape[0], tgt_max_length)
        entry_count += tgt.shape[1]

    new_src_retro = torch.zeros((tgt_max_length, entry_count)).fill_(src_pad).long()
    new_tgt_retro = torch.zeros((src_max_length+1, entry_count)).fill_(tgt_pad).long()
    new_src_forward = torch.zeros((src_max_length, entry_count)).fill_(src_pad).long()
    new_tgt_forward = torch.zeros((tgt_max_length, entry_count)).fill_(tgt_pad).long()
    new_reaction_class = torch.zeros(entry_count)
    new_id = torch.zeros(entry_count)

    for i in range(len(true_batch)):
        src, tgt, id = true_batch[i]
        new_src_retro[:, batch_size * i: batch_size * (i + 1)][:src_retro.shape[0]] = src_retro
        new_tgt_retro[:, batch_size * i: batch_size * (i + 1)][:tgt_retro.shape[0]] = tgt_retro
        new_src_forward[:, batch_size * i: batch_size * (i + 1)][:src_forward.shape[0]] = src_forward
        new_tgt_forward[:, batch_size * i: batch_size * (i + 1)][:tgt_forward.shape[0]] = tgt_forward
        new_reaction_class[batch_size * i: batch_size * (i + 1)] =  rt
        new_id[batch_size * i: batch_size * (i + 1)] =  id

    return new_src_retro, new_tgt_retro, new_src_forward, new_tgt_forward, new_reaction_class, new_id