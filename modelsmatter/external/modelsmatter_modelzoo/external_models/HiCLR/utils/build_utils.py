
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


from functools import partial
import os
import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import DataLoader
from utils.dataset_hierar import *
from models.model import MolecularTransformer


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


def load_checkpoint(checkpoint_path, model, proj=None, optimizer=None):
    """
    load both the transformer and the projection net 
    """
    print('Loading checkpoint from {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint['model'], strict=False)
    if proj is not None:
        proj.load_state_dict(checkpoint['proj'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optim'])
    step = checkpoint['step']
    return step, model, proj, optimizer


def load_checkpoint_downstream(checkpoint_path, model, model_type=None):
    print('Loading checkpoint from {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if model_type is None:
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint[model_type], strict=False)

    return model

def load_checkpoint_new_vocab(checkpoint_path, model, model_type=None):
    """
    If vocab size is changed, just need to re-initialize the word embedding layer
    """
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


def build_retro_iterator(args, mode="train", sample=False, canonicalize=True, augment=False, 
                         sample_per_class=8, random_state=0, hierar_sampling=False, shuffle=False, vocab_folder='./data'):
    if mode == "train":
        dataset = SimpleRetroDataset(mode='train', data_folder=args.data_dir,
                                     known_class=args.known_class,
                                     shared_vocab=args.shared_vocab, sample=sample, augment=augment,
                                     sample_per_class=sample_per_class, random_state=random_state,
                                     vocab_folder=vocab_folder, canonicalize=canonicalize)
        dataset_val = SimpleRetroDataset(mode='val', data_folder=args.data_dir,
                                         known_class=args.known_class,
                                         shared_vocab=args.shared_vocab,
                                         vocab_folder=vocab_folder, canonicalize=canonicalize)
        src_pad, tgt_pad = dataset.src_stoi['<pad>'], dataset.tgt_stoi['<pad>']
        train_iter = DataLoader(dataset, batch_size=args.batch_size_trn, shuffle=not sample, 
                                collate_fn=partial(collate_fn, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device))
        val_iter = DataLoader(dataset_val, batch_size=args.batch_size_val, shuffle=False, 
                              collate_fn=partial(collate_fn, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device))
        return train_iter, val_iter, dataset

    elif mode == "test":
        dataset = SimpleRetroDataset(mode='test', data_folder=args.data_dir,
                                     known_class=args.known_class,
                                     shared_vocab=args.shared_vocab, data_file=args.data_file,
                                     vocab_folder=vocab_folder, canonicalize=canonicalize) 
        src_pad, tgt_pad = dataset.src_stoi['<pad>'], dataset.tgt_stoi['<pad>']
        test_iter = DataLoader(dataset, batch_size=args.batch_size_val, shuffle=shuffle, num_workers=0,
                               collate_fn=partial(collate_fn, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device))
        return test_iter, dataset
    
    elif mode == "train_yield":
        dataset = YieldDataset(mode='test', data_folder=args.data_dir,
                               known_class=args.known_class,
                               shared_vocab=args.shared_vocab, data_file=args.data_file) 
        src_pad, tgt_pad = dataset.src_stoi['<pad>'], dataset.tgt_stoi['<pad>']
        test_iter = DataLoader(dataset, batch_size=args.batch_size_val, shuffle=shuffle, num_workers=0,
                               collate_fn=partial(collate_fn, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device, num_bins=args.num_bins))
        return test_iter, dataset
    
    elif mode == "test_yield":
        if augment:
            dataset = YieldDatasetAug(mode='test', data_folder=args.data_dir,
                                      known_class=args.known_class,
                                      shared_vocab=args.shared_vocab, data_file=args.data_file) 
        else:
            dataset = YieldDataset(mode='test', data_folder=args.data_dir,
                                   known_class=args.known_class,
                                   shared_vocab=args.shared_vocab, data_file=args.data_file) 
        src_pad, tgt_pad = dataset.src_stoi['<pad>'], dataset.tgt_stoi['<pad>']
        
        test_iter = DataLoader(dataset, batch_size=1 if augment else args.batch_size_val, shuffle=shuffle, num_workers=0,
                               collate_fn=partial(collate_fn, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device, num_bins=args.num_bins))
        return test_iter, dataset

    elif mode == "pretrain":
        dataset = PCLRetroDataset(mode="train_val", data_folder=args.data_dir,
                                  known_class=args.known_class,
                                  shared_vocab=args.shared_vocab, sample=sample, augment=augment)
        src_pad, tgt_pad = dataset.src_stoi['<pad>'], dataset.tgt_stoi['<pad>']
        if hierar_sampling:
            sampler = HierarchicalBatchSampler(batch_size=args.batch_size_trn, drop_last=False, dataset=dataset)
            train_iter = DataLoader(dataset, batch_size=1, num_workers=0,
                                    collate_fn=partial(collate_fn_pretrain, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device),
                                    sampler=sampler) #! customized sampler, not allow shuffle
        else:
            train_iter = DataLoader(dataset, batch_size=args.batch_size_trn, shuffle=not sample, num_workers=0,
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


# def collate_fn_pretrain(data, src_pad, tgt_pad, device='cuda'):
#     """Build mini-batch tensors:
#     :param sep: (int) index of src seperator
#     :param pads: (tuple) index of src and tgt padding
#     """
#     # Sort a data list by caption length
#     # data.sort(key=lambda x: len(x[0]), reverse=True)

#     if len(data) == 1:
#         src, tgt, src_1, tgt_1, src_template_mask_1, tgt_template_mask_1, src_2, tgt_2, src_template_mask_2, tgt_template_mask_2, reaction_class = data[0]
#         batch_size = len(src)
#     else:
#         src, tgt, src_1, tgt_1, src_template_mask_1, tgt_template_mask_1, src_2, tgt_2, src_template_mask_2, tgt_template_mask_2, reaction_class = zip(*data)
#         batch_size = len(data)
    

#     max_src_length = max(max([len(s) for s in src]), max([len(s) for s in src_1]), max([len(s) for s in src_2]))
#     max_tgt_length = max(max([len(t) for t in tgt]), max([len(t) for t in tgt_1]), max([len(t) for t in tgt_2]))

#     anchor = torch.zeros([], device=device)

#     # Pad_sequence
#     new_src = anchor.new_full((max_src_length, batch_size), src_pad, dtype=torch.long)
#     new_tgt = anchor.new_full((max_tgt_length, batch_size), tgt_pad, dtype=torch.long)
#     new_src_1 = anchor.new_full((max_src_length, batch_size), src_pad, dtype=torch.long)
#     new_tgt_1 = anchor.new_full((max_tgt_length, batch_size), tgt_pad, dtype=torch.long)
#     new_src_2 = anchor.new_full((max_src_length, batch_size), src_pad, dtype=torch.long)
#     new_tgt_2 = anchor.new_full((max_tgt_length, batch_size), tgt_pad, dtype=torch.long)
#     #! template_mask=1 means the token is in template-matched substructure
#     new_src_template_mask_1 = anchor.new_zeros((max_src_length, batch_size), dtype=torch.bool)
#     new_tgt_template_mask_1 = anchor.new_zeros((max_tgt_length, batch_size), dtype=torch.bool)
#     new_src_template_mask_2 = anchor.new_zeros((max_src_length, batch_size), dtype=torch.bool)
#     new_tgt_template_mask_2 = anchor.new_zeros((max_tgt_length, batch_size), dtype=torch.bool)

#     for i in range(batch_size):
#         new_src[:, i][:len(src[i])] = torch.LongTensor(src[i])
#         new_tgt[:, i][:len(tgt[i])] = torch.LongTensor(tgt[i])

#         new_src_1[:, i][:len(src_1[i])] = torch.LongTensor(src_1[i])
#         new_src_2[:, i][:len(src_2[i])] = torch.LongTensor(src_2[i])
#         new_tgt_1[:, i][:len(tgt_1[i])] = torch.LongTensor(tgt_1[i])
#         new_tgt_2[:, i][:len(tgt_2[i])] = torch.LongTensor(tgt_2[i])

#         new_src_template_mask_1[:, i][:len(src_template_mask_1[i])] = torch.BoolTensor(src_template_mask_1[i])  
#         new_tgt_template_mask_1[:, i][:len(tgt_template_mask_1[i])] = torch.BoolTensor(tgt_template_mask_1[i])
#         new_src_template_mask_2[:, i][:len(src_template_mask_2[i])] = torch.BoolTensor(src_template_mask_2[i])  
#         new_tgt_template_mask_2[:, i][:len(tgt_template_mask_2[i])] = torch.BoolTensor(tgt_template_mask_2[i])

#     return new_src, new_tgt, new_src_1, new_tgt_1, new_src_2, new_tgt_2, new_src_template_mask_1, new_tgt_template_mask_1, new_src_template_mask_2, new_tgt_template_mask_2, torch.tensor(reaction_class)


# def accumulate_batch_pretrain(true_batch, src_pad=1, tgt_pad=1):
#     src_max_length, tgt_max_length, entry_count = 0, 0, 0
#     batch_size = true_batch[0][0].shape[1]
#     hierar_level = true_batch[0][-1].shape[1]
#     for batch in true_batch:
#         src, tgt, _, _, _, _, _, _, _, _, _ = batch
#         src_max_length = max(src.shape[0], src_max_length)
#         tgt_max_length = max(tgt.shape[0], tgt_max_length)
#         entry_count += tgt.shape[1]

#     new_src = torch.zeros((src_max_length, entry_count)).fill_(src_pad).long()
#     new_tgt = torch.zeros((tgt_max_length, entry_count)).fill_(tgt_pad).long()
#     new_src_1 = torch.zeros((src_max_length, entry_count)).fill_(src_pad).long()
#     new_tgt_1 = torch.zeros((tgt_max_length, entry_count)).fill_(tgt_pad).long()
#     new_src_2 = torch.zeros((src_max_length, entry_count)).fill_(src_pad).long()
#     new_tgt_2 = torch.zeros((tgt_max_length, entry_count)).fill_(tgt_pad).long()
#     new_src_template_mask_1 = torch.zeros((src_max_length, entry_count))  # mask=1 means the token is included in templates
#     new_tgt_template_mask_1 = torch.zeros((tgt_max_length, entry_count))
#     new_src_template_mask_2 = torch.zeros((src_max_length, entry_count))
#     new_tgt_template_mask_2 = torch.zeros((tgt_max_length, entry_count))

#     new_reaction_class = torch.zeros(entry_count, hierar_level)

#     for i in range(len(true_batch)):
#         src, tgt, src_1, tgt_1, src_2, tgt_2, src_template_mask_1, tgt_template_mask_1, src_template_mask_2, tgt_template_mask_2, reaction_class = true_batch[i]
#         new_src[:, batch_size * i: batch_size * (i + 1)][:src.shape[0]] = src
#         new_tgt[:, batch_size * i: batch_size * (i + 1)][:tgt.shape[0]] = tgt
#         new_src_1[:, batch_size * i: batch_size * (i + 1)][:src_1.shape[0]] = src_1
#         new_tgt_1[:, batch_size * i: batch_size * (i + 1)][:tgt_1.shape[0]] = tgt_1
#         new_src_2[:, batch_size * i: batch_size * (i + 1)][:src_2.shape[0]] = src_2
#         new_tgt_2[:, batch_size * i: batch_size * (i + 1)][:tgt_2.shape[0]] = tgt_2

#         new_src_template_mask_1[:, batch_size * i: batch_size * (i + 1)][:src_template_mask_1.shape[0]] = src_template_mask_1
#         new_tgt_template_mask_1[:, batch_size * i: batch_size * (i + 1)][:tgt_template_mask_1.shape[0]] = tgt_template_mask_1
#         new_src_template_mask_2[:, batch_size * i: batch_size * (i + 1)][:src_template_mask_2.shape[0]] = src_template_mask_2
#         new_tgt_template_mask_2[:, batch_size * i: batch_size * (i + 1)][:tgt_template_mask_2.shape[0]] = tgt_template_mask_2

#         new_reaction_class[batch_size * i: batch_size * (i + 1)] =  reaction_class

#     return new_src, new_tgt, new_src_1, new_tgt_1, new_src_2, new_tgt_2, new_src_template_mask_1, new_tgt_template_mask_1, new_src_template_mask_2, new_tgt_template_mask_2, new_reaction_class


def collate_fn_pretrain(data, src_pad, tgt_pad, device='cuda'):
    """Build mini-batch tensors:
    :param sep: (int) index of src seperator
    :param pads: (tuple) index of src and tgt padding
    """
    # Sort a data list by caption length
    # data.sort(key=lambda x: len(x[0]), reverse=True)

    if len(data) == 1:
        src, tgt, alignment, src_1, tgt_1, src_2, tgt_2, reaction_class = data[0]
        batch_size = len(src)
    else:
        src, tgt, alignment, src_1, tgt_1, src_2, tgt_2, reaction_class = zip(*data)
        batch_size = len(data)
    

    max_src_length = max(max([len(s) for s in src]), max([len(s) for s in src_1]), max([len(s) for s in src_2]))
    max_tgt_length = max(max([len(t) for t in tgt]), max([len(t) for t in tgt_1]), max([len(t) for t in tgt_2]))

    anchor = torch.zeros([], device=device)

    # Pad_sequence
    new_src = anchor.new_full((max_src_length, batch_size), src_pad, dtype=torch.long)
    new_tgt = anchor.new_full((max_tgt_length, batch_size), tgt_pad, dtype=torch.long)
    new_src_1 = anchor.new_full((max_src_length, batch_size), src_pad, dtype=torch.long)
    new_tgt_1 = anchor.new_full((max_tgt_length, batch_size), tgt_pad, dtype=torch.long)
    new_src_2 = anchor.new_full((max_src_length, batch_size), src_pad, dtype=torch.long)
    new_tgt_2 = anchor.new_full((max_tgt_length, batch_size), tgt_pad, dtype=torch.long)
    new_alignment = anchor.new_zeros((batch_size, max_tgt_length - 1, max_src_length), dtype=torch.float)

    for i in range(batch_size):
        new_src[:, i][:len(src[i])] = torch.LongTensor(src[i])
        new_tgt[:, i][:len(tgt[i])] = torch.LongTensor(tgt[i])

        new_src_1[:, i][:len(src_1[i])] = torch.LongTensor(src_1[i])
        new_src_2[:, i][:len(src_2[i])] = torch.LongTensor(src_2[i])
        new_tgt_1[:, i][:len(tgt_1[i])] = torch.LongTensor(tgt_1[i])
        new_tgt_2[:, i][:len(tgt_2[i])] = torch.LongTensor(tgt_2[i])
        new_alignment[i, :alignment[i].shape[0], :alignment[i].shape[1]] = alignment[i].float()


    return new_src, new_tgt, new_alignment, new_src_1, new_tgt_1, new_src_2, new_tgt_2, torch.tensor(reaction_class)


def accumulate_batch_pretrain(true_batch, src_pad=1, tgt_pad=1):
    src_max_length, tgt_max_length, entry_count = 0, 0, 0
    batch_size = true_batch[0][0].shape[1]
    hierar_level = true_batch[0][-1].shape[1]
    for batch in true_batch:
        src, tgt, _, _, _, _, _, _ = batch
        src_max_length = max(src.shape[0], src_max_length)
        tgt_max_length = max(tgt.shape[0], tgt_max_length)
        entry_count += tgt.shape[1]

    new_src = torch.zeros((src_max_length, entry_count)).fill_(src_pad).long()
    new_tgt = torch.zeros((tgt_max_length, entry_count)).fill_(tgt_pad).long()
    new_align = torch.zeros((entry_count, tgt_max_length - 1, src_max_length)).float()
    new_src_1 = torch.zeros((src_max_length, entry_count)).fill_(src_pad).long()
    new_tgt_1 = torch.zeros((tgt_max_length, entry_count)).fill_(tgt_pad).long()
    new_src_2 = torch.zeros((src_max_length, entry_count)).fill_(src_pad).long()
    new_tgt_2 = torch.zeros((tgt_max_length, entry_count)).fill_(tgt_pad).long()
    new_reaction_class = torch.zeros(entry_count, hierar_level)

    for i in range(len(true_batch)):
        src, tgt, align, src_1, tgt_1, src_2, tgt_2, reaction_class = true_batch[i]
        new_src[:, batch_size * i: batch_size * (i + 1)][:src.shape[0]] = src
        new_tgt[:, batch_size * i: batch_size * (i + 1)][:tgt.shape[0]] = tgt
        new_align[batch_size * i: batch_size * (i + 1), :align.shape[1], :align.shape[2]] = align
        new_src_1[:, batch_size * i: batch_size * (i + 1)][:src_1.shape[0]] = src_1
        new_tgt_1[:, batch_size * i: batch_size * (i + 1)][:tgt_1.shape[0]] = tgt_1
        new_src_2[:, batch_size * i: batch_size * (i + 1)][:src_2.shape[0]] = src_2
        new_tgt_2[:, batch_size * i: batch_size * (i + 1)][:tgt_2.shape[0]] = tgt_2
        new_reaction_class[batch_size * i: batch_size * (i + 1)] =  reaction_class

    return new_src, new_tgt, new_align, new_src_1, new_tgt_1, new_src_2, new_tgt_2, new_reaction_class


# dataset.__getitem__ --> collate_fn(for padding) --> accumulate_batch(for using batch_size_tokens)
from utils.imbalanced_reg import get_lds_kernel_window, get_bin_idx
from scipy.ndimage import convolve1d
from collections import Counter
def collate_fn(data, src_pad, tgt_pad, device='cuda', num_bins=None):
    """Build mini-batch tensors:
    :param sep: (int) index of src seperator
    :param pads: (tuple) index of src and tgt padding
    """
    # Sort a data list by caption length
    # data.sort(key=lambda x: len(x[0]), reverse=True)
    src, tgt, reaction_class, reaction_id, reagents, alignment = zip(*data)
    if isinstance(src[0], tuple):
        src, tgt, alignment = src[0], tgt[0], alignment[0]
    max_src_length = max([len(s) for s in src])
    max_tgt_length = max([len(t) for t in tgt])
    batch_size = len(src)

    anchor = torch.zeros([], device=device)

    # Pad_sequence
    new_src = anchor.new_full((max_src_length, batch_size), src_pad, dtype=torch.long)
    new_tgt = anchor.new_full((max_tgt_length, batch_size), tgt_pad, dtype=torch.long)
    new_alignment = anchor.new_zeros((batch_size, max_tgt_length - 1, max_src_length), dtype=torch.int)

    for i in range(batch_size):
        new_src[:, i][:len(src[i])] = torch.LongTensor(src[i])
        new_tgt[:, i][:len(tgt[i])] = torch.LongTensor(tgt[i])
        new_alignment[i, :alignment[i].shape[0], :alignment[i].shape[1]] = alignment[i].float()
    
    if num_bins is not None:
        # bin_index_per_label = [get_bin_idx(label, num_bins=num_bins) for label in reaction_class]
        # # calculate empirical (original) label distribution: [Nb,]
        # # "Nb" is the number of bins
        # Nb = max(bin_index_per_label) + 1
        # num_samples_of_bins = dict(Counter(bin_index_per_label))
        # emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]
        # # lds_kernel_window: [ks,], here for example, we use gaussian, ks=5, sigma=2
        # lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=5, sigma=2)
        # # calculate effective label distribution: [Nb,]
        # eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')
        # eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
        # weights = torch.tensor([np.float32(1 / x) for x in eff_num_per_label])
        
        weights = torch.tensor(Smooth_Label(reaction_class, num_bins))
    else:
        weights = None
    
    reaction_class = torch.tensor(reaction_class)
    reaction_id = torch.tensor(reaction_id)
    if isinstance(src[0], tuple):
        reaction_class = reaction_class.repeat(batch_size)
        reaction_id = reaction_id.repeat(batch_size)
    
    if isinstance(reagents[0], torch.Tensor):  #! [num_condition, len_fp]
        new_reagents = torch.stack(reagents, dim=1)  #! [num_condition, batch_size, len_fp]
        if isinstance(src[0], tuple):
            new_reagents = new_reagents.repeat(1, batch_size, 1)
        return new_src, new_tgt, reaction_class, reaction_id, new_reagents, weights
    else:
        return new_src, new_tgt, reaction_class, reaction_id, None, weights


def Smooth_Label(Label_new, num_bins):
    labels = Label_new
    # for i in range(len(labels)):
    #     labels[i] = labels[i] - min(labels)
    # bin_index_per_label = [int(label*10) for label in labels]
    bin_index_per_label = [get_bin_idx(label, num_bins=num_bins) for label in labels]
    # print(bin_index_per_label)
    Nb = max(bin_index_per_label) + 1
    # print(Nb)
    num_samples_of_bins = dict(Counter(bin_index_per_label))
    # print(num_samples_of_bins)
    emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]
    # print(emp_label_dist, len(emp_label_dist))
    eff_label_dist = []
    beta = 0.9
    for i in range(len(emp_label_dist)):
        eff_label_dist.append((1-math.pow(beta, emp_label_dist[i])) / (1-beta))
    # print(eff_label_dist)
    eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
    weights = [np.float32(1 / x) for x in eff_num_per_label]
    weights = np.array(weights)
    # print(weights)
    # print(len(weights))
    return weights

def accumulate_batch(true_batch, src_pad=1, tgt_pad=1):
    src_max_length, tgt_max_length, entry_count = 0, 0, 0
    batch_size = true_batch[0][0].shape[1]
    for batch in true_batch:
        src, tgt, rt, id, reagents, align = batch
        src_max_length = max(src.shape[0], src_max_length)
        tgt_max_length = max(tgt.shape[0], tgt_max_length)
        entry_count += tgt.shape[1]

    new_src = torch.zeros((src_max_length, entry_count)).fill_(src_pad).long()
    new_tgt = torch.zeros((tgt_max_length, entry_count)).fill_(tgt_pad).long()
    new_reaction_class = torch.zeros(entry_count)
    new_id = torch.zeros(entry_count)
    new_align = torch.zeros((entry_count, tgt_max_length - 1, src_max_length)).float()

    for i in range(len(true_batch)):
        src, tgt, rt, id, reagents, align = true_batch[i]
        new_src[:, batch_size * i: batch_size * (i + 1)][:src.shape[0]] = src
        new_tgt[:, batch_size * i: batch_size * (i + 1)][:tgt.shape[0]] = tgt
        new_reaction_class[batch_size * i: batch_size * (i + 1)] =  rt
        new_id[batch_size * i: batch_size * (i + 1)] =  id
        # new_align[batch_size * i: batch_size * (i + 1), :align.shape[1], :align.shape[2]] = align

    return new_src, new_tgt, new_reaction_class, new_id, new_align


def build_iterator(args, mode="train", sample=False, sample_per_class=8, random_state=0):
    df = pd.read_csv(os.path.join(args.data_dir, f"raw_{mode}_fps.csv.gz"), compression="gzip")
    df[args.fp_col] = df[args.fp_col].apply(eval)
    fps = df[args.fp_col].tolist()
    # fps = np.load(os.path.join(args.data_dir, f"{args.fp_col}_fps.npz"))[mode]
    
    if sample:
        reaction_class = df["class"].unique()
        sample_index = np.array([])
        for class_id in reaction_class:
            class_index = df[df["class"]==class_id].sample(n=sample_per_class, random_state=random_state).index.values
            sample_index = np.concatenate([sample_index, class_index])
        df = df.iloc[sample_index, :].reset_index(drop=True)
        fps = fps[sample_index.astype(np.int32)]

    labels = df["class"].tolist()
    dataset = list(zip(fps, labels))
    len_fp = len(fps[0])

    if mode == "train":
        train_iter = DataLoader(dataset, batch_size=args.batch_size_trn, shuffle=True, 
                                collate_fn=collate_fps)
        df[args.fp_col] = df[args.fp_col].apply(np.array)
        result = df.groupby("class").apply(lambda x: x[args.fp_col].mean(0))
        class_centroids = np.stack(result.values.tolist(), axis=0)
        return train_iter, labels, class_centroids

    else:
        test_iter = DataLoader(dataset, batch_size=args.batch_size_val, shuffle=False, 
                               collate_fn=collate_fps)
        return test_iter
    

def collate_fps(data):
    fps, labels = map(list, zip(*data))

    return torch.tensor(fps).float(), torch.tensor(labels)


class DataCollector:
    """
    Provide task-specific collate_fn and batch accumulater
    """
    def __init__(self, collect_type="pretrain"):
        self.collect_type = collect_type
    
    def get_collector(self):
        if self.collect_type == "pretrain":  #* smiles augmentation for contrastive learning
            return collate_fn_pretrain, accumulate_batch_pretrain 
        elif self.collect_type == "retroformer":  #* smiles_graph, context_align, non-reactive mask
            return collate_fn_retroformer, accumulate_batch_retroformer
        elif self.collect_type == "uspto": #* just src token and tgt token
            return collate_fn, accumulate_batch
        elif self.collect_type == "ecreact":  #* EC number
            return collate_fn_ecreact, accumulate_batch_ecreact
        elif self.collect_type == "fingerprints": #* just pre-calculated fps and classification/regression labels
            return collate_fps
        elif self.collect_type == "raw": #* accept raw smiles as input (canonicalize and tokenize on-the-fly)
            pass  


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