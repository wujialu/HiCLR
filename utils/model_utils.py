import copy
import math
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F


def reallocate_batch(batch, location='cpu'):
    batch = list(batch)
    for i in range(len(batch)):
        batch[i] = batch[i].to(location)
    return tuple(batch)


def validate(model, val_iter, pad_idx=1):
    pred_token_list, gt_token_list = [], []
    model.eval()
    for batch in tqdm(val_iter):
        src, tgt, _, _, pesudo_proto = batch

        # Infer:
        with torch.no_grad():
            scores, attns, src_reps, tgt_reps, pred_proto = model(src, tgt)

        # Token accuracy:
        pred_token_logit = scores.view(-1, scores.size(2))
        _, pred_token_label = pred_token_logit.topk(1, dim=-1)
        gt_token_label = tgt[1:].view(-1)
        pred_token_list.append(pred_token_label[gt_token_label != pad_idx])
        gt_token_list.append(gt_token_label[gt_token_label != pad_idx])

    pred_tokens = torch.cat(pred_token_list).view(-1)
    gt_tokens = torch.cat(gt_token_list).view(-1)
    token_acc = (pred_tokens == gt_tokens).float().mean().item()

    return token_acc

def validate_ecreact(model, val_iter, pad_idx=1, prompt=False):
    pred_token_list, gt_token_list = [], []
    model.eval()
    for batch in tqdm(val_iter):
        src, tgt, ec = batch

        # Infer:
        with torch.no_grad():
            if prompt:
                scores, attns = model.forward_with_given_prompt(src, tgt, prompt=ec)
            else:
                scores, attns, _, _ = model(src, tgt)

        # Token accuracy:
        pred_token_logit = scores.view(-1, scores.size(2))
        _, pred_token_label = pred_token_logit.topk(1, dim=-1)
        gt_token_label = tgt[1:].view(-1)
        pred_token_list.append(pred_token_label[gt_token_label != pad_idx])
        gt_token_list.append(gt_token_label[gt_token_label != pad_idx])

    pred_tokens = torch.cat(pred_token_list).view(-1)
    gt_tokens = torch.cat(gt_token_list).view(-1)
    token_acc = (pred_tokens == gt_tokens).float().mean().item()

    return token_acc


def freeze_params_by_layers(model, num_enc_layers, num_frozen_layers=0):
    additional_frozen_params = ['model.shared.weight', 'model.encoder.embed_positions.weight',
                                'model.decoder.embed_positions.weight']
    for name, par in model.named_parameters():
        if name in additional_frozen_params:
            par.requires_grad = False
        elif not name.startswith('model'):
            print(f'{name} will update!')
        else:
            try:
                layer_idx = int(name.split('.')[3])
            except ValueError:
                par.requires_grad = False
                continue
            is_decoder = 'decoder' in name
            if is_decoder:
                layer_idx += num_enc_layers
            if layer_idx < num_frozen_layers:
                par.requires_grad = False


def freeze_params(model, except_para_l=()):
    for name, par in model.named_parameters():
        skip = False
        for except_para in except_para_l:
            if except_para in name:
                print(f'{name} |skipped when alterning requires_grad')
                skip = True
                break
        if skip:
            continue
        par.requires_grad = False


def unfreeze_params(model, except_para=None):
    for name, par in model.named_parameters():
        if except_para is not None and except_para in name:
            par.requires_grad = False
        else:
            par.requires_grad = True