import math
import copy
import torch
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from collections import Counter
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.smiles_utils import *

#! scorer, translator


def scale(x):
    if (x != 0).sum() == 0:
        return x
    return (x - x.min()) / (x.max() - x.min())


def var(a):
    return a.clone().detach()
    # return torch.tensor(a, requires_grad=False)


def rvar(a, beam_size=10):
    if len(a.size()) == 3:
        return var(a.repeat(1, beam_size, 1))
    else:
        return var(a.repeat(1, beam_size))


def translate_batch(model, batch, eos_idx=3, beam_size=10, max_length=200,
                    invalid_token_indices=[], dataset=None, prompt=False):
    """
    :param inputs: tuple of (src, src_am, src_seg, tgt), tgt is only used to retrieve conditional reaction class token
    :param fixed_z: latent variable flag
    :param seed: latent variable flag
    :param target_mask_num: available only when generalize=False; constraint the amount of generated fragment = num of <MASK>
    :param sep_idx: target seperator '>>' index, only use when generalize=True; constraint the beam search from getting seperator too early
    :param prefix_sequence: list of prefix tokens, only use in customized template generation stage
    :return:
    """
    #TODO: top-k sampling; penalty
    # encoder foward --> decoder forward --> generator forward
    model.eval()
    if len(batch) == 5:
        src, tgt, rt, id, pseudo_proto = batch
    else:
        src, tgt, rt, id = batch
    batch_size = src.shape[1]

    pred_tokens = src.new_ones((batch_size, beam_size, max_length + 1), dtype=torch.long)
    pred_scores = src.new_zeros((batch_size, beam_size), dtype=torch.float)
    pred_tokens[:, :, 0] = 2  # ['<unk>', '<pad>', '<sos>', '<eos>']
    batch2finish = {i: False for i in range(batch_size)}

    # (1) Run the Encoder on the src.
    with torch.no_grad():
        if prompt:
            src_emb, prior_encoder_out, _, prompt_attn = model.encoder(src)
        else:
            src_emb, prior_encoder_out, _ = model.encoder(src)
    
    # (2) Repeat src objects `beam_size` times.
    memory_bank_repeat = rvar(prior_encoder_out.data, beam_size=beam_size)
    # dec_state.repeat_beam_size_times(beam_size) # repeat dec_state.src.data
    src_repeat = rvar(src.data, beam_size=beam_size)
    dec_state = \
        model.decoder.init_decoder_state(src_repeat, prior_encoder_out)

    # (3) run the decoder to generate sentences, using beam search.
    # end with <eos> token or > max_length
    for step in range(max_length):
        inp = pred_tokens.transpose(0, 1).contiguous().view(-1, pred_tokens.size(2))[:, :step + 1].transpose(0, 1)

        # run one step
        with torch.no_grad():
            outputs, dec_state, attns, _ = model.decoder(inp, memory_bank_repeat, state=dec_state, step=step) 
            scores = model.generator(outputs[-1])  # (max_len, batch_size, vocab_size_tgt)

        unbottle_scores = scores.view(beam_size, batch_size, -1)  # num_words

        # Avoid invalid token:
        unbottle_scores[:, :, invalid_token_indices] = -1e25

        # Avoid token that end earily
        #? 增大early-stop steps?
        if step < 2:
            unbottle_scores[:, :, eos_idx] = -1e25

        # Beam Search:
        # selected_indices = []
        for j in range(batch_size):
            prev_score = pred_scores[j].clone()
            batch_score = unbottle_scores[:, j]
            num_words = batch_score.size(1)
            # Get previous token to identify <eos>
            prev_token = pred_tokens[j, :, step]
            eos_index = prev_token.eq(eos_idx)
            # Prevent <eos> sequence to have children
            prev_score[eos_index] = -1e20

            if beam_size == eos_index.sum():  # all beam has finished
                pred_tokens[j, :, step + 1] = eos_idx
                batch2finish[j] = True
                # selected_indices.append(torch.arange(beam_size, dtype=torch.long, device=src.device))
            else:
                beam_scores = batch_score + prev_score.unsqueeze(1).expand_as(batch_score)

                if step == 0:
                    flat_beam_scores = beam_scores[0].view(-1)
                else:
                    flat_beam_scores = beam_scores.view(-1)  # (10, 89)-->890

                # Select the top-k highest accumulative scores
                k = beam_size - eos_index.sum().item()
                best_scores, best_scores_id = flat_beam_scores.topk(k, 0, True, True)

                # Freeze the tokens which has already finished
                frozed_tokens = pred_tokens[j][eos_index]
                if frozed_tokens.shape[0] > 0:
                    frozed_tokens[:, step + 1] = eos_idx
                frozed_scores = pred_scores[j][eos_index]  # -1e20

                # Update the rest of tokens
                origin_tokens = pred_tokens[j][best_scores_id // num_words]  # select the token from prev step
                origin_tokens[:, step + 1] = best_scores_id % num_words  # add the new token

                updated_scores = torch.cat([best_scores, frozed_scores])
                updated_tokens = torch.cat([origin_tokens, frozed_tokens])  # (beam_size, batch_size)

                pred_tokens[j] = updated_tokens
                pred_scores[j] = updated_scores

                if eos_index.sum() > 0:
                    tmp_indices = src.new_zeros(beam_size, dtype=torch.long)  #! new_zeros: keep the tensor dtype and device same as src
                    tmp_indices[:len(best_scores_id // num_words)] = (best_scores_id // num_words)
                    selected_indices = tmp_indices
                else:
                    selected_indices = (best_scores_id // num_words)

                dec_state.beam_update(idx=j, positions=selected_indices, beam_size=beam_size)

            if dataset is not None:
                if j == 0:
                    hypos = [''.join(dataset.reconstruct_smi(tokens, src=False)) for tokens in updated_tokens]
                    print('[step {}]'.format(step))
                    for hypo in hypos:
                        print(hypo)
                    # print(hypos[0])
                    print('------------------`')

        # if selected_indices:
        #     reorder_state_cache(state_cache, selected_indices)

        if sum(batch2finish.values()) == len(batch2finish):
            break

    # (Sorting is done in explain_batch)
    return pred_tokens, pred_scores


def translate_batch_ecreact(model, batch, eos_idx=3, beam_size=10, max_length=200,
                            invalid_token_indices=[], dataset=None, prompt=False):
    """
    :param inputs: tuple of (src, src_am, src_seg, tgt), tgt is only used to retrieve conditional reaction class token
    :param fixed_z: latent variable flag
    :param seed: latent variable flag
    :param target_mask_num: available only when generalize=False; constraint the amount of generated fragment = num of <MASK>
    :param sep_idx: target seperator '>>' index, only use when generalize=True; constraint the beam search from getting seperator too early
    :param prefix_sequence: list of prefix tokens, only use in customized template generation stage
    :return:
    """
    #TODO: top-k sampling; penalty
    # encoder foward --> decoder forward --> generator forward
    model.eval()
    src, tgt, ec = batch
    batch_size = src.shape[1]

    pred_tokens = src.new_ones((batch_size, beam_size, max_length + 1), dtype=torch.long)
    pred_scores = src.new_zeros((batch_size, beam_size), dtype=torch.float)
    pred_tokens[:, :, 0] = 2  # ['<unk>', '<pad>', '<sos>', '<eos>']
    batch2finish = {i: False for i in range(batch_size)}

    # (1) Run the Encoder on the src.
    with torch.no_grad():
        if prompt:
            src_emb, prior_encoder_out = model.encoder.forward_with_given_prompt(src=src, prompt=ec)
        else:
            src_emb, prior_encoder_out, _ = model.encoder(src)
    
    # (2) Repeat src objects `beam_size` times.
    memory_bank_repeat = rvar(prior_encoder_out.data, beam_size=beam_size)
    # dec_state.repeat_beam_size_times(beam_size) # repeat dec_state.src.data
    src_repeat = rvar(src.data, beam_size=beam_size)
    dec_state = \
        model.decoder.init_decoder_state(src_repeat, prior_encoder_out)

    # (3) run the decoder to generate sentences, using beam search.
    # end with <eos> token or > max_length
    for step in range(max_length):
        inp = pred_tokens.transpose(0, 1).contiguous().view(-1, pred_tokens.size(2))[:, :step + 1].transpose(0, 1)

        # run one step
        with torch.no_grad():
            if prompt:
                outputs, dec_state, attns = model.decoder.forward_with_prompt(inp, memory_bank_repeat, state=dec_state, step=step) 
            else:
                outputs, dec_state, attns, _ = model.decoder(inp, memory_bank_repeat, state=dec_state, step=step) 
            scores = model.generator(outputs[-1])  # (max_len, batch_size, vocab_size_tgt)

        unbottle_scores = scores.view(beam_size, batch_size, -1)  # num_words

        # Avoid invalid token:
        unbottle_scores[:, :, invalid_token_indices] = -1e25

        # Avoid token that end earily
        #? 增大early-stop steps?
        if step < 2:
            unbottle_scores[:, :, eos_idx] = -1e25

        # Beam Search:
        # selected_indices = []
        for j in range(batch_size):
            prev_score = pred_scores[j].clone()
            batch_score = unbottle_scores[:, j]
            num_words = batch_score.size(1)
            # Get previous token to identify <eos>
            prev_token = pred_tokens[j, :, step]
            eos_index = prev_token.eq(eos_idx)
            # Prevent <eos> sequence to have children
            prev_score[eos_index] = -1e20

            if beam_size == eos_index.sum():  # all beam has finished
                pred_tokens[j, :, step + 1] = eos_idx
                batch2finish[j] = True
                # selected_indices.append(torch.arange(beam_size, dtype=torch.long, device=src.device))
            else:
                beam_scores = batch_score + prev_score.unsqueeze(1).expand_as(batch_score)

                if step == 0:
                    flat_beam_scores = beam_scores[0].view(-1)
                else:
                    flat_beam_scores = beam_scores.view(-1)  # (10, 89)-->890

                # Select the top-k highest accumulative scores
                k = beam_size - eos_index.sum().item()
                best_scores, best_scores_id = flat_beam_scores.topk(k, 0, True, True)

                # Freeze the tokens which has already finished
                frozed_tokens = pred_tokens[j][eos_index]
                if frozed_tokens.shape[0] > 0:
                    frozed_tokens[:, step + 1] = eos_idx
                frozed_scores = pred_scores[j][eos_index]  # -1e20

                # Update the rest of tokens
                origin_tokens = pred_tokens[j][best_scores_id // num_words]  # select the token from prev step
                origin_tokens[:, step + 1] = best_scores_id % num_words  # add the new token

                updated_scores = torch.cat([best_scores, frozed_scores])
                updated_tokens = torch.cat([origin_tokens, frozed_tokens])  # (beam_size, batch_size)

                pred_tokens[j] = updated_tokens
                pred_scores[j] = updated_scores

                if eos_index.sum() > 0:
                    tmp_indices = src.new_zeros(beam_size, dtype=torch.long)  
                    tmp_indices[:len(best_scores_id // num_words)] = (best_scores_id // num_words)
                    selected_indices = tmp_indices
                else:
                    selected_indices = (best_scores_id // num_words)

                dec_state.beam_update(idx=j, positions=selected_indices, beam_size=beam_size)

            if dataset is not None:
                if j == 0:
                    hypos = [''.join(dataset.reconstruct_smi(tokens, src=False)) for tokens in updated_tokens]
                    print('[step {}]'.format(step))
                    for hypo in hypos:
                        print(hypo)
                    # print(hypos[0])
                    print('------------------`')

        # if selected_indices:
        #     reorder_state_cache(state_cache, selected_indices)

        if sum(batch2finish.values()) == len(batch2finish):
            break

    # (Sorting is done in explain_batch)
    return pred_tokens, pred_scores


def reorder_state_cache(state_cache, selected_indices):
    """Reorder state_cache of the decoder
    params state_cache: list of indices
    params selected_indices: size (batch_size x beam_size)
    """
    batch_size, beam_size = len(selected_indices), len(selected_indices[0])
    indices_mapping = torch.arange(batch_size * beam_size,
                                   device=selected_indices[0].device).reshape(beam_size, batch_size).transpose(0, 1)
    reorder_indices = []
    for batch_i, indices in enumerate(selected_indices):
        reorder_indices.append(indices_mapping[batch_i, indices])
    reorder_indices = torch.stack(reorder_indices, dim=1).view(-1)

    new_state_cache = []
    for key in state_cache:
        if isinstance(state_cache[key], dict):
            for subkey in state_cache[key]:
                state_cache[key][subkey] = state_cache[key][subkey][reorder_indices]

        elif isinstance(state_cache[key], torch.Tensor):
            state_cache[key] = state_cache[key][reorder_indices]
        else:
            raise Exception
