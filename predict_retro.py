import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
from utils.smiles_utils import *
from utils.translate_utils import translate_batch
from utils.build_utils import build_model, build_retro_iterator, load_checkpoint_downstream
from datetime import datetime
from utils.logging import init_logger, TensorboardLogger
import json
import copy
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda', help='device GPU/CPU')
    parser.add_argument('--batch_size_trn', type=int, default=32, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=32, help='batch size')
    parser.add_argument('--beam_size', type=int, default=10, help='beam size')

    parser.add_argument('--encoder_num_layers', type=int, default=4, help='number of layers of transformer')
    parser.add_argument('--decoder_num_layers', type=int, default=4, help='number of layers of transformer')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model representation')
    parser.add_argument('--heads', type=int, default=8, help='number of heads of multi-head attention')
    parser.add_argument('--d_ff', type=int, default=2048, help='')
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--known_class', action="store_true", default=False)
    parser.add_argument('--shared_vocab', action="store_true", default=False)
    parser.add_argument('--shared_encoder', action="store_true", default=False)

    parser.add_argument('--data_dir', type=str, default='./data/uspto_50k_typed', help='base directory')
    parser.add_argument('--exp_dir', type=str, default='./result/uspto_50k_untyped/rebuttal', help='result directory')
    parser.add_argument('--checkpoint', type=str, help='checkpoint model file')
    
    # delta tuning params
    parser.add_argument('--ft_mode', type=str, default='petl', choices=["petl", "full", "none"]) 
    parser.add_argument('--ffn_mode', type=str, default='none', choices=["none", "adapter"])
    parser.add_argument('--ffn_option', type=str, default="none", choices=["parallel", "sequential", "none"])
    parser.add_argument('--ffn_bn', type=int, default=256)
    parser.add_argument('--ffn_adapter_scalar', type=str, default="1")  # learnable or fixed 
    parser.add_argument('--ffn_adapter_init_option', type=str, default="lora", choices=["bert", "lora"])
    parser.add_argument('--ffn_adapter_layernorm_option', type=str, default="none", choices=["in", "out", "none"]) 
    parser.add_argument('--attn_mode', type=str, default='none', choices=["none", "prefix", "lora", "adpater"])
    parser.add_argument('--attn_bn', type=int, default=10)
    parser.add_argument('--attn_dim', type=int, default=32)
    
    parser.add_argument('--prompt', action="store_true", default=False)
    parser.add_argument('--input_prompt_attn', action="store_true", default=False)
    parser.add_argument('--proto_hierarchy', type=int, default=3, help='the number of hierarchy')
    parser.add_argument('--proto_path', type=str, default='./result/ecreact')
    parser.add_argument('--proto_version', type=str, default="top", choices=["bottom", "middle", "top", "hierarchy", "namerxn"])
    parser.add_argument('--freeze_proto', action="store_true", default=False)
    
    args = parser.parse_args()
    return args


def translate(iterator, model, dataset):
    ground_truths_src = []
    ground_truths_tgt = []
    generations = []
    invalid_token_indices = [dataset.tgt_stoi['<RX_{}>'.format(i)] for i in range(1, 11)]
    invalid_token_indices += [dataset.tgt_stoi['<UNK>'], dataset.tgt_stoi['<unk>'], dataset.tgt_stoi['<mask>']] 
    invalid_token_indices += [dataset.tgt_stoi['<unused{}>'.format(i)] for i in range(1, 11)]
    # Translate:
    for batch in tqdm(iterator, total=len(iterator)):
        src, tgt, _, _, _ = batch

        #! Graph2Smiles: model.predict_step()
        pred_tokens, pred_scores = translate_batch(model, batch, device=args.device, beam_size=args.beam_size,
                                                   invalid_token_indices=invalid_token_indices,
                                                   max_length=args.max_length, prompt=args.prompt)
        for idx in range(batch[0].shape[1]): # batch_size
            gt_src = ''.join(dataset.reconstruct_smi(src[:, idx], src=True))
            gt_tgt = ''.join(dataset.reconstruct_smi(tgt[:, idx], src=False)) #! remove <pad>/<sos>/<eos>
            # map id to tokens
            hypos = np.array([''.join(dataset.reconstruct_smi(tokens, src=False)) for tokens in pred_tokens[idx]])
            hypo_len = np.array([len(smi_tokenizer(ht)) for ht in hypos])
            new_pred_score = copy.deepcopy(pred_scores[idx]).cpu().numpy() / hypo_len
            ordering = np.argsort(new_pred_score)[::-1]

            ground_truths_src.append(gt_src)
            ground_truths_tgt.append(gt_tgt)
            generations.append(hypos[ordering])

    return ground_truths_src, ground_truths_tgt, generations


def main(args):
    # Build Data Iterator:
    train_iter, val_iter, dataset = build_retro_iterator(args, mode="train")
    test_iter, _ = build_retro_iterator(args, mode="test")

    # Load Checkpoint Model:
    args.weight = torch.randn(13, 512)
    model = build_model(args, dataset.src_itos, dataset.tgt_itos)
    model = load_checkpoint_downstream(args.checkpoint, model)
    model.to(args.device)

    # Get Output Path:
    exp_version = 'typed' if args.known_class == 'True' else 'untyped'
    aug_version = '_augment' if 'augment' in args.checkpoint else ''
    output_path = os.path.join(args.exp_dir,'bs_top{}_generation_{}{}.pk'.format(args.beam_size, exp_version, aug_version))
    print('Output path: {}'.format(output_path))

    # Begin Translating:
    #! select train/val/test_iter
    ground_truths_src, ground_truths_tgt, generations = translate(val_iter, model, dataset)
    accuracy_matrix = np.zeros((len(ground_truths_tgt), args.beam_size))
    for i in range(len(ground_truths_tgt)):
        gt_i = canonical_smiles(ground_truths_tgt[i])
        generation_i = [canonical_smiles(gen) for gen in generations[i]]
        for j in range(args.beam_size):
            if gt_i in generation_i[:j + 1]:
                accuracy_matrix[i][j] = 1

    with open(output_path, 'wb') as f:
        pickle.dump((ground_truths_src, ground_truths_tgt, generations), f)

    for j in range(args.beam_size):
        logger.info('Top-{}: {}'.format(j + 1, round(np.mean(accuracy_matrix[:, j]), 4)))

    return


def calculate_topk_acc(predicted_path):
    with open(predicted_path, "rb") as f:
        ground_truths, generations = pickle.load(f)
    beam_size = len(generations[0])
    # print(beam_size)
    accuracy_matrix = np.zeros((len(ground_truths), beam_size))
    for i in tqdm(range(len(ground_truths))):
        gt_i = canonical_smiles(ground_truths[i])
        generation_i = [canonical_smiles(gen) for gen in generations[i]]
        for j in range(beam_size):
            if gt_i in generation_i[:j + 1]:
                accuracy_matrix[i][j] = 1
    for j in range(beam_size):
        print('Top-{}: {}'.format(j + 1, round(np.mean(accuracy_matrix[:, j]), 4)))


if __name__ == "__main__":
    args = arg_parse()    
    
    dt = datetime.now()
    args.exp_dir = os.path.join(args.exp_dir, '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second))
    os.makedirs(args.exp_dir, exist_ok=True)
    args.data_file = None
    args.shared_vocab = True
    args.known_class = False
    args.prompt = False
    args.checkpoint = "./result/uspto_50k_untyped/rebuttal/model_39000_augment.pt"

    logger = init_logger(os.path.join(args.exp_dir, "log_metrics.txt"))
    with open(os.path.join(args.exp_dir, 'config_predict.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    main(args)
