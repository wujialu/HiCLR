import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import copy
import argparse
import json
import lmdb
import pickle
from utils.build_utils import build_retro_iterator, build_model, load_checkpoint_downstream, accumulate_batch, set_random_seed
from utils.logging import init_logger, TensorboardLogger
from utils.loss_utils import LabelSmoothingLoss
from utils.model_utils import validate
from utils.optimizers import build_optim
from datetime import datetime


def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action="store_true", default=False)
    parser.add_argument("--enable_tensorboard",
                        action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False, help="continue training from a given checkpoint")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size_trn', type=int, default=4, help='raw train batch size')
    parser.add_argument('--batch_size_val', type=int, default=4, help='val/test batch size')
    parser.add_argument('--batch_size_token', type=int, default=8192,
                        help='train batch token number')  #! retroformer: batch_size_token=16384, batch_size=8
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--max_train_step', type=int, default=300000)
    parser.add_argument('--report_per_step', type=int, default=200, help='train loss reporting steps frequency')
    parser.add_argument('--save_per_step', type=int, default=1000, help='checkpoint saving steps frequency')
    parser.add_argument('--val_per_step', type=int, default=1000, help='validation steps frequency')
    
    # model parameters 
    parser.add_argument('--data_dir', type=str, default='./data/uspto_50k_typed', help='base directory')
    parser.add_argument('--seed', type=int, default=0, help="Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers for dataset loading')
    parser.add_argument('--exp_dir', type=str, default='./result/ecreact', help='result directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint model file')
    parser.add_argument('--encoder_num_layers', type=int, default=4, help='number of layers of transformer')
    parser.add_argument('--decoder_num_layers', type=int, default=4, help='number of layers of transformer')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model representation')
    parser.add_argument('--heads', type=int, default=8, help='number of heads of multi-head attention')
    parser.add_argument('--d_ff', type=int, default=2048, help='')
    parser.add_argument('--known_class', action="store_true", default=False)
    parser.add_argument('--shared_vocab', action="store_true", default=False)
    parser.add_argument('--shared_encoder', action="store_true", default=False)
    parser.add_argument('--augment', action="store_true", default=False)

    # optimizer params
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--learning_rate', type=float, default=2.)
    parser.add_argument('--warmup_steps', type=int, default=8000)
    parser.add_argument('--decay_method', type=str, default='noam')
    parser.add_argument('--learning_rate_decay', type=float, default=0.5)
    parser.add_argument('--start_decay_steps', type=int, default=50000)
    parser.add_argument('--decay_steps', type=int, default=10000)
    parser.add_argument('--adagrad_accumulator_init', type=float, default=0.)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--max_grad_norm', type=float, default=0)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.998)

    parser.add_argument('--label_smoothing', type=float, default=0.1, help="label smoothing for language modeling loss")
    parser.add_argument('--disp_interval', type=int, default=1000, help='the display interval')
    
    # delta tuning params
    parser.add_argument('--ft_mode', type=str, default='petl', choices=["petl", "full", "none"]) 
    parser.add_argument('--ffn_mode', type=str, default='none', choices=["none", "adapter"])
    parser.add_argument('--ffn_option', type=str, default="none", choices=["parallel", "sequential", "none"])
    parser.add_argument('--ffn_bn', type=int, default=256)
    parser.add_argument('--ffn_adapter_scalar', type=str, default="1")  # learnable or fixed 
    parser.add_argument('--ffn_adapter_init_option', type=str, default="lora", choices=["bert", "lora"])
    parser.add_argument('--ffn_adapter_layernorm_option', type=str, default="none", choices=["in", "out", "none"]) 
    parser.add_argument('--attn_mode', type=str, default='none', choices=["none", "prefix", "lora", "adapter"])
    parser.add_argument('--attn_bn', type=int, default=32) # bottleneck dim
        
    parser.add_argument('--prompt', action="store_true", default=False)
    parser.add_argument('--input_prompt_attn', action="store_true", default=False)
    parser.add_argument('--proto_hierarchy', type=int, default=3, help='the number of hierarchy')
    parser.add_argument('--proto_path', type=str, default='./result/ecreact')
    parser.add_argument('--proto_version', type=str, default="top", choices=["bottom", "middle", "top", "hierarchy", "namerxn"])
    parser.add_argument('--freeze_proto', action="store_true", default=False)
    parser.add_argument('--proto_label_smoothing', type=float, default=0.1)

    args = parser.parse_args()
    return args


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    return n_params, enc, dec


def train(args, model, train_iter, val_iter, optimizer, pad_idx):
    model.train()

    criterion_tokens = LabelSmoothingLoss(ignore_index=model.embedding_tgt.word_padding_idx,
                                          reduction='mean', apply_logsoftmax=False,
                                          smoothing=args.label_smoothing)

    token_loss_cnt = []
    pred_token_list = []
    gt_token_list = []

    true_batch = []
    entry_count, src_max_length, tgt_max_length = 0, 0, 0
    global step
    for batch in tqdm(train_iter, desc="Iteration"):
        raw_src, raw_tgt, _, _ = batch
        src_max_length = max(src_max_length, raw_src.shape[0])
        tgt_max_length = max(tgt_max_length, raw_tgt.shape[0])
        entry_count += raw_tgt.shape[1]

        if (src_max_length + tgt_max_length) * entry_count < args.batch_size_token:
            true_batch.append(batch)
        else:
            # Accumulate Batch
            src, tgt, _, _ = accumulate_batch(true_batch)
            src, tgt = src.to(args.device), tgt.to(args.device)
            del true_batch
            torch.cuda.empty_cache()
            
            # get token representations
            generative_scores, attns, _, _ = model(src, tgt)
            
            # language modeling loss
            pred_token_logit = generative_scores.view(-1, generative_scores.size(2))
            _, pred_token_label = pred_token_logit.topk(1, dim=-1)
            gt_token_label = tgt[1:].view(-1)
            token_loss = criterion_tokens(pred_token_logit, gt_token_label)

            token_loss_cnt.append(token_loss.item())
            pred_token_list.append(pred_token_label[gt_token_label != pad_idx])
            gt_token_list.append(gt_token_label[gt_token_label != pad_idx])

            # optimization
            optimizer.zero_grad()
            loss = token_loss
            loss.backward()
            optimizer.step()

            if step % args.disp_interval == 0:
                # 1. report
                pred_tokens = torch.cat(pred_token_list).view(-1)
                gt_tokens = torch.cat(gt_token_list).view(-1)
                token_acc = (pred_tokens == gt_tokens).float().mean().item()
                
                args.logger.info(
                    'Training iteration: %d, token_loss: %f, accuracy_token: %f' % (
                        step, np.mean(token_loss_cnt), token_acc))

                args.tensorboard_logger.add_scalar(key="Train/token_loss", value=round(np.mean(token_loss_cnt), 4), use_context=False)

                # 2. save checkpoint
                if not args.debug:
                    aug_version = '_augment' if args.augment else ''
                    checkpoint_path = os.path.join(args.exp_dir, f"model_{step}{aug_version}.pt")
                    torch.save({'model': model.state_dict(), 'step': step, 'optim': optimizer.state_dict()}, checkpoint_path)
                    args.logger.info('Checkpoint saved to {}'.format(checkpoint_path))
                
                # 3. validate
                val_token_acc = validate(model, val_iter, model.embedding_tgt.word_padding_idx)
                args.logger.info('Validation accuracy: {}'.format(round(val_token_acc, 4)))

                token_loss_cnt = []
                pred_token_list = []
                gt_token_list = []
                
            # Restart Accumulation
            step += 1
            if step > args.max_train_step:
                print('Finish training.')
                break
            else:
                true_batch = [batch]
                entry_count, src_max_length, tgt_max_length = raw_src.shape[1], raw_src.shape[0], raw_tgt.shape[0]


def main(args):
    # load data and build iterator (PCLdataset)
    #* 1. train retro on new vocab
    #* 2. pretraining (first stage add ar loss), enlarge prototype loss weight, reduce lr
    #* 3. pretraining based on (src - tgt) -->  ECFP
    train_iter, val_iter, dataset = \
        build_retro_iterator(args, mode="train", augment=args.augment) 
    
    model = build_model(args, dataset.src_itos, dataset.tgt_itos)
    n_params, enc, dec = _tally_parameters(model)
    args.logger.info('encoder: %d' % enc)
    args.logger.info('decoder: %d' % dec)
    args.logger.info('* number of parameters: %d' % n_params)

    # load pre-trained model
    if args.checkpoint is not None:
        model = load_checkpoint_downstream(args.checkpoint, model)

    # build optimizer
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    if args.ffn_mode != "none" or args.attn_mode != "none":
        for name, parameter in model.named_parameters():
            if "adapter" in name:
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False
    optimizer = build_optim(model, args)
    
    # train
    global step
    step = 1
    for epoch in range(1, args.epochs + 1):
        args.logger.info("====epoch " + str(epoch))
        train(args, model, train_iter, val_iter, optimizer, model.embedding_tgt.word_padding_idx)

    # save logged_data
    args.tensorboard_logger.save(os.path.join(args.exp_dir, 'logged_data.pkl'))


if __name__ == "__main__":
    args = arg_parse()
    args.shared_vocab = True
    args.known_class = False
    
    if args.debug:        
        args.exp_dir = os.path.join(args.exp_dir, "debug")
    
    else:
        if args.checkpoint is not None:
            if args.ffn_mode != "none" or args.attn_mode != "none":
                args.exp_dir = os.path.join(args.exp_dir, "peft")
            else:
                args.exp_dir = os.path.join(args.exp_dir, "finetune")
        else:
            args.exp_dir = os.path.join(args.exp_dir, "scratch")

    dt = datetime.now()
    args.exp_dir = os.path.join(args.exp_dir, '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second))
    os.makedirs(args.exp_dir, exist_ok=True)

    with open(os.path.join(args.exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    if args.enable_tensorboard:
        args.tensorboard_logger = TensorboardLogger(args)
    log_file_name = os.path.join(args.exp_dir, "log.txt")
    args.logger = init_logger(log_file_name)

    set_random_seed(args.seed)

    main(args)