import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from models.model import MLP, ProjectNet
from utils.build_utils import build_retro_iterator, build_forward_iterator, build_model, load_checkpoint_downstream, accumulate_batch, set_random_seed, load_checkpoint_downstream
from utils.model_utils import freeze_params
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
from datetime import datetime
from utils.logging import init_logger, TensorboardLogger
import json
from args_parse import args_parser


def train(encoder, decoder, optimizer, loader, criterion): 
    encoder.train()
    decoder.train()

    train_loss = []
    train_acc = []
    train_f1 = []
    y_true = []
    y_pred = []

    for batch in tqdm(loader, desc="Iteration"):
        src, tgt, rt, ids, _ = batch 
        src, tgt, rt = src.to(args.device), tgt.to(args.device), rt.to(args.device)
        del batch
        torch.cuda.empty_cache()

        # get token representations
        src_reps, tgt_reps = encoder.extract_reaction_fp(src, tgt)
        
        reaction_fp = torch.cat([src_reps, tgt_reps], dim=-1)
        logits = decoder(reaction_fp)

        optimizer.zero_grad()
        loss = criterion(input=logits, target=rt) # unnormalized logits(batch_size, num_class), rt with class indices
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        y_true.append(rt)
        y_pred.append(logits.topk(1, dim=1)[1].squeeze(1))

    y_true = torch.cat(y_true).detach().cpu()
    y_pred = torch.cat(y_pred).detach().cpu()
    train_acc = accuracy_score(y_true, y_pred)
    train_f1 = f1_score(y_true, y_pred, average="macro")

    return np.mean(train_loss).round(4), np.mean(train_acc).round(4), np.mean(train_f1).round(4)


def evaluate(encoder, decoder, loader, criterion):
    encoder.eval()
    decoder.eval()

    eval_loss = []
    eval_acc = []
    eval_f1 = []
    y_true = []
    y_pred = []

    for batch in tqdm(loader, desc="Iteration"):
        src, tgt, rt, ids, _ = batch 
        src, tgt, rt = src.to(args.device), tgt.to(args.device), rt.to(args.device)
        del batch
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            src_reps, tgt_reps = encoder.extract_reaction_fp(src, tgt)
            # reaction_fp = torch.cat([proj_src(src_reps), proj_tgt(tgt_reps)],dim=-1)
            reaction_fp = torch.cat([src_reps, tgt_reps], dim=-1)
            logits = decoder(reaction_fp)

        loss = criterion(logits, rt)
        eval_loss.append(loss.item())
        y_true.append(rt)
        y_pred.append(logits.topk(1, dim=1)[1].squeeze(1))

    y_true = torch.cat(y_true).detach().cpu()
    y_pred = torch.cat(y_pred).detach().cpu()
    eval_acc = accuracy_score(y_true, y_pred)
    eval_f1 = f1_score(y_true, y_pred, average="macro")

    return np.mean(eval_loss).round(4), np.mean(eval_acc).round(4), np.mean(eval_f1).round(4)


def main(args):
    train_acc_all, val_acc_all, test_acc_all = [], [], []
    train_f1_all, val_f1_all, test_f1_all = [], [], []
    for i in range(args.num_runs):
        logger.info(f"----------Round {i + 1}, Seed {args.seed + i}----------")
        
        # set random seed
        set_random_seed(args.seed + i)

        # build_iterator
        if args.mode == "backward":
            train_iter, val_iter, vocab_itos_src, vocab_itos_tgt = \
                build_retro_iterator(args, mode="train", sample=True, augment=False, 
                                     sample_per_class=args.num_train_samples, random_state=args.seed + i)
            test_iter, _ = \
                build_retro_iterator(args, mode="test", sample=False, augment=False)
        elif args.mode == "forward":
            train_iter, val_iter, vocab_itos_src, vocab_itos_tgt, _, _ = \
                build_forward_iterator(args, mode="train", sample=True, augment=False,
                                       sample_per_class=args.num_train_samples, random_state=args.seed + i) 
            test_iter, _ = \
                build_forward_iterator(args, mode="test", sample=False, augment=False)
        
        # load pre-trained encoder (mol_transformer)
        encoder = build_model(args, vocab_itos_src, vocab_itos_tgt)
        
        if args.checkpoint is not None:
            encoder = load_checkpoint_downstream(args.checkpoint, encoder)

        # build decoder (classification head)
        if args.decoder_type == "linear":
            args.size_layer = [2*encoder.d_model] + [args.num_class]
        else:
            args.size_layer = [2*encoder.d_model] + args.decoder_hidden_size + [args.num_class]
        decoder = MLP(size_layer=args.size_layer, dropout=args.dropout).to(args.device)
        
        # build optimizer
        #! freeze params outside the task adapter
        model_param_group = [{"params": decoder.parameters(), "lr": args.lr}]
        if args.ft_mode == "full":
            model_param_group += [{"params": encoder.parameters(), "lr": args.encoder_lr}]
        elif args.ft_mode == "petl":
            model_param_group += [{"params": encoder.parameters(), "lr": args.encoder_lr}] # args.lr or args.lr*0.1
            if args.update_layernorm:
                freeze_params(encoder, except_para_l=("adapter", "prefix", "lora", "norm"))
            else:
                freeze_params(encoder, except_para_l=("adapter", "prefix", "lora"))
        elif args.ft_mode == "none":
            pass
        optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

        # loss function
        criterion = nn.CrossEntropyLoss()

        # train and validate
        best_val_acc = 0.
        counter = 0.
        for epoch in tqdm(range(args.epochs), desc="Epoch"):
            train_loss, train_acc, train_f1 = train(encoder, decoder, optimizer, train_iter, criterion)
            val_loss, val_acc, val_f1 = evaluate(encoder, decoder, val_iter, criterion)
            logger.info(f"Epoch: {epoch + 1}, Train loss: {train_loss}, Train accuracy: {train_acc}, Train F1: {train_f1}, Valid loss: {val_loss}, Valid accuracy: {val_acc}, Valid F1: {val_f1}")

            # if early-stopping, break
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint_path = os.path.join(args.exp_dir, f"model_round_{i + 1}.pt")
                torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict(), 'step': epoch + 1, 'optim': optimizer.state_dict()}, checkpoint_path)
                counter = 0.
            else:
                counter += 1
                if counter >= args.patience:
                    logger.info(f"----------Early-stopping at epoch {epoch + 1}----------")
                    break

        # test and report classification result
        #! load checkpoint
        encoder = load_checkpoint_downstream(checkpoint_path, encoder, model_type="encoder")
        decoder = load_checkpoint_downstream(checkpoint_path, decoder, model_type="decoder")
        _, train_acc, train_f1 = evaluate(encoder, decoder, train_iter, criterion)
        _, val_acc, val_f1 = evaluate(encoder, decoder, val_iter, criterion)
        _, test_acc, test_f1 = evaluate(encoder,decoder, test_iter, criterion)
        logger.info(f"Round: {i + 1}, Train accuracy: {train_acc}, Train F1: {train_f1}, Valid accuracy: {val_acc}, Valid F1: {val_f1}, Test accuracy: {test_acc}, Test F1: {test_f1} ")
        train_acc_all.append(train_acc)
        train_f1_all.append(train_f1)
        val_acc_all.append(val_acc)
        val_f1_all.append(val_f1)
        test_acc_all.append(test_acc)
        test_f1_all.append(test_f1)

    # report and save final result
    train_acc_mean, train_acc_std, train_f1_mean, train_f1_std = np.mean(train_acc_all).round(4), np.std(train_acc_all).round(4), np.mean(train_f1_all).round(4), np.std(train_f1_all).round(4)
    val_acc_mean, val_acc_std, val_f1_mean, val_f1_std = np.mean(val_acc_all).round(4), np.std(val_acc_all).round(4), np.mean(val_f1_all).round(4), np.std(val_f1_all).round(4)
    test_acc_mean, test_acc_std, test_f1_mean, test_f1_std = np.mean(test_acc_all).round(4), np.std(test_acc_all).round(4), np.mean(test_f1_all).round(4), np.std(test_f1_all).round(4)
    logger.info(f"train_result: acc: {train_acc_mean} ± {train_acc_std}, f1: {train_f1_mean} ± {train_f1_std}")
    logger.info(f"val_result: acc: {val_acc_mean} ± {val_acc_std}, f1: {val_f1_mean} ± {val_f1_std}")
    logger.info(f"test_result: acc: {test_acc_mean} ± {test_acc_std}, f1: {test_f1_mean} ± {test_f1_std}")


if __name__ == "__main__":
    args = args_parser()
    args.checkpoint = f"./checkpoint/supcon_hierar/model_pretrain_best_mAP.pt"
    args.data_dir = os.path.join(args.data_dir, args.mode)

    args.shared_vocab = True
    args.data_file = None
    dt = datetime.now()
    args.exp_dir = os.path.join(args.exp_dir, f"ft_mode_{args.ft_mode}", \
        '{}_{:02d}-{:02d}-{:02d}'.format(dt.date(), dt.hour, dt.minute, dt.second))
    
    for x in [4,8,16,32,64,128]:
        args.num_train_samples = x
        args.exp_dir_tmp = os.path.join(args.exp_dir, f"train_{args.num_train_samples}_per_class")
        os.makedirs(args.exp_dir_tmp, exist_ok=True)
        logger = init_logger(log_file=os.path.join(args.exp_dir_tmp, f"log.txt"))
        with open(os.path.join(args.exp_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
        main(args)

# python train_classification_finetune.py --mode backward --ft_mode full --decoder_type linear 