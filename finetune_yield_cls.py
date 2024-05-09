import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from models.model import MLP, YieldNet
from utils.build_utils import build_retro_iterator, build_model, load_checkpoint_downstream, set_random_seed
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from datetime import datetime
import json
from utils.logging import init_logger, TensorboardLogger
from utils.loss_utils import focal_loss
from args_parse import args_parser


def train(args, encoder, decoder, optimizer, loader, criterion): 
    encoder.train()
    decoder.train()

    train_loss = []

    for batch in tqdm(loader, desc="Iteration"):
        src, tgt, rt, idx, reagents, _ = batch 
        src, tgt, rt, reagents = src.to(args.device), tgt.to(args.device), rt.to(args.device), reagents.to(args.device)
        del batch
        torch.cuda.empty_cache()

        # get token representations
        src_reps, tgt_reps = encoder.extract_reaction_fp(src, tgt, cond=reagents)
        # reagents = decoder.linear_transform(reagents)
        # reaction_fp = torch.cat([src_reps, tgt_reps], dim=-1) + reagents
        # reaction_fp = torch.cat([src_reps, tgt_reps, torch.mul(src_reps, tgt_reps), src_reps - tgt_reps, reagents], dim=-1)
        reaction_fp = torch.cat([src_reps, tgt_reps], dim=-1)
        logits = decoder(reaction_fp)

        optimizer.zero_grad()
        loss = criterion(logits, rt.long()) # unnormalized logits(batch_size, num_class), rt with class indices
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    return np.mean(train_loss).round(4)


def evaluate(args, encoder, decoder, loader, criterion):
    encoder.eval()
    decoder.eval()

    eval_loss = []
    y_true = []
    y_pred = []

    for batch in tqdm(loader, desc="Iteration"):
        src, tgt, rt, idx, reagents, _ = batch 
        src, tgt, rt, reagents = src.to(args.device), tgt.to(args.device), rt.to(args.device), reagents.to(args.device)
        del batch
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            src_reps, tgt_reps = encoder.extract_reaction_fp(src, tgt, cond=reagents)
            # reagents = decoder.linear_transform(reagents)
            # reaction_fp = torch.cat([src_reps, tgt_reps], dim=-1) + reagents
            # reaction_fp = torch.cat([src_reps, tgt_reps, reagents], dim=-1)
            # reaction_fp = torch.cat([src_reps, tgt_reps, torch.mul(src_reps, tgt_reps), src_reps - tgt_reps, reagents], dim=-1)
            reaction_fp = torch.cat([src_reps, tgt_reps], dim=-1)
            logits = decoder(reaction_fp)

        loss = criterion(logits, rt.long())
        eval_loss.append(loss.item())
        y_true.append(rt)
        y_pred.append(torch.argmax(F.softmax(logits, dim=1), dim=1, keepdim=True))
        #! augment (batch size for test is 1)
        # y_true.append(rt[0].view(-1))
        # y_pred.append(torch.argmax(F.softmax(torch.mean(logits, 0, keepdim=True), dim=1), dim=1, keepdim=True))

    y_true = torch.cat(y_true).detach().cpu()
    y_pred = torch.cat(y_pred).detach().cpu()
    acc = accuracy_score(y_true, y_pred)

    return np.mean(eval_loss).round(4), acc.round(4), y_true, y_pred


def main(args):
    train_acc_all, val_acc_all, test_acc_all = [], [], []
    for i in range(args.num_runs):
        logger.info(f"----------Round {i + 1}, Seed {args.seed + i}----------")
        # set random seed
        set_random_seed(args.seed + i)

        # build_iterator
        if args.dataset_name == "reaxys_yield":
            train_iter, val_iter, train_dataset = \
                build_retro_iterator(args, mode="train", sample=False, augment=False)
            args.data_file = "raw_test"
            test_iter, _ = \
                build_retro_iterator(args, mode="test", sample=False, augment=False)
        else:
            args.data_file = f"train_random_split_{i}"
            #! change: augment=False-->True
            train_iter, train_dataset = \
                build_retro_iterator(args, mode="test_yield", sample=False, augment=False, shuffle=True)  
            args.data_file = f"test_random_split_{i}"
            val_iter, _ = \
                build_retro_iterator(args, mode="test_yield", sample=False, augment=False)
            test_iter = val_iter
        
        # load pre-trained encoder (mol_transformer)
        encoder = build_model(args, train_dataset.src_itos, train_dataset.tgt_itos)
        
        if args.checkpoint is not None:
            encoder = load_checkpoint_downstream(args.checkpoint, encoder)

        # build decoder and optimizer
        args.size_layer = [2*encoder.d_model] + args.decoder_hidden_size + [args.num_class]
        # args.size_layer = [2*encoder.d_model+512] + args.decoder_hidden_size + [args.num_class]
        decoder = MLP(size_layer=args.size_layer, dropout=args.dropout).to(args.device)
        model_param_group = [{"params": decoder.parameters()}]
        if not args.fix_encoder:
            if args.ffn_mode != "none" or args.attn_mode != "none":
                for name, parameter in encoder.named_parameters():
                    if "adapter" in name or "cond" in name:
                        parameter.requires_grad = True
                    else:
                        parameter.requires_grad = False
                model_param_group += [{"params": filter(lambda p: p.requires_grad, encoder.parameters()), "lr": args.encoder_lr}]
            else:
                model_param_group += [{"params": encoder.parameters(), "lr": args.encoder_lr}]
        
        optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

        # loss function
        criterion = nn.CrossEntropyLoss()
        # criterion = focal_loss(num_classes=4, device=args.device)

        # train and validate
        best_val_acc = -1.0
        counter = 0.
        for epoch in tqdm(range(args.epochs), desc="Epoch"):
            train_loss = train(args, encoder, decoder, optimizer, train_iter, criterion)
            val_loss, val_acc, _, _ = evaluate(args, encoder, decoder, val_iter, criterion)
            logger.info(f"Epoch: {epoch + 1}, Train loss: {train_loss}, Valid loss: {val_loss}, Valid acc: {val_acc}")
            test_loss, test_acc, _, _ = evaluate(args, encoder, decoder, test_iter, criterion)
            logger.info(f"Epoch: {epoch + 1}, Test acc: {test_acc}")

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
        _, train_acc, _, _ = evaluate(args, encoder, decoder, train_iter, criterion)
        _, val_acc, _, _ = evaluate(args, encoder, decoder, val_iter, criterion)
        _, test_acc, y_true, y_pred = evaluate(args, encoder, decoder, test_iter, criterion)
        logger.info(f"Round: {i + 1}, Train acc: {train_acc}, Valid acc: {val_acc}, Test acc: {test_acc}")
        train_acc_all.append(train_acc)
        val_acc_all.append(val_acc)
        test_acc_all.append(test_acc)
        
        df = pd.DataFrame()
        df["y_true"] = list(y_true.numpy().reshape(-1))
        df["y_pred"] = list(y_pred.numpy().reshape(-1))
        df.to_csv(os.path.join(args.exp_dir, f"test_pred_{i}.csv"), index=False)
        
    # report and save final result
    train_acc_mean, train_acc_std = np.mean(train_acc_all).round(4), np.std(train_acc_all).round(4)
    val_acc_mean, val_acc_std = np.mean(val_acc_all).round(4), np.std(val_acc_all).round(4)
    test_acc_mean, test_acc_std = np.mean(test_acc_all).round(4), np.std(test_acc_all).round(4)
   
    logger.info(f"train_result: acc: {train_acc_mean} ± {train_acc_std}")
    logger.info(f"val_result: acc: {val_acc_mean} ± {val_acc_std}")
    logger.info(f"test_result: acc: {test_acc_mean} ± {test_acc_std}")


def zeroshot(args):
    test_acc_all = []
    for i in range(args.num_runs):
        logger.info(f"----------Round {i + 1}, Seed {args.seed + i}----------")
        # set random se
        set_random_seed(args.seed + i)

        # args.data_dir = './data/buchward/az/classification_5class_0'
        # args.data_file = f"test_random_split_{i}"
        # args.num_class = 5
        
        args.data_dir = './data/buchward/az/classification_4class'
        args.data_file = f"processed"
        args.num_class = 4
        
        test_iter, dataset = \
            build_retro_iterator(args, mode="test_yield", sample=False, augment=False)
        
        # load pre-trained encoder (mol_transformer)
        encoder = build_model(args, dataset.src_itos, dataset.tgt_itos)

        # build decoder and optimizer
        args.size_layer = [2*encoder.d_model] + args.decoder_hidden_size + [args.num_class] # [1024/512, 512, 256, 4]
        # args.size_layer = [2*encoder.d_model] + [args.num_class]
        decoder = MLP(size_layer=args.size_layer, dropout=args.dropout).to(args.device)
        # decoder = Net(args.size_layer, dropout=args.dropout).to(args.device)
        
        checkpoint_path = os.path.join(args.exp_dir, f"model_round_{i + 1}.pt")
        encoder = load_checkpoint_downstream(checkpoint_path, encoder, model_type="encoder")
        decoder = load_checkpoint_downstream(checkpoint_path, decoder, model_type="decoder")

        # loss function
        criterion = nn.CrossEntropyLoss()

        _, test_acc, y_true, y_pred = evaluate(args, encoder, decoder, test_iter, criterion)
        logger.info(f"Round: {i + 1}, Test acc: {test_acc}")
        test_acc_all.append(test_acc)
        
        df = pd.DataFrame()
        df["y_true"] = list(y_true.numpy().reshape(-1))
        df["y_pred"] = list(y_pred.numpy().reshape(-1))
        df.to_csv(os.path.join(args.exp_dir, f"test_pred_{i}.csv"), index=False)

    # report and save final result
    test_acc_mean, test_acc_std = np.mean(test_acc_all).round(4), np.std(test_acc_all).round(4)
    logger.info(f"test_result: acc: {test_acc_mean} ± {test_acc_std}")

if __name__ == "__main__":
    args = args_parser()
    args.data_dir = f'./data/{args.dataset_name}/classification_4class_react_additive'
    args.exp_dir = f'./result/{args.dataset_name}/classification_4class_react_additive'
    task = "supcon_hierar"
    args.checkpoint = f"./result/uspto_1K_TPL_backward/final/{task}/model_pretrain_best_mAP.pt"
    args.shared_vocab = True
    
    dt = datetime.now()
    args.exp_dir = os.path.join(args.exp_dir, "cond_adapter", '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second))
    os.makedirs(args.exp_dir, exist_ok=True)
    logger = init_logger(os.path.join(args.exp_dir, "log.txt"))
    with open(os.path.join(args.exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    main(args)

# python finetune_yield_cls.py --num_runs 30 --dataset_name buchward/dy/yield_gnn --lr 1e-3 --encoder_lr 1e-4 --decoder_hidden_size 512 256 --dropout 0.2 --device cuda:x
# python finetune_yield_cls.py --num_runs 1 --decoder_hidden_size 512 256 --dropout 0.2 --lr 1e-3 --encoder_lr 1e-4 --ffn_mode adapter --attn_mode adapter --ffn_option sequential --device cuda:7 --dataset_name reaxys_yield
