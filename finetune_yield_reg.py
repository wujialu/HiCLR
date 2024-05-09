import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from models.model import MLP, YieldNet
from utils.build_utils import build_retro_iterator, build_model, load_checkpoint_downstream, set_random_seed, load_checkpoint_downstream
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
from datetime import datetime
import json
from utils.logging import init_logger, TensorboardLogger
from utils.loss_utils import weighted_mse_loss, weighted_focal_mse_loss, BMCLoss
from args_parse import args_parser


def train(args, encoder, decoder, optimizer, loader, criterion): 
    encoder.train()
    decoder.train()

    train_loss = []

    for batch in tqdm(loader, desc="Iteration"):
        src, tgt, rt, ids, reagents, weights = batch 
        src, tgt, rt, reagents, weights = src.to(args.device), tgt.to(args.device), rt.to(args.device), reagents.to(args.device), weights.to(args.device)
        del batch
        torch.cuda.empty_cache()

        # get token representations
        src_reps, tgt_reps = encoder.extract_reaction_fp(src, tgt, cond=reagents)
        # reaction_fp = torch.cat([src_reps, tgt_reps, reagents], dim=-1)
        reaction_fp = torch.cat([src_reps, tgt_reps], dim=-1)
        logits = decoder(reaction_fp)

        optimizer.zero_grad()
            
        if "weighted" in args.criterion:
            loss = criterion(logits.view(-1), (rt/100).clamp(0, 1), weights)
        else:
            loss = criterion(logits.view(-1), (rt/100).clamp(0, 1))
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
        src, tgt, rt, ids, reagents, weights = batch 
        src, tgt, rt, reagents, weights = src.to(args.device), tgt.to(args.device), rt.to(args.device), reagents.to(args.device), weights.to(args.device)
        del batch
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            src_reps, tgt_reps = encoder.extract_reaction_fp(src, tgt, cond=reagents)
            reaction_fp = torch.cat([src_reps, tgt_reps], dim=-1)
            logits = decoder(reaction_fp)

        if "weighted" in args.criterion:
            loss = criterion(logits.view(-1), (rt/100).clamp(0, 1), weights)
        else:
            loss = criterion(logits.view(-1), (rt/100).clamp(0, 1))
        eval_loss.append(loss.item())
        y_true.append(rt)
        y_pred_batch = logits.squeeze()
        if len(y_pred_batch.shape) == 0:
            y_pred_batch = y_pred_batch.unsqueeze(0)
        y_pred.append(y_pred_batch)

    y_true = torch.cat(y_true).clip(0, 100).detach().cpu()
    y_pred = (100 * torch.cat(y_pred)).clip(0, 100).detach().cpu()
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    return np.mean(eval_loss).round(4), r2.round(4), rmse.round(4), mae.round(4), y_true, y_pred


def main(args):
    train_r2_all, val_r2_all, test_r2_all = [], [], []
    train_rmse_all, val_rmse_all, test_rmse_all = [], [], []
    train_mae_all, val_mae_all, test_mae_all = [], [], []
    y_true_top10_mean, y_true_top10_std, y_pred_top10_mean, y_pred_top10_std = [], [], [], []
    
    # loss function
    if args.criterion == "mse":
        criterion = nn.MSELoss(reduction="mean")
    elif args.criterion == "weighted_mse":
        criterion = weighted_mse_loss
    elif args.criterion == "weighted_focal_mse" or args.criterion == "focal_mse":
        criterion = weighted_focal_mse_loss
    elif args.criterion == "bmc":
        criterion = BMCLoss(1.)
    
    #! pre-training
    if args.do_pretrain:
        args.data_dir = './data/buchward/dy'
        args.data_file = "processed"
        train_iter, train_dataset = \
            build_retro_iterator(args, mode="test_yield", sample=False, augment=False, shuffle=True)  
            
        encoder = build_model(args, train_dataset.src_itos, train_dataset.tgt_itos)
        args.size_layer = [2*encoder.d_model] + args.decoder_hidden_size + [args.num_class]
        decoder = MLP(size_layer=args.size_layer, dropout=args.dropout).to(args.device)
        
        if args.pretrain_checkpoint is not None:
            encoder = load_checkpoint_downstream(args.pretrain_checkpoint, encoder)

        model_param_group = [{"params": decoder.parameters()}]
        if not args.fix_encoder:
            if args.ffn_mode != "none" or args.attn_mode != "none":
                for name, parameter in encoder.named_parameters():
                    if "adapter" in name or "cond" in name: # cond_attn & lin_cond
                        parameter.requires_grad = True
                    else:
                        parameter.requires_grad = False
                model_param_group += [{"params": filter(lambda p: p.requires_grad, encoder.parameters()), "lr": args.encoder_lr}]
            else:
                model_param_group += [{"params": encoder.parameters(), "lr": args.encoder_lr}]
        
        optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
        if args.criterion == "bmc":
            optimizer.add_param_group({'params': criterion.noise_sigma, 'lr': args.encoder_lr, 'name': 'noise_sigma'})

        for epoch in tqdm(range(args.pretrain_epochs), desc="Epoch"):
            train_loss = train(args, encoder, decoder, optimizer, train_iter, criterion)
        args.yield_checkpoint = os.path.join(args.exp_dir, f"HTE_pretrain.pt")
        torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict(), 'step': epoch + 1, 'optim': optimizer.state_dict()}, args.yield_checkpoint)
            
        args.data_dir = './data/buchward/az_test0.8/10_random_splits'
    
    #! training
    for i in range(0, args.num_runs):
        logger.info(f"----------Round {i + 1}, Seed {args.seed + i}----------")
        # set random seed
        set_random_seed(args.seed + i)
        # data_name = NAME_SPLIT[i][0]
        
        # build_iterator
        # args.data_file = f"train_{data_name}"
        args.data_file = f"train_random_split_{i}"
        train_iter, train_dataset = \
            build_retro_iterator(args, mode="test_yield", sample=False, augment=False, shuffle=True)  
        # args.data_file = f"test_{data_name}"
        args.data_file = f"test_random_split_{i}"
        val_iter, _ = \
            build_retro_iterator(args, mode="test_yield", sample=False, augment=False)
        
        # y_mean = train_dataset.y_mean
        # y_std = train_dataset.y_std
        
        # load pre-trained encoder (mol_transformer)
        encoder = build_model(args, train_dataset.src_itos, train_dataset.tgt_itos)
        args.size_layer = [2*encoder.d_model] + args.decoder_hidden_size + [args.num_class]
        # args.size_layer = [2*encoder.d_model] + [args.num_class]
        #! w/o output_activation is better than w/ sigmoid
        decoder = MLP(size_layer=args.size_layer, dropout=args.dropout).to(args.device)
        # decoder = YieldNet(dims=[256,256,256]).to(args.device)
        # ecfp_transform = MLP(size_layer=[1024, 512, 512], dropout=args.dropout).to(args.device)

        if args.do_pretrain:
            encoder = load_checkpoint_downstream(args.yield_checkpoint, encoder, model_type="encoder")
            decoder = load_checkpoint_downstream(args.yield_checkpoint, decoder, model_type="decoder")
        else:
            if args.yield_checkpoint is not None: # transfer leanring between yield datasets
                encoder = load_checkpoint_downstream(args.yield_checkpoint, encoder, model_type="encoder")
                decoder = load_checkpoint_downstream(args.yield_checkpoint, decoder, model_type="decoder")
            elif args.pretrain_checkpoint is not None:
                encoder = load_checkpoint_downstream(args.pretrain_checkpoint, encoder)

        model_param_group = [{"params": decoder.parameters()}]
        if not args.fix_encoder:
            if args.ffn_mode != "none" or args.attn_mode != "none":
                for name, parameter in encoder.named_parameters():
                    if "adapter" in name or "cond" in name or "norm" in name: # cond_attn & lin_cond & layer_norm
                        parameter.requires_grad = True
                    else:
                        parameter.requires_grad = False
                model_param_group += [{"params": filter(lambda p: p.requires_grad, encoder.parameters()), "lr": args.encoder_lr}]
            else:
                #! full fine-tuning
                model_param_group += [{"params": encoder.parameters(), "lr": args.encoder_lr}]
        
        optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
        checkpoint_path = os.path.join(args.exp_dir, f"model_round_{i + 1}.pt")
        
        # train and validate
        if args.do_train:
            best_val_r2 = -1.0
            counter = 0.
            for epoch in tqdm(range(args.epochs), desc="Epoch"):
                train_loss = train(args, encoder, decoder, optimizer, train_iter, criterion)
                val_loss, val_r2, val_rmse, val_mae, _, _ = evaluate(args, encoder, decoder, val_iter, criterion)
                logger.info(f"Epoch: {epoch + 1}, Train loss: {train_loss}, Valid loss: {val_loss}, Valid r2: {val_r2}, Valid rmse: {val_rmse}, Valid mae: {val_mae}")

                # if early-stopping, break
                if val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict(), 'step': epoch + 1, 'optim': optimizer.state_dict()}, checkpoint_path)
                    counter = 0.
                else:
                    counter += 1
                    if counter >= args.patience:
                        logger.info(f"----------Early-stopping at epoch {epoch + 1}----------")
                        break

        # test and report classification result
        encoder = load_checkpoint_downstream(checkpoint_path, encoder, model_type="encoder")
        decoder = load_checkpoint_downstream(checkpoint_path, decoder, model_type="decoder")
        _, train_r2, train_rmse, train_mae, _, _ = evaluate(args, encoder, decoder, train_iter, criterion)
        _, val_r2, val_rmse, val_mae, y_true, y_pred = evaluate(args, encoder, decoder, val_iter, criterion)
        # _, test_r2, test_rmse, test_mae = evaluate(args, encoder, decoder, test_iter, criterion)
        logger.info(f"Round: {i + 1}, Train r2: {train_r2}, Train rmse: {train_rmse}, Train mae: {train_mae}, Valid r2: {val_r2}, Valid rmse: {val_rmse}, Valid mae: {val_mae}")
        train_r2_all.append(train_r2)
        train_rmse_all.append(train_rmse)
        train_mae_all.append(train_mae)
        val_r2_all.append(val_r2)
        val_rmse_all.append(val_rmse)
        val_mae_all.append(val_mae)
        # test_r2_all.append(test_r2)
        # test_rmse_all.append(test_rmse)
        # test_mae_all.append(test_mae)
        
        # report mean and std of top-10 reactions 
        y_true_top10 = torch.topk(y_true, 10)[0]
        y_pred_top10 = y_true[torch.topk(y_pred, 10)[1]]
        logger.info("Top-10 true highest yield: {:.2f} ± {:.2f}".format(y_true_top10.mean(), y_true_top10.std()))
        logger.info("Top-10 predicted highest yield: {:.2f} ± {:.2f}".format(y_pred_top10.mean(), y_pred_top10.std()))
        y_true_top10_mean.append(y_true_top10.mean().item())
        y_true_top10_std.append(y_true_top10.std().item())
        y_pred_top10_mean.append(y_pred_top10.mean().item())
        y_pred_top10_std.append(y_pred_top10.std().item())
        
        #! save predictions
        df = pd.DataFrame()
        df["y_true"] = list(y_true.numpy())
        df["y_pred"] = list(y_pred.numpy())
        df.to_csv(os.path.join(args.exp_dir, f"test_pred_{i}.csv"), index=False)
        
    # report and save final result
    train_r2_mean, train_r2_std, train_rmse_mean, train_rmse_std, train_mae_mean, train_mae_std = \
        np.mean(train_r2_all).round(4), np.std(train_r2_all).round(4), np.mean(train_rmse_all).round(4), np.std(train_rmse_all).round(4), np.mean(train_mae_all).round(4), np.std(train_mae_all).round(4)
    val_r2_mean, val_r2_std, val_rmse_mean, val_rmse_std, val_mae_mean, val_mae_std = \
        np.mean(val_r2_all).round(4), np.std(val_r2_all).round(4), np.mean(val_rmse_all).round(4), np.std(val_rmse_all).round(4), np.mean(val_mae_all).round(4), np.std(val_mae_all).round(4)
    # test_r2_mean, test_r2_std, test_rmse_mean, test_rmse_std, test_mae_mean, test_mae_std = \
    #     np.mean(test_r2_all).round(4), np.std(test_r2_all).round(4), np.mean(test_rmse_all).round(4), np.std(test_rmse_all).round(4), np.mean(test_mae_all).round(4), np.std(test_mae_all).round(4)
    logger.info(f"train_result: r2: {train_r2_mean} ± {train_r2_std}, rmse: {train_rmse_mean} ± {train_rmse_std}, mae: {train_mae_mean} ± {train_mae_std}")
    logger.info(f"val_result: r2: {val_r2_mean} ± {val_r2_std}, rmse: {val_rmse_mean} ± {val_rmse_std}, mae: {val_mae_mean} ± {val_mae_std}")
    # logger.info(f"test_result: r2: {test_r2_mean} ± {test_r2_std}, rmse: {test_rmse_mean} ± {test_rmse_std}, mae: {test_mae_mean} ± {test_mae_std}")

    result = pd.DataFrame()
    result["test_r2"] = val_r2_all
    result["test_mae"] = val_mae_all
    result["y_true_top10_mean"] = y_true_top10_mean
    result["y_true_top10_std"] = y_true_top10_std
    result["y_pred_top10_mean"] = y_pred_top10_mean
    result["y_pred_top10_std"] = y_pred_top10_std
    result.to_csv(os.path.join(args.exp_dir, "result.csv"), index=False)
    
if __name__ == "__main__":
    args = args_parser()
    dt = datetime.now()
    if args.fix_encoder:
        setting = "fix_encoder"
    else:
        if args.ffn_mode != "none" or args.attn_mode != "none":
            setting = "peft"
        else:
            setting = "full_finetuning"
    
    if args.do_train:
        args.exp_dir = os.path.join(args.exp_dir, "wo_scaler", setting, "cond_adapter", '{}_{:02d}-{:02d}-{:02d}'.format(
            dt.date(), dt.hour, dt.minute, dt.second))
    else:
        # args.exp_dir = "./result/buchward/az/30_random_splits/wo_scaler/peft/mse/2024-04-22_18-02-59"
        args.exp_dir = os.path.join(args.exp_dir, setting, "debug")
        
    os.makedirs(args.exp_dir, exist_ok=True)
    args.shared_vocab = True
    
    task = "supcon_hierar"
    args.pretrain_checkpoint = f"./checkpoint/{task}/model_pretrain_best_mAP.pt"
    args.yield_checkpoint = None
    
    logger = init_logger(os.path.join(args.exp_dir, "log.txt"))
    if args.do_train:
        with open(os.path.join(args.exp_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
 
    main(args)

# python finetune_az_yields_wo_scaler.py --device cuda:2 --dropout 0.2 --lr 1e-3 --encoder_lr 1e-4 --decoder_hidden_size 512 256 --num_runs 30