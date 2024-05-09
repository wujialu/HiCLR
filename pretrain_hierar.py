import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import json
from models.model import ProjectNet
from utils.build_utils import build_forward_iterator, build_retro_iterator, build_model, load_checkpoint, accumulate_batch_pretrain, set_random_seed
from utils.logging import init_logger, TensorboardLogger
from utils.loss_utils import LabelSmoothingLoss, SupConLoss, HMLC, SelfPacedSupConLoss, SelfPacedHMLC, is_normalized
from datetime import datetime
from args_parse import args_parser

# filter warnings
import warnings
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


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


def generate_reaction_reps(args, model, proj, loader):
    model.eval()
    proj.eval()
    
    reaction_reps = []
    reaction_labels = []
    reaction_ids = []

    for i, batch in enumerate(tqdm(loader, desc="Iteration")):
        if len(batch) == 8: #* train_loader
            src, tgt, _, _, _, _, _, reaction_class = batch
        else:  #* test_loader
            src, tgt, reaction_class, reaction_id = batch

        src, tgt = src.to(args.device), tgt.to(args.device)

        # get token representations
        with torch.no_grad():
            generative_scores, context_scores, src_reps, tgt_reps = model(src, tgt)

        # feature projection (for retreival)
        reaction_reps_batch = proj(torch.cat([src_reps, tgt_reps], dim=-1))


        reaction_reps.append(reaction_reps_batch)
        reaction_labels.append(reaction_class)
        reaction_ids.append(reaction_id)

    reaction_reps = torch.cat(reaction_reps, dim=0)  #*[len_data, proj_dim]
    reaction_labels = torch.cat(reaction_labels, dim=0).to(args.device)  #*[len_data, label_hierarchy]
    reaction_ids = torch.cat(reaction_ids, dim=0).to(args.device)
    return reaction_reps, reaction_labels, reaction_ids


def calculate_map(query_feats, gallery_feats, query_labels, gallery_labels, r=-1):
    """
    query_feats: [num_query, n_dim]
    gallery_feats: [num_gallery, n_dim]
    query_labels: [num_query]
    gallery_labels: [num_gallery]
    r: mAP@R, default -1 means mAP@ALL
    """
    assert is_normalized(query_feats) and is_normalized(gallery_feats), f"features need to be normalized first"

    pair_cosine_sim = torch.matmul(query_feats, gallery_feats.t())  #*[num_query, num_gallery]
    del query_feats, gallery_feats
    torch.cuda.empty_cache()

    rank = torch.argsort(1-pair_cosine_sim, axis=1) 
    del pair_cosine_sim
    torch.cuda.empty_cache()

    retrieval_idx = rank[:, :r]  #*[num_query, r]
    retrieval_labels = gallery_labels.unsqueeze(0).repeat(retrieval_idx.shape[0], 1).gather(index=retrieval_idx, dim=1) #* [num_query, r]
    result = retrieval_labels.eq(query_labels.unsqueeze(1).repeat(1, r))
    correct_cnt = torch.stack([result[:,:i].sum(dim=1) for i in range(1, r+1)], dim=1).cpu()  #*[num_query, r]
    mAP = (correct_cnt / (torch.arange(r) + 1))
    return mAP


def train(args, model, proj, loader, query_loader, gallery_loader, optimizer):
    global max_mAP
    model.train()
    proj.train()
    
    pad_idx = model.embedding_tgt.word_padding_idx
    criterion_tokens = LabelSmoothingLoss(ignore_index=pad_idx,
                                          reduction='mean', apply_logsoftmax=False,  #! already apply logsoftmax in model.generator
                                          smoothing=args.label_smoothing)
    criterion_context_align = LabelSmoothingLoss(reduction='mean', smoothing=0.5) 
    
    if args.contrastive_loss_type == "supcon":
        criterion_rep = SupConLoss(temperature=args.tau, base_temperature=args.tau, supcon_level=args.supcon_level)
    elif args.contrastive_loss_type == "hmlc":
        criterion_rep = HMLC(temperature=args.tau, base_temperature=args.tau, loss_type=args.hmlc_loss_type, layer_penalty=args.layer_penalty)
    elif args.contrastive_loss_type == "selfpaced_supcon":
        criterion_rep = SelfPacedSupConLoss(temperature=args.tau, weight_update=args.sp_strategy, correct_grad=args.correct_grad)
    elif args.contrastive_loss_type == "selfpaced_hmlc":
        criterion_rep = SelfPacedHMLC(temperature=args.tau, base_temperature=args.tau, loss_type=args.hmlc_loss_type, weight_update=args.sp_strategy, correct_grad=args.correct_grad)
    
    contrast_loss_cnt = []
    token_loss_cnt = []
    context_align_loss_cnt = []
    pred_token_list = []
    gt_token_list = []

    true_batch = []
    entry_count, src_max_length, tgt_max_length = 0, 0, 0
    global step
    for batch in tqdm(loader, desc="Iteration"):
        if step > args.max_pretrain_step:
            print('Finish training.')
            break

        raw_src, raw_tgt, _, _, _, _, _, _ = batch  # len(batch)=11
        src_max_length = max(src_max_length, raw_src.shape[0])
        tgt_max_length = max(tgt_max_length, raw_tgt.shape[0])
        entry_count += raw_tgt.shape[1]

        if (src_max_length + tgt_max_length) * entry_count < args.batch_size_token:
            true_batch.append(batch)
        else:
            #! set gamma for self-paced learning
            if args.contrastive_loss_type == "selfpaced_hmlc":
                curr_gamma = args.sp_gamma_min + (args.sp_gamma_max - args.sp_gamma_min) * ((step / args.max_pretrain_step) ** 0.5)
                criterion_rep.set_gamma(curr_gamma)

            # Accumulate Batch
            src, tgt, gt_context_alignment, src_1, tgt_1, src_2, tgt_2, reaction_class = accumulate_batch_pretrain(true_batch)
            src, tgt, gt_context_alignment, src_1, tgt_1, src_2, tgt_2, reaction_class =  \
                src.to(args.device), tgt.to(args.device), gt_context_alignment.to(args.device), \
                src_1.to(args.device), tgt_1.to(args.device), src_2.to(args.device), tgt_2.to(args.device),  \
                reaction_class.to(args.device)
            del true_batch
            torch.cuda.empty_cache()
            
            # get token representations (autoregressive loss on cano_smiles?)
            generative_scores, context_scores, _, _ = model(src, tgt, template_pooling=False)
            del src
            torch.cuda.empty_cache()
            # _, _, src_reps_1, tgt_reps_1, src_template_reps_1, tgt_template_reps_1 = \
            #     model(src_1, tgt_1, src_template_mask=src_template_mask_1, tgt_template_mask=tgt_template_mask_1, template_pooling=True)
            # del src_1, tgt_1, src_template_mask_1, tgt_template_mask_1
            # torch.cuda.empty_cache()
            # _, _, src_reps_2, tgt_reps_2, src_template_reps_2, tgt_template_reps_2 = \
            #     model(src_2, tgt_2, src_template_mask=src_template_mask_2, tgt_template_mask=tgt_template_mask_2, template_pooling=True)
            # del src_2, tgt_2, src_template_mask_2, tgt_template_mask_2
            # torch.cuda.empty_cache()

            _, _, src_reps_1, tgt_reps_1 = model(src_1, tgt_1)
            _, _, src_reps_2, tgt_reps_2 = model(src_2, tgt_2)
            del src_1, tgt_1, src_2, tgt_2
            torch.cuda.empty_cache()

            # feature projection
            #* reaction_reps: [batch_size, model_dim]
            #* norm: [batch_size, 1]
            src_reps = {"1": src_reps_1, "2": src_reps_2}
            tgt_reps = {"1": tgt_reps_1, "2": tgt_reps_2}
            # src_template_reps = {"1": src_template_reps_1, "2": src_template_reps_2}
            # tgt_template_reps = {"1": tgt_template_reps_1, "2": tgt_template_reps_2}
            reaction_reps = []
            template_reps = []
            reaction_reps_norm = []
            template_reps_norm = []

            if args.reaction_rep_mode == "concat":
                for i, key in enumerate(src_reps.keys()):
                    reaction_reps.append(proj(torch.cat([src_reps[key], tgt_reps[key]], dim=-1)))
                    reaction_reps_norm.append(reaction_reps[i] / torch.norm(reaction_reps[i], dim=-1).unsqueeze(1))
                    # template_reps.append(proj(torch.cat([src_template_reps[key], tgt_template_reps[key]], dim=-1)))
                    # template_reps_norm.append(template_reps[i] / torch.norm(template_reps[i], dim=-1).unsqueeze(1))
            elif args.reaction_rep_mode == "substract":
                for i, key in enumerate(src_reps.keys()):
                    reaction_reps.append(proj(src_reps[key] - tgt_reps[key]))
                    reaction_reps_norm.append(reaction_reps[i] / torch.norm(reaction_reps[i], dim=-1).unsqueeze(1))
                    # template_reps.append(proj(src_template_reps[key] - tgt_template_reps[key]))
                    # template_reps_norm.append(template_reps[i] / torch.norm(template_reps[i], dim=-1).unsqueeze(1))
            
            #* [batch_size, n_views, model_dim]
            reaction_reps_norm = torch.stack(reaction_reps_norm, dim=1) # add a dimension of n_views
            # template_reps_norm = torch.stack(template_reps_norm, dim=1)

            # del src_reps_1, tgt_reps_1, src_template_reps_1, tgt_template_reps_1
            # del src_reps_2, tgt_reps_2, src_template_reps_2, tgt_template_reps_2
            # del reaction_reps, template_reps, src_reps, tgt_reps
            # torch.cuda.empty_cache()

            # contrastive learning loss
            if args.supervised_scale == "reaction":
                contrast_loss = criterion_rep(features=reaction_reps_norm, labels=reaction_class)
            elif args.supervised_scale == "template":
                contrast_loss = criterion_rep(features=template_reps_norm, labels=reaction_class)

            # language modeling loss
            pred_token_logit = generative_scores.view(-1, generative_scores.size(2))
            _, pred_token_label = pred_token_logit.topk(1, dim=-1)
            gt_token_label = tgt[1:].view(-1)
            token_loss = criterion_tokens(pred_token_logit, gt_token_label)

            # smiles-alignment loss
            is_inferred = (gt_context_alignment.sum(dim=-1) == 0)
            gt_context_align_label = gt_context_alignment[~is_inferred].view(-1, gt_context_alignment.shape[-1])
            context_score = context_scores[-1]
            pred_context_align_logit = context_score[~is_inferred].view(-1, context_score.shape[-1])

            context_align_loss = criterion_context_align(pred_context_align_logit,
                                                         gt_context_align_label) 

            contrast_loss_cnt.append(contrast_loss.item())
            token_loss_cnt.append(token_loss.item())    
            context_align_loss_cnt.append(context_align_loss.item())      
            pred_token_list.append(pred_token_label[gt_token_label != pad_idx])
            gt_token_list.append(gt_token_label[gt_token_label != pad_idx])

            # optimization
            optimizer.zero_grad()
            loss = args.contrast_loss_coeff * contrast_loss + args.token_loss_coeff * token_loss + context_align_loss
            loss.backward()
            optimizer.step()

            if step % args.report_per_step == 0:
                pred_tokens = torch.cat(pred_token_list).view(-1)
                gt_tokens = torch.cat(gt_token_list).view(-1)
                token_acc = (pred_tokens == gt_tokens).float().mean().item()

                args.logger.info(
                    'iteration: %d, contrastive_loss: %f, token_loss: %f, accuracy_token: %f, context_align_loss: %f' % (
                        step, np.mean(contrast_loss_cnt), np.mean(token_loss_cnt), token_acc, np.mean(context_align_loss_cnt)))

                args.tensorboard_logger.add_scalar(key="Train/contrastive_loss", value=round(np.mean(contrast_loss_cnt), 4), use_context=False)
                args.tensorboard_logger.add_scalar(key="Train/token_loss", value=round(np.mean(token_loss_cnt), 4), use_context=False)
                
            if step % args.val_per_step == 0:
                # evalute retrieval performance
                mAP = evaluate(args, model, proj, query_loader, gallery_loader)[0]
                args.logger.info("Reaction retrieval mAP@10: %f" % (mAP))
                args.tensorboard_logger.add_scalar(key="mAP", value=round(mAP, 4), use_context=False)
                if mAP > max_mAP:
                    max_mAP = mAP
                    # save curr checkpoint
                    checkpoint_path = os.path.join(args.exp_dir, f"model_pretrain_best_mAP.pt")
                    torch.save({'model': model.state_dict(), 'proj': proj.state_dict(), 'step': step, 'optim': optimizer.state_dict()}, checkpoint_path)
                    args.logger.info('Checkpoint saved to {}'.format(checkpoint_path))

                model.train()
                proj.train()

            if not args.debug and step % args.save_per_step == 0:
                # save curr checkpoint
                checkpoint_path = os.path.join(args.exp_dir, f"model_pretrain_{step}.pt")
                torch.save({'model': model.state_dict(), 'proj': proj.state_dict(), 'step': step, 'optim': optimizer.state_dict()}, checkpoint_path)
                args.logger.info('Checkpoint saved to {}'.format(checkpoint_path))
                    
                contrast_loss_cnt = []
                token_loss_cnt = []
                
            # Restart Accumulation
            step += 1
            true_batch = [batch]
            entry_count, src_max_length, tgt_max_length = raw_src.shape[1], raw_src.shape[0], raw_tgt.shape[0]


def evaluate(args, model, proj, query_loader, gallery_loader, r=10):
    """
    1. generate reaction reps using current model 
    2. reaction retrieval (for each query reaction, retrieval top-10 reactions in gallery set)
    3. calculate mAP@R, R=10
    """

    gallery_reps, gallery_labels, gallery_ids = generate_reaction_reps(args, model, proj, gallery_loader)
    query_reps, query_labels, query_ids = generate_reaction_reps(args, model, proj, query_loader)

    # exist duplicates when using hierar_batch_sampler
    # _, unique_indices = unique(train_labels[:, 2], dim=0)
    # train_reps = train_reps[unique_indices]
    # train_labels = train_labels[unique_indices]

    gallery_reps_norm = gallery_reps / torch.norm(gallery_reps, dim=-1).unsqueeze(1) #*[len_train, n_dim] 
    query_reps_norm = query_reps / torch.norm(query_reps, dim=-1).unsqueeze(1) #*[len_test, n_dim]

    # calculate map at template-class level
    # for r_ in range(1, r+1):
    #     mAP = calculate_map(query_feats=query_reps_norm, gallery_feats=gallery_reps_norm, query_labels=query_labels, gallery_labels=gallery_labels, r=r_)
    #     print(f"mAP@{r_}: {mAP.mean().item()}")

    mAP = calculate_map(query_feats=query_reps_norm, gallery_feats=gallery_reps_norm, query_labels=query_labels, gallery_labels=gallery_labels, r=r)
    print(f"mAP@{r}: {mAP.mean().item()}")

    return mAP.mean().item(), query_ids


def main(args):
    # load data and build iterator
    if args.mode == "forward":
        args.shared_vocab = False
        train_iter, vocab_itos_src, vocab_itos_tgt = \
            build_forward_iterator(args, mode="pretrain", sample=False, augment=True)
    elif args.mode == "backward":
        args.shared_vocab = True

        train_iter, vocab_itos_src, vocab_itos_tgt = \
            build_retro_iterator(args, mode="pretrain", sample=False, augment=True, hierar_sampling=args.hierar_sampling)
            
        args.data_file = "retrieval_gallery_set"
        gallery_iter, _ = build_retro_iterator(args, mode="test", sample=False, augment=False, hierar_sampling=False)
        args.data_file = "retrieval_query_set"
        query_iter, _ = build_retro_iterator(args, mode="test", sample=False, augment=False, hierar_sampling=False)

    # build model
    model = build_model(args, vocab_itos_src, vocab_itos_tgt)

    if args.reaction_rep_mode == "concat":
        input_dim = args.d_model*2
    else:
        input_dim = args.d_model
    if args.proj_mode == "linear":
        proj = ProjectNet(input_dim=input_dim, output_dim=args.d_model).to(args.device)
    else:
        proj = ProjectNet(input_dim=input_dim, hidden_dim=args.d_model, output_dim=args.d_model).to(args.device) 

    n_params, enc, dec = _tally_parameters(model)
    args.logger.info('encoder: %d' % enc)
    args.logger.info('decoder: %d' % dec)
    args.logger.info('* number of parameters: %d' % n_params)

    # build optimizer
    model_param_group = [{"params": model.parameters(), "lr": args.lr},
                         {"params": proj.parameters(), "lr": args.lr}]
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay, eps=1e-9)

    # Do pretraining with only local objective
    # report iteration, train loss and classification/token prediction accuracy
    global step 
    step = 1
    global max_mAP
    max_mAP = 0.

    if args.checkpoint:
        step, model, proj, optimizer = load_checkpoint(args.checkpoint, model, proj, optimizer)
        step += 1

    for epoch in range(1, args.epochs + 1):
        args.logger.info("====epoch " + str(epoch))
        train(args, model, proj, train_iter, query_iter, gallery_iter, optimizer)

    # save logged data
    args.tensorboard_logger.save(os.path.join(args.exp_dir, 'logged_data.pkl.gz'))


if __name__ == "__main__":
    args = args_parser()
    
    dt = datetime.now()
    args.exp_dir = os.path.join(args.exp_dir, '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second))
    os.makedirs(args.exp_dir, exist_ok=True)
    print(f"Saving result to: {args.exp_dir}")

    with open(os.path.join(args.exp_dir, 'config_pretrain.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    if args.enable_tensorboard:
        args.tensorboard_logger = TensorboardLogger(args)
    log_file_name = os.path.join(args.exp_dir, "log.txt")
    args.logger = init_logger(log_file_name)

    set_random_seed(args.seed)
    
    main(args)
