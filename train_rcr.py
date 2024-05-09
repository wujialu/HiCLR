import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import pickle
from collections import OrderedDict
from pickletools import optimize
import random
import time
from matplotlib import use
from matplotlib.pyplot import savefig
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
import yaml
from torch.utils.data import Dataset, DataLoader
import json

from models.condition_model import RXNConditionModel
from args_parse import args_parser
from utils.smiles_utils import smi_tokenizer
from utils.build_utils import build_model, load_vocab
from functools import partial

# from torch.utils.tensorboard import SummaryWriter
# import wandb
# wandb.init(project="baseline_condition_model", entity="wangxr")

work_space = os.path.dirname(__file__)


def load_dataset(data_root, database_name, src_stoi, tgt_stoi, use_temperature=False):
    print('Loading condition data from {}'.format(data_root))

    print('Loading database dataframe...')
    database_df = pd.read_csv(os.path.join(data_root, f'{database_name}.csv'))
    print('Loaded {} data'.format(len(database_df)))

    print('Loading calculated fps...')
    prod_fps = np.load(os.path.join(
        data_root, f'{database_name}_prod_fps.npz'))['fps']
    rxn_fps = np.load(os.path.join(
        data_root, f'{database_name}_rxn_fps.npz'))['fps'] #! replace the rxn_fps with our fps (fixed or fine-tune)
    print('prod_fps shape:', prod_fps.shape)
    print('rxn_fps shape:', rxn_fps.shape)
    print('########################################################')
    prod_fps_dict = {}
    rxn_fps_dict = {}

    rxn_dict = {}
    if use_temperature:
        temperature_dict = {}
    for dataset in list(set(database_df['dataset'].tolist())):  #! split into train/val/test
        dataset_prod_fps = prod_fps[database_df.loc[database_df['dataset'] == dataset].index]
        prod_fps_dict[dataset] = dataset_prod_fps
        dataset_rxn_fps = rxn_fps[database_df.loc[database_df['dataset'] == dataset].index]
        rxn_fps_dict[dataset] = dataset_rxn_fps
        print('{} prod_fps shape: {}'.format(dataset, dataset_prod_fps.shape))
        print('{} rxn_fps shape: {}'.format(dataset, dataset_rxn_fps.shape))
        
        print("Tokenizing rxn_smiles...")
        rxn_smiles = database_df.loc[database_df['dataset'] == dataset]['canonical_rxn'].values
        rxn_tokens = []
        for rxn in tqdm(rxn_smiles):
            reacts, prod = rxn.split(">>")
            src_tokens = ["<UNK>"] + smi_tokenizer(prod)
            tgt_tokens =  ['<sos>'] + smi_tokenizer(reacts) + ['<eos>'] 

            src_token_ids = [src_stoi.get(t, src_stoi['<unk>']) for t in src_tokens]
            tgt_token_ids = [tgt_stoi.get(t, tgt_stoi['<unk>']) for t in tgt_tokens]

            rxn_tokens.append([src_token_ids, tgt_token_ids])
        rxn_dict[dataset] = rxn_tokens

        if use_temperature:
            dataset_temperature = database_df.loc[database_df['dataset']
                                                  == dataset]['temperature'].values
            temperature_dict[dataset] = dataset_temperature
            print('{} temperature array shape: {}'.format(
                dataset, dataset_temperature.shape))
        print('########################################################')
         
    print('Loading label dict...')
    label_dict = OrderedDict()
    for name in ['catalyst1', 'solvent1', 'solvent2', 'reagent1', 'reagent2']:
        with open(os.path.join(data_root, '{}_{}.pkl'.format(database_name, name)), 'rb') as f:
            _label_dic = pickle.load(f)
        print('Condition name: {}, categories: {}'.format(
            name, len(_label_dic[0])))
        name_data = database_df[name]
        name_data[pd.isna(name_data)] = ''
        name_labels = {}
        for dataset in list(set(database_df['dataset'].tolist())):
            condition2label = _label_dic[1]
            name_labels[dataset] = [condition2label[x]
                                    for x in name_data.loc[database_df['dataset'] == dataset].tolist()]
            name_labels[dataset] = np.array(name_labels[dataset])
            print('{}: {}'.format(dataset, len(name_labels[dataset])))

        label_dict[name] = [_label_dic, name_labels]
        print('########################################################')
    if use_temperature:
        label_dict['temperature'] = temperature_dict
    fps_dict = {'prod_fps': prod_fps_dict, 'rxn_fps': rxn_fps_dict}
    return fps_dict, label_dict, rxn_dict


class ConditionDataset(Dataset):
    def __init__(self, fps_dict, label_dict, dataset, use_temperature, rxn_dict=None):
        self.prod_fps = torch.tensor(fps_dict['prod_fps'][dataset])
        self.rxn_fps = torch.tensor(fps_dict['rxn_fps'][dataset])
        self.use_rxn_smiles = False
        if rxn_dict is not None:
            self.rxn_smiles = rxn_dict[dataset]
            self.use_rxn_smiles = True

        self.use_temperature = use_temperature
        self.label_dict = {}
        self.condition2label = {}
        self.label2condition = {}
        self.label_names = ['catalyst1', 'solvent1',
                            'solvent2', 'reagent1', 'reagent2']
        for name in self.label_names:
            self.label_dict[name] = torch.tensor(label_dict[name][1][dataset])
            self.label2condition[name] = label_dict[name][0][0]
            self.condition2label[name] = label_dict[name][0][1]
        # self.fp_dim = self.prod_fps.size(1)
        self.fp_dim = 256
        self.c1_dim = len(self.condition2label['catalyst1'])
        self.s1_dim = len(self.condition2label['solvent1'])
        self.s2_dim = len(self.condition2label['solvent2'])
        self.r1_dim = len(self.condition2label['reagent1'])
        self.r2_dim = len(self.condition2label['reagent2'])
        if use_temperature:
            self.temperature = torch.tensor(
                label_dict['temperature'][dataset]).float()

    def __len__(self):
        return self.prod_fps.shape[0]

    def __getitem__(self, index):
        pfp = self.prod_fps[index]
        rfp = self.rxn_fps[index]  #! rfp means react_fps in our implementation
        labels = {}
        for name in self.label_names:
            labels[name] = self.label_dict[name][index]
        if self.use_rxn_smiles:
            src, tgt = self.rxn_smiles[index]
            return src, tgt, labels['catalyst1'], labels['solvent1'], labels['solvent2'], labels['reagent1'], labels['reagent2']
        elif self.use_temperature:
            return pfp, rfp, labels['catalyst1'], labels['solvent1'], labels['solvent2'], labels['reagent1'], labels['reagent2'], self.temperature[index]
        else:
            return pfp, rfp, labels['catalyst1'], labels['solvent1'], labels['solvent2'], labels['reagent1'], labels['reagent2']


def get_one_hot_input(label, dim):
    onehot = F.one_hot(label, num_classes=dim)
    return onehot.float()


def caculate_weighted_loss(loss, weights=1):
    loss = loss * weights
    return loss


def train_one_epoch(model, train_loader, loss_fn_list, optimizer, device, condition2label, it, loss_weight=[1, 1, 1, 1, 1, 0.0001], writer=None, use_temperature=False, 
                    rxn_encoder=None, finetune_rxn_encoder=False):
    #! using transformer to extract reaction fps
    model.train()
    model.not_softmax_out = True
    if finetune_rxn_encoder:
        rxn_encoder.train()
    else:
        rxn_encoder.eval()
    loss_all = 0.0
    losses = []
    for data in tqdm(train_loader):
        optimizer.zero_grad()
        loss = 0.0
        batch_loss_list = []
        data = [x.to(device) for x in data]
        if not use_temperature:
            pfp, rfp, c1, s1, s2, r1, r2 = data
        else:
            pfp, rfp, c1, s1, s2, r1, r2, t = data

        # src, tgt, c1, s1, s2, r1, r2 = data
        # pfp, rfp = rxn_encoder.extract_reaction_fp(src, tgt) 

        fp_emb = model.fp_func(pfp, rfp)
        # fp_emb = torch.cat([pfp, rfp], dim=1)
        
        loss_c1 = loss_fn_list[0](model.c1_func(fp_emb), c1)
        batch_loss_list += [loss_c1]
        input_c1 = get_one_hot_input(c1, dim=len(condition2label['catalyst1']))
        loss_s1 = loss_fn_list[0](model.s1_func(fp_emb, input_c1), s1)
        batch_loss_list += [loss_s1]
        input_s1 = get_one_hot_input(s1, dim=len(condition2label['solvent1']))
        loss_s2 = loss_fn_list[0](
            model.s2_func(fp_emb, input_c1, input_s1), s2)
        batch_loss_list += [loss_s2]
        input_s2 = get_one_hot_input(s2, dim=len(condition2label['solvent2']))
        loss_r1 = loss_fn_list[0](model.r1_func(
            fp_emb, input_c1, input_s1, input_s2), r1)
        batch_loss_list += [loss_r1]
        input_r1 = get_one_hot_input(r1, dim=len(condition2label['reagent1']))
        loss_r2 = loss_fn_list[0](model.r2_func(fp_emb, input_c1,
                                                input_s1, input_s2, input_r1), r2)
        batch_loss_list += [loss_r2]

        if use_temperature:
            input_r2 = get_one_hot_input(
                r2, dim=len(condition2label['reagent2']))
            loss_t = loss_fn_list[1](model.T_func(fp_emb, input_c1,
                                                  input_s1, input_s2, input_r1,
                                                  input_r2).squeeze(), t)
            batch_loss_list += [loss_t]

        for _loss, w in zip(batch_loss_list, loss_weight):
            loss += caculate_weighted_loss(_loss, w)
        loss /= len(batch_loss_list)
        # loss_c1.backward()
        # loss_s1.backward()
        # loss_s2.backward()
        # loss_r1.backward()
        # loss_r2.backward()
        # loss = (loss_c1 + loss_s1 + loss_s2 + loss_r1 + loss_r2) / 5.
        loss.backward()
        # loss = (loss_c1 + loss_s1 + loss_s2 + loss_r1 + loss_r2) / 5.
        optimizer.step()
        loss_all += loss.item() * c1.shape[0]
        losses.append(loss.item())
        it.set_postfix(loss=np.mean(losses[-10:]) if losses else None)
        # if writer:
        #     writer.add_scalar('training loss', np.mean(losses), epoch)
        torch.cuda.empty_cache()
    return loss_all / len(train_loader.dataset)


def validation(model, data_loader, loss_fn_list, device, condition2label, loss_weight=[1, 1, 1, 1, 1, 0.0001], use_temperature=False, rxn_encoder=None):
    model.eval()
    model.not_softmax_out = True
    rxn_encoder.eval()
    loss_all = 0.0
    temp_mse = 0.0
    for data in tqdm(data_loader):
        loss = 0.0
        batch_loss_list = []
        with torch.no_grad():
            data = [x.to(device) for x in data]
            if not use_temperature:
                pfp, rfp, c1, s1, s2, r1, r2 = data
            else:
                pfp, rfp, c1, s1, s2, r1, r2, t = data

            # src, tgt, c1, s1, s2, r1, r2 = data
            # pfp, rfp = rxn_encoder.extract_reaction_fp(src, tgt) 

            fp_emb = model.fp_func(pfp, rfp)

            loss_c1 = loss_fn_list[0](model.c1_func(fp_emb), c1)
            batch_loss_list += [loss_c1]
            input_c1 = get_one_hot_input(
                c1, dim=len(condition2label['catalyst1']))
            loss_s1 = loss_fn_list[0](model.s1_func(fp_emb, input_c1), s1)
            batch_loss_list += [loss_s1]
            input_s1 = get_one_hot_input(
                s1, dim=len(condition2label['solvent1']))
            loss_s2 = loss_fn_list[0](
                model.s2_func(fp_emb, input_c1, input_s1), s2)
            batch_loss_list += [loss_s2]
            input_s2 = get_one_hot_input(
                s2, dim=len(condition2label['solvent2']))
            loss_r1 = loss_fn_list[0](model.r1_func(
                fp_emb, input_c1, input_s1, input_s2), r1)
            batch_loss_list += [loss_r1]
            input_r1 = get_one_hot_input(
                r1, dim=len(condition2label['reagent1']))
            loss_r2 = loss_fn_list[0](model.r2_func(fp_emb, input_c1,
                                                    input_s1, input_s2, input_r1), r2)
            batch_loss_list += [loss_r2]
            if use_temperature:
                input_r2 = get_one_hot_input(
                    r2, dim=len(condition2label['reagent2']))

                loss_t = loss_fn_list[1](model.T_func(fp_emb, input_c1,
                                                      input_s1, input_s2, input_r1,
                                                      input_r2).squeeze(), t)
                batch_loss_list += [loss_t]

            for _loss, w in zip(batch_loss_list, loss_weight):
                loss += caculate_weighted_loss(_loss, w)
            loss /= len(batch_loss_list)
            loss_all += loss.item() * c1.shape[0]
            if use_temperature:
                temp_mse += loss_t.item() * c1.shape[0]
    if use_temperature:
        print('Temperature MSE: {}'.format(
            temp_mse / len(data_loader.dataset)))  # mse reduce 是默认的mean
    torch.cuda.empty_cache()
    return loss_all / len(data_loader.dataset)


def caculate_accuracy(model, data_loader, device, condition2label, rxn_encoder, topk_rank_thres=None, save_topk_path=None, use_temperature=False, top_fname=None, topk_get=[1, 3, 5, 10, 15], condition_to_calculate=['c1', 's1', 's2', 'r1', 'r2']):
    print('Caculating validataion topk accuracy...')
    if not topk_rank_thres:
        topk_rank_thres = {
            'c1': 1,
            's1': 3,
            's2': 1,
            'r1': 5,
            'r2': 1,
        }

    def get_accuracy_for_one(one_pred, one_ground_truth, topk_get=[1, 3, 5, 10, 15], condition_to_calculate=['c1', 's1', 's2', 'r1', 'r2']):
        
        condition_item2cols = {
            'c1':0, 's1':1, 's2':2, 'r1':3, 'r2':4
        }
        
        calculate_cols = [condition_item2cols[x] for x in condition_to_calculate]
        
        repeat_number = one_pred.size(0)
        hit_mat = one_ground_truth.unsqueeze(
            0).repeat(repeat_number, 1) == one_pred
        hit_mat = hit_mat[:, calculate_cols]
        overall_hit_mat = hit_mat.sum(1) == hit_mat.size(1)
        topk_hit_df = pd.DataFrame()
        for k in topk_get:
            hit_mat_k = hit_mat[:k, :]
            overall_hit_mat_k = overall_hit_mat[:k]
            topk_hit = []
            for col_idx in range(hit_mat.size(1)):
                if hit_mat_k[:, col_idx].sum() != 0:
                    topk_hit.append(1)
                else:
                    topk_hit.append(0)
            if overall_hit_mat_k.sum() != 0:
                topk_hit.append(1)
            else:
                topk_hit.append(0)
            topk_hit_df[k] = topk_hit
        # topk_hit_df.index = ['c1', 's1', 's2', 'r1', 'r2']
        return topk_hit_df

    model.eval()
    model.not_softmax_out = False  # 输出通过softmax激活函数
    rxn_encoder.eval()
    topk_acc_mat = np.zeros((len(condition_to_calculate) + 1, 5))
    closest_erro = 0.0
    for data in tqdm(data_loader):

        with torch.no_grad():

            data = [x.to(device) for x in data]
            if not use_temperature:
                pfp, rfp, c1, s1, s2, r1, r2 = data
            else:
                pfp, rfp, c1, s1, s2, r1, r2, t = data

            # src, tgt, c1, s1, s2, r1, r2 = data
            # pfp, rfp = rxn_encoder.extract_reaction_fp(src, tgt) 

            fp_emb = model.fp_func(pfp, rfp)
            one_batch_ground_truth = torch.cat([c1.unsqueeze(1), s1.unsqueeze(1), s2.unsqueeze(
                1), r1.unsqueeze(1), r2.unsqueeze(1)], dim=-1)
            # if device != torch.device('cpu'):
            #     one_batch_ground_truth = torch.cat([c1.unsqueeze(1), s1.unsqueeze(1), s2.unsqueeze(
            #         1), r1.unsqueeze(1), r2.unsqueeze(1)], dim=-1).cpu().numpy()
            # else:
            #     one_batch_ground_truth = torch.cat([c1.unsqueeze(1), s1.unsqueeze(1), s2.unsqueeze(
            #         1), r1.unsqueeze(1), r2.unsqueeze(1)], dim=-1).numpy()

            # one_batch_ground_truth_df = pd.DataFrame(one_batch_ground_truth)
            # one_batch_ground_truth_df.columns = [
            #     'ground_truth-c1', 'ground_truth-s1', 'ground_truth-s2', 'ground_truth-r1', 'ground_truth-r2', ]

            # teached_c1_pred = model.c1_func(fp_emb)
            # _, teached_c1_top_15 = teached_c1_pred.squeeze().topk(15)
            # input_c1 = get_one_hot_input(
            #     c1, dim=len(condition2label['catalyst1']))
            # teached_s1_pred = model.s1_func(fp_emb, input_c1)
            # _, teached_s1_top_15 = teached_s1_pred.squeeze().topk(15)
            # input_s1 = get_one_hot_input(
            #     s1, dim=len(condition2label['solvent1']))
            # teached_s2_pred = model.s2_func(fp_emb, input_c1, input_s1)
            # _, teached_s2_top_15 = teached_s2_pred.squeeze().topk(15)
            # input_s2 = get_one_hot_input(
            #     s2, dim=len(condition2label['solvent2']))
            # teached_r1_pred = model.r1_func(
            #     fp_emb, input_c1, input_s1, input_s2)
            # _, teached_r1_top_15 = teached_r1_pred.squeeze().topk(15)
            # input_r1 = get_one_hot_input(
            #     r1, dim=len(condition2label['reagent1']))
            # teached_r2_pred = model.r2_func(fp_emb, input_c1,
            #                                 input_s1, input_s2, input_r1)
            # _, teached_r2_top_15 = teached_r2_pred.squeeze().topk(15)

            # if device != torch.device('cpu'):
            #     teached_top_15 = torch.cat([teached_c1_top_15, teached_s1_top_15, teached_s2_top_15,
            #                                teached_r1_top_15, teached_r2_top_15, ], dim=-1).cpu().numpy()
            # else:
            #     teached_top_15 = torch.cat(
            #         [teached_c1_top_15, teached_s1_top_15, teached_s2_top_15, teached_r1_top_15, teached_r2_top_15, ], dim=-1).numpy()
            # one_batch_prediction_df = pd.DataFrame(teached_top_15)
            # one_batch_prediction_df.columns = ['teached_c1_top_{}'.format(x+1) for x in range(15)] + \
            #     ['teached_s1_top_{}'.format(x+1) for x in range(15)] + ['teached_s2_top_{}'.format(x+1) for x in range(15)] + \
            #     ['teached_r1_top_{}'.format(
            #         x+1) for x in range(15)] + ['teached_r2_top_{}'.format(x+1) for x in range(15)]
            # # one_batch_prediction_df = pd.concat([one_batch_prediction_df, one_batch_prediction_df_teached_top_15], axis=1)

            one_batch_preds = []
            one_batch_scores = []
            one_batch_t_preds = []

            c1_preds = model.c1_func(fp_emb)
            c1_scores, c1_cdts = c1_preds.squeeze().topk(topk_rank_thres['c1'])
            for c1_top in range(c1_cdts.size(-1)):
                # if device != torch.device('cpu'):
                #     pred_list = c1_cdts[:, c1_top].cpu().numpy().tolist()
                # else:
                #     pred_list = c1_cdts[:, c1_top].numpy().tolist()

                # one_batch_prediction_df['c1_top-{}'.format(c1_top+1)
                #                         ] = pred_list
                # c1_pred_list = pred_list
                c1_pred = c1_cdts[:, c1_top]
                c1_score = c1_scores[:, c1_top]

                c1_input = get_one_hot_input(
                    c1_pred, len(condition2label['catalyst1']))

                s1_preds = model.s1_func(fp_emb, c1_input)
                s1_scores, s1_cdts = s1_preds.squeeze().topk(
                    topk_rank_thres['s1'])
                for s1_top in range(s1_cdts.size(-1)):
                    # if device != torch.device('cpu'):
                    #     pred_list = s1_cdts[:, s1_top].cpu().numpy().tolist()
                    # else:
                    #     pred_list = s1_cdts[:, s1_top].numpy().tolist()
                    # one_batch_prediction_df['c1_top-{}_s1_top-{}'.format(c1_top+1, s1_top+1)
                    #                         ] = pred_list
                    # s1_pred_list = pred_list
                    s1_pred = s1_cdts[:, s1_top]
                    s1_score = s1_scores[:, s1_top]
                    s1_input = get_one_hot_input(
                        s1_pred, len(condition2label['solvent1']))

                    s2_preds = model.s2_func(fp_emb, c1_input, s1_input)
                    s2_scores, s2_cdts = s2_preds.squeeze().topk(
                        topk_rank_thres['s2'])
                    for s2_top in range(s2_cdts.size(-1)):
                        # if device != torch.device('cpu'):
                        #     pred_list = s2_cdts[:,
                        #                         s2_top].cpu().numpy().tolist()
                        # else:
                        #     pred_list = s2_cdts[:, s2_top].numpy().tolist()
                        # one_batch_prediction_df['c1_top-{}_s1_top-{}_s2_top-{}'.format(c1_top+1, s1_top+1, s2_top+1)
                        #                         ] = pred_list
                        # s2_pred_list = pred_list
                        s2_pred = s2_cdts[:, s2_top]
                        s2_score = s2_scores[:, s2_top]
                        s2_input = get_one_hot_input(
                            s2_pred, len(condition2label['solvent2']))

                        r1_preds = model.r1_func(
                            fp_emb, c1_input, s1_input, s2_input)
                        r1_scores, r1_cdts = r1_preds.squeeze().topk(
                            topk_rank_thres['r1'])
                        for r1_top in range(r1_cdts.size(-1)):
                            # if device != torch.device('cpu'):
                            #     pred_list = r1_cdts[:, r1_top].cpu(
                            #     ).numpy().tolist()
                            # else:
                            #     pred_list = r1_cdts[:, r1_top].numpy().tolist()
                            # one_batch_prediction_df['c1_top-{}_s1_top-{}_s2_top-{}_r1_top-{}'.format(c1_top+1, s1_top+1, s2_top+1, r1_top+1)
                            #                         ] = pred_list
                            # r1_pred_list = pred_list
                            r1_pred = r1_cdts[:, r1_top]
                            r1_score = r1_scores[:, r1_top]
                            r1_input = get_one_hot_input(
                                r1_pred, len(condition2label['reagent1']))

                            r2_preds = model.r2_func(
                                fp_emb, c1_input, s1_input, s2_input, r1_input)
                            r2_scores, r2_cdts = r2_preds.squeeze().topk(
                                topk_rank_thres['r2'])
                            for r2_top in range(r2_cdts.size(-1)):
                                # if device != torch.device('cpu'):
                                #     pred_list = r2_cdts[:, r2_top].cpu(
                                #     ).numpy().tolist()
                                # else:
                                #     pred_list = r2_cdts[:,
                                #                         r2_top].numpy().tolist()
                                r2_pred = r2_cdts[:, r2_top]
                                r2_score = r2_scores[:, r2_top]
                                # one_batch_prediction_df['c1_top-{}_s1_top-{}_s2_top-{}_r1_top-{}_r2_top-{}'.format(c1_top+1, s1_top+1, s2_top+1, r1_top+1, r2_top+1)
                                #                         ] = pred_list
                                # r2_pred_list = pred_list

                                one_pred = torch.cat([c1_pred.unsqueeze(1), s1_pred.unsqueeze(
                                    1), s2_pred.unsqueeze(1), r1_pred.unsqueeze(1), r2_pred.unsqueeze(1)], dim=-1)
                                one_score = c1_score * s1_score * s2_score * r1_score * r2_score
                                one_batch_preds.append(one_pred)
                                one_batch_scores.append(one_score)

                                if use_temperature:
                                    r2_input = get_one_hot_input(
                                        r2_pred, len(condition2label['reagent2']))
                                    t_preds = model.T_func(
                                        fp_emb, c1_input, s1_input, s2_input, r1_input, r2_input)
                                    t_preds = t_preds.squeeze()
                                    one_batch_t_preds.append(t_preds)

            one_batch_preds = torch.cat(
                [x.unsqueeze(0) for x in one_batch_preds], dim=0)
            one_batch_scores = torch.cat(
                [x.unsqueeze(0) for x in one_batch_scores], dim=0)
            if use_temperature:
                one_batch_t_preds = torch.cat(
                    [x.unsqueeze(0) for x in one_batch_t_preds], dim=0
                )

            sorted_one_batch_preds = []
            sorted_one_batch_t_preds = []
            batch_number = pfp.size(0)
            for n in range(batch_number):
                sorted_one_batch_preds.append(
                    one_batch_preds[one_batch_scores[:, n].argsort(
                        descending=True), n, :]
                )
                if use_temperature:
                    sorted_one_batch_t_preds.append(
                        one_batch_t_preds[one_batch_scores[:,
                                                           n].argsort(descending=True), n]
                    )
            sorted_one_batch_preds = torch.cat(
                [x.unsqueeze(0) for x in sorted_one_batch_preds], dim=0)
            if use_temperature:
                sorted_one_batch_t_preds = torch.cat(
                    [x.unsqueeze(0) for x in sorted_one_batch_t_preds], dim=0)
                closest_one_batch_t_preds = sorted_one_batch_t_preds[:, 0]
                closest_erro += torch.abs(closest_one_batch_t_preds - t).sum()

            one_batch_topk_acc_mat = np.zeros(topk_acc_mat.shape)
            # topk_get = [1, 3, 5, 10, 15]
            for idx in range(sorted_one_batch_preds.size(0)):
                topk_hit_df = get_accuracy_for_one(
                    sorted_one_batch_preds[idx], one_batch_ground_truth[idx], topk_get=topk_get, condition_to_calculate=condition_to_calculate)
                one_batch_topk_acc_mat += topk_hit_df.values
            topk_acc_mat += one_batch_topk_acc_mat
    topk_acc_mat /= len(data_loader.dataset)
    topk_acc_df = pd.DataFrame(topk_acc_mat)

    topk_acc_df.columns = [f'top-{k} accuracy' for k in topk_get]
    # topk_acc_df.index = ['c1', 's1', 's2', 'r1', 'r2', 'overall']
    topk_acc_df.index = condition_to_calculate + ['overall']
    if use_temperature:
        closest_temp_mae = (closest_erro / len(data_loader.dataset)).item()
        topk_acc_df.loc['closest_pred_temp_mae'] = [
            closest_temp_mae] * len(topk_acc_df.columns)
    topk_acc_df = topk_acc_df.round(4)
    print(topk_acc_df)
    if top_fname:
        
        if save_topk_path:
            topk_acc_df.to_csv(os.path.join(save_topk_path, top_fname))
    else:
        if save_topk_path:
            topk_acc_df.to_csv(os.path.join(save_topk_path, 'test_accuacy.csv'))
    return topk_acc_df


def save_model(model_dir, config, state_dict):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(state_dict, os.path.join(model_dir, 'model.pth'))
    with open(os.path.join(model_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f)


def load_model(model=None, model_dir=''):
    if os.path.isdir(model_dir):
        state = torch.load(os.path.join(model_dir, 'model.pth'), map_location=torch.device('cpu'))
    else:
        state = torch.load(model_dir, map_location=torch.device('cpu'))["model"]

    if model:
        model.load_state_dict(state)
        return model
    else:
        return state
    

def load_pretrain_model_state(model, pretrained_state):
    model_state = model.state_dict()
    pretrained_state_filter = {}
    extra_layers = []
    different_shape_layers = []
    need_train_layers = []
    for name, parameter in pretrained_state.items():
        if name in model_state and parameter.size() == model_state[name].size():
            pretrained_state_filter[name] = parameter
        elif name not in model_state:
            extra_layers.append(name)
        elif parameter.size() != model_state[name].size():
            different_shape_layers.append(name)
    for name, parameter in model_state.items():
        if name not in pretrained_state_filter:
            need_train_layers.append(name)

    model_state.update(pretrained_state_filter)
    model.load_state_dict(model_state)
    
    print('Extra layers:', extra_layers)
    print('Different shape layers:', different_shape_layers)
    print('Need to train layers:', need_train_layers)
    return model


if __name__ == '__main__':
    debug = False
    train_model = True
    config_path = './config/condition/peft.yaml'

    print('Debug: {}, Training: {}'.format(debug, train_model))

    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    args = args_parser()
    args.checkpoint = "./checkpoint/supcon_hierar/model_pretrain_best_mAP.pt"
    args.vocab_file = "./data/vocab_share.pk"
    device = f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu"
    args.device = device

    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if debug:
        config['database_path'] = './data/uspto_condition/USPTO_condition_final_debug'
        config['model_name'] = 'debug_model_output'

    time_str = time.strftime('%Y-%m-%d_%Hh-%Mm-%Ss',
                             time.localtime(time.time()))
    # writer = SummaryWriter(f'runs/{time_str}')

    #! load vocab
    src_itos, src_stoi, tgt_itos, tgt_stoi = load_vocab(args.vocab_file) 
    final_condition_data_path = config['database_path']
    fps_dict, label_dict, rxn_dict = load_dataset(
        final_condition_data_path, config['database_name'], src_stoi, tgt_stoi, use_temperature=config['use_temperature'])
    src_pad_idx = src_stoi['<pad>']
    tgt_pad_idx = tgt_stoi['<pad>']
    print("Successfully load vocab...")

    train_dataset = ConditionDataset(
        fps_dict=fps_dict, label_dict=label_dict, dataset='train', use_temperature=config['use_temperature'])
    val_dataset = ConditionDataset(
        fps_dict=fps_dict, label_dict=label_dict, dataset='val', use_temperature=config['use_temperature'])
    test_dataset = ConditionDataset(
        fps_dict=fps_dict, label_dict=label_dict, dataset='test', use_temperature=config['use_temperature'])

    condition2label = train_dataset.condition2label
    label2condition = train_dataset.label2condition
    fp_dim = train_dataset.fp_dim  # concat (or concat+product, concat+substract)
    c1_dim = train_dataset.c1_dim  # according to number of reaction conditions
    s1_dim = train_dataset.s1_dim
    s2_dim = train_dataset.s2_dim
    r1_dim = train_dataset.r1_dim
    r2_dim = train_dataset.r2_dim

    config['fp_dim'] = fp_dim
    config['c1_dim'] = c1_dim
    config['s1_dim'] = s1_dim
    config['s2_dim'] = s2_dim
    config['r1_dim'] = r1_dim
    config['r2_dim'] = r2_dim

    #! using collate_fn for padding
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False)  

    model = RXNConditionModel(
        fp_dim=fp_dim,
        h_dim=config['h_dim'],
        dropout_rate=config['dropout_rate'],
        c1_dim=c1_dim,
        s1_dim=s1_dim,
        s2_dim=s2_dim,
        r1_dim=r1_dim,
        r2_dim=r2_dim,
        # is_train=True,
    )

    if 'train_from_checkpoints' in config:
        pretrained_state = load_model(model_dir=config['train_from_checkpoints'])
        model = load_pretrain_model_state(model, pretrained_state)
    
    model = model.to(device)

    #! load pre-trained transformer
    transformer = build_model(args, src_itos, tgt_itos)
    pretrained_state = load_model(model_dir=args.checkpoint)
    transformer = load_pretrain_model_state(transformer, pretrained_state)
    transformer = transformer.to(device)
    print("Successfully load pre-trained transformer as rxn_encoder")

    loss_fn_list = [
        torch.nn.CrossEntropyLoss(),
        torch.nn.MSELoss()
    ]
    # loss_fn = torch.nn.CrossEntropyLoss()
    print('############################# RCR Training config #############################')
    print(yaml.dump(config))
    print('###############################################################################')
    print(model)

    #! model_params_group
    model_param_group = [{"params": model.parameters(), "lr": config['lr']},
                         {"params": transformer.parameters(), "lr": config['lr']*0.1}]

    optimizer = torch.optim.Adam(model_param_group, lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.7,
        patience=2,
        min_lr=0.000001)
    it = trange(config['epochs'])
    best_loss = np.inf

    patience = 10
    early_stop_count = 0

    print(config["train_transformer"])

    if train_model:

        for epoch in it:
            loss = train_one_epoch(model, train_loader, loss_fn_list, optimizer,
                                   device=device, condition2label=condition2label, it=it, use_temperature=config['use_temperature'],
                                   rxn_encoder=transformer, finetune_rxn_encoder=config["train_transformer"])

            print('lr:', scheduler.optimizer.param_groups[0]['lr'])

            val_loss = validation(model, val_loader, loss_fn_list,
                                  device, condition2label=condition2label, use_temperature=config['use_temperature'],
                                  rxn_encoder=transformer)
            print("Train loss:", loss, 'Validation loss:', val_loss)
            scheduler.step(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                print(best_loss)
                save_model(os.path.join(
                    config['model_path'], config['model_name']), config, model.state_dict())
                save_model(os.path.join(
                    config['model_path'], config['model_name'], "transformer"), config, transformer.state_dict())
                early_stop_count = 0
            else:
                early_stop_count += 1
            
            if early_stop_count >= patience:
                break

    print('Testing model...')
    model = load_model(model, os.path.join(
        config['model_path'], config['model_name']))
    model = model.to(device)
    transformer = load_model(transformer, os.path.join(
        config['model_path'], config['model_name'], "transformer"))
    transformer = transformer.to(device)
    caculate_accuracy(model,
                      test_loader,
                      device=device,
                      condition2label=condition2label,
                      rxn_encoder=transformer,
                      topk_rank_thres={
                        'c1': 1,
                        's1': 3,
                        's2': 1,
                        'r1': 5,
                        'r2': 1,
                      },
                      save_topk_path=os.path.join(config['model_path'], config['model_name']), use_temperature=config['use_temperature'],
                    #   condition_to_calculate=['s1', 'r1']
                     )

    # writer.close()
