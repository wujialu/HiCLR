"Ref: source code of MolCLR https://github.com/yuyangw/MolCLR/blob/master/finetune.py"

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import shutil
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

from utils.dataset_test import MolTestDatasetWrapper
from utils.build_utils import build_model
from models.model import FinetunePredictor
from args_parse import args_parser

def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config/config_finetune.yaml', os.path.join(model_checkpoints_folder, 'config_finetune.yaml'))


def seed_everything(seed=0):
    # To fix the random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class FineTune(object):
    def __init__(self, dataset, config, args):
        self.config = config
        self.device = self._get_device()
        self.args = args

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = current_time + '_' + config['task_name']
        log_dir = os.path.join('./result/moleculenet_finetune', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.dataset = dataset
        if config['dataset']['task'] == 'classification':
            # self.criterion = nn.CrossEntropyLoss()
            self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        elif config['dataset']['task'] == 'regression':
            if self.config["task_name"] in ['qm7', 'qm8', 'qm9']:
                self.criterion = nn.L1Loss(reduction="none")
            else:
                self.criterion = nn.MSELoss(reduction="none")
        #TODO: task_pos_weight for classification problem?

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step(self, model, x, y, mask, n_iter):
        # get the prediction
        pred = model(x)  # [N,C]

        if self.config['dataset']['task'] == 'classification':
            loss = self.criterion(pred, y.reshape(pred.shape))

        elif self.config['dataset']['task'] == 'regression':
            if self.normalizer: #! whether use normalizer in default
                loss = self.criterion(pred, self.normalizer.norm(y))
            else:
                loss = self.criterion(pred, y)
        
        loss = (loss * mask).sum() / mask.sum()

        return loss

    def train(self):
        train_loader, valid_loader, test_loader, dataset = self.dataset.get_data_loaders()

        self.normalizer = None
        if self.config["task_name"] in ['qm7', 'qm9']: #! bug
            labels = []
            for x, y in train_loader:
                labels.append(y)
            labels = torch.cat(labels)
            self.normalizer = Normalizer(labels)
            print(self.normalizer.mean, self.normalizer.std, labels.shape)

        # if self.config['model_type'] == 'gin':
        #     from models.ginet_finetune import GINet
        #     model = GINet(self.config['dataset']['task'], **self.config["model"]).to(self.device)
        #     model = self._load_pre_trained_weights(model)
        # elif self.config['model_type'] == 'gcn':
        #     from models.gcn_finetune import GCN
        #     model = GCN(self.config['dataset']['task'], **self.config["model"]).to(self.device)
        #     model = self._load_pre_trained_weights(model)
        
        if config["finetune"] == "adapter":
            self.args.attn_mode = "adapter"
            self.args.ffn_mode = "adapter"
            self.args.ffn_option = "sequential"
            self.args.dropout = config["model"]["drop_ratio"]

        transformer = build_model(self.args, dataset.src_itos, dataset.tgt_itos)
        transformer = self._load_pre_trained_weights(transformer)
        model = FinetunePredictor(transformer, config["dataset"]["task"], config["dataset"]["target"], dropout=config["model"]["drop_ratio"],
                                  pool=config["model"]["pool"],
                                  decoder_start_token_id=dataset.tgt_stoi["<sos>"] )
        
        if config["finetune"] == "adapter":
            freeze_params(model.encoder, except_para_l=("adapter", "prefix", "lora", "norm"))
            freeze_params(model.decoder, except_para_l=("adapter", "prefix", "lora", "norm"))
        elif config["finetune"] == "dec" and config['model']['pool'] == "dec":
            freeze_params(model.encoder)
        elif config["finetune"] == "last_layer":
            freeze_params(model.encoder, except_para_l=("adapter", "prefix", "lora", "norm", "transformer.3"))
        model.to(self.device)

        layer_list = []
        for name, param in model.named_parameters():
            if 'pred_head' in name:
                print(name, param.requires_grad)
                layer_list.append(name)

        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))
        optimizer = torch.optim.Adam(
            [{'params': base_params, 'lr': self.config['init_base_lr']}, {'params': params}],
            self.config['init_lr'], weight_decay=eval(self.config['weight_decay'])
        )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_rgr = np.inf
        best_valid_cls = 0
        patience = 10
        early_stop = 0

        for epoch_counter in range(self.config['epochs']):
            for bn, data in enumerate(train_loader):
                optimizer.zero_grad()

                # data = data.to(self.device)
                x, y, mask = data
                x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
                loss = self._step(model, x, y, mask, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print(epoch_counter, bn, loss.item())
                
                loss.backward()
                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                if self.config['dataset']['task'] == 'classification': 
                    valid_loss, valid_cls = self._validate(model, valid_loader)
                    if valid_cls > best_valid_cls:
                        # save the model weights
                        best_valid_cls = valid_cls
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                        early_stop = 0
                    else:
                        early_stop += 1

                elif self.config['dataset']['task'] == 'regression': 
                    valid_loss, valid_rgr = self._validate(model, valid_loader)
                    if valid_rgr < best_valid_rgr:
                        # save the model weights
                        best_valid_rgr = valid_rgr
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                        early_stop = 0
                    else:
                        early_stop += 1

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
            
            if early_stop >= patience:
                break
        
        self._test(model, test_loader)

    def _load_pre_trained_weights(self, model):
        try:
            ckpt_path = os.path.join("./checkpoint", config["fine_tune_from"], "model_pretrain_best_mAP.pt")
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            state_dict = checkpoint['model']
            model.load_state_dict(state_dict, strict=False)  #? model.load_my_state_dict
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        predictions = []
        labels = []
        masks = []
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):
                x, y, mask = data
                x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)

                pred = model(x)
                loss = self._step(model, x, y, mask, bn)

                valid_loss += loss.item() * y.size(0)
                num_data += y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    # pred = F.softmax(pred, dim=-1)
                    pred = F.sigmoid(pred)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(y.reshape(pred.shape).numpy())
                    masks.extend(mask.numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(y.reshape(pred.shape).cpu().numpy())
                    masks.extend(mask.cpu().numpy())

            valid_loss /= num_data
        
        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                mae = mean_absolute_error(labels, predictions)
                print('Validation loss:', valid_loss, 'MAE:', mae)
                return valid_loss, mae
            else:
                rmse = mean_squared_error(labels, predictions, squared=False)
                print('Validation loss:', valid_loss, 'RMSE:', rmse)
                return valid_loss, rmse

        elif self.config['dataset']['task'] == 'classification': 
            predictions = np.array(predictions)
            labels = np.array(labels)
            masks = np.array(masks)
            scores = []
            for i in range(labels.shape[1]):
                task_w = masks[:, i]
                task_y_true = labels[:, i][task_w != 0]
                task_y_pred = predictions[:, i][task_w != 0]
                scores.append(roc_auc_score(task_y_true, task_y_pred))
            roc_auc = np.mean(scores)
            print('Validation loss:', valid_loss, 'ROC AUC:', roc_auc)
            return valid_loss, roc_auc

    def _test(self, model, test_loader):
        model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        # test steps
        predictions = []
        labels = []
        masks = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                x, y, mask = data
                x, y, mask = x.squeeze(0).to(self.device), y.to(self.device), mask.to(self.device)

                pred = model(x)
                loss = self._step(model, x, y.repeat(x.shape[0], 1, 1), mask.repeat(x.shape[0], 1), bn)

                test_loss += loss.item() * x.size(0)
                num_data += x.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    # pred = F.softmax(pred, dim=-1)
                    pred = F.sigmoid(pred)
                
                pred = pred.mean(dim=0, keepdim=True)  # [batch_size, num_tasks]-->[1, num_tasks]

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(y.reshape(pred.shape).numpy())
                    masks.extend(mask.numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(y.reshape(pred.shape).cpu().numpy())
                    masks.extend(mask.cpu().numpy())

            test_loss /= num_data
        
        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                self.mae = mean_absolute_error(labels, predictions)
                print('Test loss:', test_loss, 'Test MAE:', self.mae)
            else:
                self.rmse = mean_squared_error(labels, predictions, squared=False)
                print('Test loss:', test_loss, 'Test RMSE:', self.rmse)

        elif self.config['dataset']['task'] == 'classification': 
            predictions = np.array(predictions)
            labels = np.array(labels)
            masks = np.array(masks)
            scores = []
            for i in range(labels.shape[1]):
                task_w = masks[:, i]
                task_y_true = labels[:, i][task_w != 0]
                task_y_pred = predictions[:, i][task_w != 0]
                scores.append(roc_auc_score(task_y_true, task_y_pred))
            self.roc_auc = scores
            print('Test loss:', test_loss, 'Test ROC AUC:', self.roc_auc)
            
            
def main(args, config):
    dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'])
    fine_tune = FineTune(dataset, config, args)
    fine_tune.train()
    
    if config['dataset']['task'] == 'classification':
        return fine_tune.roc_auc
    if config['dataset']['task'] == 'regression':
        if config['task_name'] in ['qm7', 'qm8', 'qm9']:
            return fine_tune.mae
        else:
            return fine_tune.rmse


if __name__ == "__main__":
    args = args_parser()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    config['dataset']['num_smiles_aug'] = 20

    if config['task_name'] == 'BBBP':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/molformer/bbbp/'
        target_list = ["p_np"]

    elif config['task_name'] == 'Tox21':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/molformer/tox21'
        target_list = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", 
            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]

    elif config['task_name'] == 'ClinTox':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/molformer/clintox'
        target_list = ['CT_TOX', 'FDA_APPROVED']

    elif config['task_name'] == 'HIV':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/molformer/hiv/'
        target_list = ["HIV_active"]
        config['dataset']['num_smiles_aug'] = 5

    elif config['task_name'] == 'BACE':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/molformer/bace/'
        target_list = ["Class"]

    elif config['task_name'] == 'SIDER':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/molformer/sider'
        target_list = [
            "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues", 
            "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders", 
            "Gastrointestinal disorders", "Social circumstances", "Immune system disorders", 
            "Reproductive system and breast disorders", 
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)", 
            "General disorders and administration site conditions", "Endocrine disorders", 
            "Surgical and medical procedures", "Vascular disorders", 
            "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders", 
            "Congenital, familial and genetic disorders", "Infections and infestations", 
            "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders", 
            "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions", 
            "Ear and labyrinth disorders", "Cardiac disorders", 
            "Nervous system disorders", "Injury, poisoning and procedural complications"
        ]
    
    elif config['task_name'] == 'MUV':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/moleculenet/muv.csv'
        target_list = [
            'MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644', 'MUV-548', 'MUV-852',
            'MUV-600', 'MUV-810', 'MUV-712', 'MUV-737', 'MUV-858', 'MUV-713', 'MUV-733',
            'MUV-652', 'MUV-466', 'MUV-832'
        ]

    elif config['task_name'] == 'FreeSolv':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/moleculenet/freesolv.csv'
        target_list = ["expt"]
    
    elif config["task_name"] == 'ESOL':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/moleculenet/esol.csv'
        target_list = ["measured log solubility in mols per litre"]

    elif config["task_name"] == 'Lipo':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/moleculenet/Lipophilicity.csv'
        target_list = ["exp"]
    
    elif config["task_name"] == 'qm7':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/moleculenet/qm7.csv'
        target_list = ["u0_atom"]

    elif config["task_name"] == 'qm8':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/moleculenet/qm8.csv'
        target_list = [
            "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", 
            "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM", "f2-CAM"
        ]
    
    elif config["task_name"] == 'qm9':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/moleculenet/qm9.csv'
        target_list = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv']

    else:
        raise ValueError('Undefined downstream task!')

    print(config)

    # set random seed
    seed_everything(config['seed'])
    
    results_list = []

    # single task training
    # for target in target_list:
    #     config['dataset']['target'] = target
    #     result = main(config)
    #     results_list.append([target, result])

    # multi-task training
    config['dataset']['target'] = target_list
    result = main(args, config)  # averaged auroc score
    for i, target in enumerate(target_list):
        results_list.append([target, result[i]])

    os.makedirs('experiments', exist_ok=True)
    df = pd.DataFrame(results_list, columns=["target", "test_score"])
    df = df.append({"target": "mean", "test_score": df.test_score.mean()}, ignore_index=True)
    result_dir = 'experiments_molformer_aug/{}/'.format(config['finetune'])
    os.makedirs(result_dir, exist_ok=True)
    df.to_csv(
        'experiments_molformer_aug/{}/{}_{}_finetune_lr{}_baselr{}_drop{}.csv'.format(config['finetune'], config['model']['pool'], config['task_name'], config['init_lr'], config['init_base_lr'], config['model']['drop_ratio']), 
         mode='a', index=False
    )