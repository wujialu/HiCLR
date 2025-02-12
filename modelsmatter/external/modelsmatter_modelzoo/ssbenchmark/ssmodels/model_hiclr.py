import os
import sys
import argparse
import pickle
from tqdm import tqdm
from rdkit import Chem
import numpy as np
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from ssbenchmark.ssmodels.base_model import Registries, SSMethod
from ssbenchmark.utils import canonicalize_smiles, split_reactions
from external_models.HiCLR.utils.dataset_test import ReactionDataset
from external_models.HiCLR.utils.translate_utils import translate_batch
from external_models.HiCLR.utils.build_utils import build_model, build_retro_iterator, load_checkpoint_downstream


@Registries.ModelChoice.register(key="hiclr")
class model_hiclr(SSMethod):
    def __init__(self, module_path=None):
        self.model_name = "HiCLR"
        if module_path is not None:
            sys.path.insert(len(sys.path), os.path.abspath(module_path))

    def preprocess(self, data, reaction_col):
        pass

    def process_input(self, data, reaction_col):
        pass

    def preprocess_store(self):
        pass

    def process_output(self):
        pass

    def model_setup(self, use_gpu=False, **kwargs):
        DEFAULT_BATCH_SIZE = 64
        DEFAULT_NUM_BEAMS = 50
        DEFAULT_VOCAB_PATH = "./external/modelsmatter_modelzoo/external_models/HiCLR/vocab_share.pk"
        parser = argparse.ArgumentParser()
        parser.add_argument("--reactants_path", type=str)
        parser.add_argument("--model_path", type=str)
        parser.add_argument("--products_path", type=str)
        parser.add_argument("--vocab_path", type=str)
        parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
        parser.add_argument("--num_beams", type=int, default=DEFAULT_NUM_BEAMS)
        parser.add_argument("--task", type=str)
        parser.add_argument('--num_workers', type=int, default=1, help='number of workers for dataset loading')
        parser.add_argument('--encoder_num_layers', type=int, default=4, help='number of layers of transformer')
        parser.add_argument('--decoder_num_layers', type=int, default=4, help='number of layers of transformer')
        parser.add_argument('--condition_attn_layers', type=int, default=0, help='number of layers of transformer decoder with cond_attn')
        parser.add_argument('--d_model', type=int, default=256, help='dimension of model representation')
        parser.add_argument('--heads', type=int, default=8, help='number of heads of multi-head attention')
        parser.add_argument('--d_ff', type=int, default=2048, help='')
        parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
        parser.add_argument('--max_length', type=int, default=200)

        parser.add_argument('--known_class', default=0)
        parser.add_argument('--shared_vocab', default=1)
        parser.add_argument('--shared_encoder', action="store_true", default=False)
        parser.add_argument('--ffn_mode', type=str, default='none', choices=["none", "adapter"])
        parser.add_argument('--ffn_option', type=str, default="none", choices=["parallel", "sequential", "none"])
        parser.add_argument('--ffn_bn', type=int, default=256)
        parser.add_argument('--ffn_adapter_scalar', type=str, default="1") 
        parser.add_argument('--ffn_adapter_init_option', type=str, default="lora", choices=["bert", "lora"])
        parser.add_argument('--ffn_adapter_layernorm_option', type=str, default="none", choices=["in", "out", "none"]) 
        parser.add_argument('--attn_mode', type=str, default='none', choices=["none", "prefix", "lora", "adapter"])
        parser.add_argument('--attn_bn', type=int, default=10)
        parser.add_argument('--attn_dim', type=int, default=32)
        parser.add_argument('--prompt', action="store_true", default=False)
        parser.add_argument('--input_prompt_attn', action="store_true", default=False)
        parser.add_argument('--proto_hierarchy', type=int, default=3, help='the number of hierarchy')
        parser.add_argument('--proto_path', type=str, default='./result/ecreact')
        parser.add_argument('--proto_version', type=str, default="top", choices=["bottom", "middle", "top", "hierarchy", "namerxn"])
        parser.add_argument('--freeze_proto', action="store_true", default=False)

        args = parser.parse_args(
            [
                f"--reactants_path={kwargs.get('reactants_path', None)}",
                f"--model_path={kwargs.get('model_path', None)}",
                f"--products_path={kwargs.get('products_path', None)}",
                f"--vocab_path={kwargs.get('vocab_path', DEFAULT_VOCAB_PATH)}",
                f"--batch_size={kwargs.get('batch_size', DEFAULT_BATCH_SIZE)}",
                f"--num_beams={kwargs.get('num_beams', DEFAULT_NUM_BEAMS)}",
                f"--task={kwargs.get('task', 'backward_prediction')}",
            ]
        )
        self.args = args
        self.device = f"cuda:0" if use_gpu else "cpu"
        self.args.device = self.device

        with open(args.vocab_path, 'rb') as f:
            self.src_itos, self.tgt_itos = pickle.load(f)
        self.src_stoi = {self.src_itos[i]: i for i in range(len(self.src_itos))}
        self.tgt_stoi = {self.tgt_itos[i]: i for i in range(len(self.tgt_itos))}
        model = build_model(args, self.src_itos, self.tgt_itos)
        self.model = load_checkpoint_downstream(args.model_path, model)
        print(f"Finished loading model")

    def translate(self, iterator, model, dataset):
        model.to(self.device)
        model.eval()

        generations = []
        invalid_token_indices = [dataset.tgt_stoi['<RX_{}>'.format(i)] for i in range(1, 11)]
        invalid_token_indices += [dataset.tgt_stoi['<UNK>'], dataset.tgt_stoi['<unk>'], dataset.tgt_stoi['<mask>']] 
        invalid_token_indices += [dataset.tgt_stoi['<unused{}>'.format(i)] for i in range(1, 11)]

        for batch in tqdm(iterator, total=len(iterator)):
            pred_tokens, pred_scores = translate_batch(model, batch, device=self.device, beam_size=self.args.num_beams,
                                                       invalid_token_indices=invalid_token_indices,
                                                       max_length=self.args.max_length, 
                                                       dataset=dataset)
            
            for idx in range(batch[0].shape[1]):
                #! generate invalid smiles
                hypos = [dataset.reconstruct_smi(tokens, src=False) for tokens in pred_tokens[idx]]
                hypo_len = [len(hypo) for hypo in hypos]
                hypos = np.array(["".join(hypo) for hypo in hypos])
                new_pred_scores = copy.deepcopy(pred_scores[idx]).cpu().numpy() / hypo_len
                ordering = np.argsort(new_pred_scores)[::-1] # sorted from high to low
                generations.append(hypos[ordering])
        return generations, new_pred_scores

    def _model_call(self, X):
        print("Reading dataset...")
        dataset = ReactionDataset(X, X, vocab_file=self.args.vocab_path, task=self.args.task)
        print("Finished dataset.")

        test_loader = DataLoader(dataset, batch_size=self.args.batch_size, drop_last=False,
                                 collate_fn=dataset._collate_fn)
        print("Finished loader.")

        print("Evaluating model...")
        # the shpae of log_lhs: [batch_size, num_beams]
        smiles, log_lhs = self.translate(test_loader, self.model, dataset)
        output = F.softmax(
            torch.Tensor(log_lhs).view(-1, self.args.num_beams), dim=1
        ) 

        return smiles, output.tolist()
    
    def model_call(self, X):
        return self._model_call(X)