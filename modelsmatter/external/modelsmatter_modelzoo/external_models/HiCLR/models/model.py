""" Onmt NMT Model base class definition """
import copy
import torch.nn as nn
import torch
import torch.nn.functional as F

from models.embedding import Embedding, TokenEmbedding, PromptEmbedding
from models.module import MultiHeadedAttention, LayerNorm
from models.encoder import TransformerEncoder
from models.decoder import TransformerDecoder
from models.compacter.seq2seq.adapters.adapter_modeling import HyperComplexAdapter
from models.film import FiLMGen


class MolecularTransformer(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, args, encoder_num_layers, decoder_num_layers, d_model, heads, d_ff, dropout,
                 vocab_size_src, vocab_size_tgt, shared_vocab, shared_encoder=False, src_pad_idx=1,
                 tgt_pad_idx=1, multigpu=False, vocab_size_prompt=10):
        self.multigpu = multigpu
        super(MolecularTransformer, self).__init__()

        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.d_model = d_model
        self.heads = heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.shared_vocab = shared_vocab
        self.shared_encoder = shared_encoder
        if shared_vocab: #! vocab_size = 364
            assert vocab_size_src == vocab_size_tgt and src_pad_idx == tgt_pad_idx
            self.embedding_src = self.embedding_tgt = Embedding(vocab_size=vocab_size_src + 1, embed_size=d_model,
                                                                padding_idx=src_pad_idx)
        else:
            self.embedding_src = Embedding(vocab_size=vocab_size_src + 1, embed_size=d_model, padding_idx=src_pad_idx)
            self.embedding_tgt = Embedding(vocab_size=vocab_size_tgt + 1, embed_size=d_model, padding_idx=tgt_pad_idx)
        
        if args.prompt:
            self.embedding_prompt = PromptEmbedding(vocab_size=vocab_size_prompt, embed_size=d_model)  # pretrained_emb
        else:
            self.embedding_prompt = None
        
        multihead_attn_modules_en = nn.ModuleList(
            [MultiHeadedAttention(args, heads, d_model, dropout=dropout) # use_adapter=False
             for _ in range(encoder_num_layers)])
        if shared_encoder:
            assert encoder_num_layers == decoder_num_layers
            multihead_attn_modules_de = multihead_attn_modules_en
        else:
            multihead_attn_modules_de = nn.ModuleList(
                [MultiHeadedAttention(args, heads, d_model, dropout=dropout)
                 for _ in range(decoder_num_layers)])

        self.encoder = TransformerEncoder(args, num_layers=encoder_num_layers,
                                          d_model=d_model, heads=heads,
                                          d_ff=d_ff, dropout=dropout,
                                          embeddings=self.embedding_src,
                                          attn_modules=multihead_attn_modules_en,
                                          prompt_embeddings=self.embedding_prompt)

        self.decoder = TransformerDecoder(args, num_layers=decoder_num_layers,
                                          d_model=d_model, heads=heads,
                                          d_ff=d_ff, dropout=dropout,
                                          embeddings=self.embedding_tgt,
                                          self_attn_modules=multihead_attn_modules_de)

        self.generator = nn.Sequential(nn.Linear(d_model, vocab_size_tgt),
                                       nn.LogSoftmax(dim=-1))
        
        self.lin_cond = nn.Linear(512, d_model)
        self.cond_layer_norm = LayerNorm(d_model)
        
    def forward(self, src, tgt, dec_state=None, src_template_mask=None, tgt_template_mask=None, template_pooling=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt = tgt[:-1]  # exclude last target from inputs
        
        if template_pooling:
            src_emb, encoder_outputs, src_reps, src_template_reps = self.encoder(src, src_template_mask=src_template_mask, template_pooling=template_pooling)
            enc_state = \
                self.decoder.init_decoder_state(src, encoder_outputs)
            del src, src_template_mask, src_emb
            decoder_outputs, dec_state, attns, tgt_reps, tgt_template_reps = self.decoder(tgt, encoder_outputs,
                                                                                          enc_state if dec_state is None
                                                                                          else dec_state,
                                                                                          tgt_template_mask=tgt_template_mask, template_pooling=template_pooling)

        else:
            src_emb, encoder_outputs, src_reps = self.encoder(src, src_template_mask=src_template_mask)
            enc_state = \
                self.decoder.init_decoder_state(src, encoder_outputs)
            decoder_outputs, dec_state, attns, tgt_reps = self.decoder(tgt, encoder_outputs,
                                                                       enc_state if dec_state is None
                                                                       else dec_state,
                                                                       tgt_template_mask=tgt_template_mask)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        
        generative_scores = self.generator(decoder_outputs) 
        if template_pooling:
            return generative_scores, attns, src_reps, tgt_reps, src_template_reps, tgt_template_reps
        else:
            return generative_scores, attns, src_reps, tgt_reps
        
    def forward_with_given_prompt(self, src, tgt, prompt, dec_state=None):
        tgt = tgt[:-1]
        src_emb, encoder_outputs = self.encoder.forward_with_given_prompt(src, prompt)
        enc_state = \
            self.decoder.init_decoder_state(src, encoder_outputs)
        decoder_outputs, dec_state, attns = self.decoder.forward_with_prompt(tgt, encoder_outputs,
                                                                             enc_state if dec_state is None
                                                                             else dec_state)
        generative_scores = self.generator(decoder_outputs) 
        return generative_scores, attns
        
    def forward_with_pred_prompt(self, src, tgt):
        src_emb, encoder_outputs, src_reps, prompt_attn = self.encoder.forward_with_pred_prompt(src)
        enc_state = \
            self.decoder.init_decoder_state(src, encoder_outputs)
        decoder_outputs, dec_state, attns, tgt_reps = self.decoder.forward_with_prompt(tgt, encoder_outputs,
                                                                                       enc_state if dec_state is None
                                                                                       else dec_state,
                                                                                       prompt=True)
        generative_scores = self.generator(decoder_outputs) 
        return generative_scores, attns, src_reps, tgt_reps, prompt_attn
    
    def extract_reaction_fp(self, src, tgt, cond=None, dec_state=None, return_cross_attn=False, return_token_featuers=False):
        tgt = tgt[:-1]
        if cond is not None:
            cond_emb = self.cond_layer_norm(self.lin_cond(cond)) # 3*256
        else:
            cond_emb = None
        # cond_emb = torch.sum(self.lin_cond(cond), 0, keepdim=True) # shape: (1, batch_size, feature_dim)
        src_emb, encoder_outputs, src_reps = self.encoder(src, cond_emb=cond_emb)
        enc_state = \
            self.decoder.init_decoder_state(src, encoder_outputs)
        decoder_outputs, dec_state, attns, tgt_reps = self.decoder(tgt, encoder_outputs,
                                                                    enc_state if dec_state is None
                                                                    else dec_state,
                                                                    cond_emb=cond_emb)
        if return_cross_attn:
            return src_reps, tgt_reps, attns
        elif return_token_featuers:
            return encoder_outputs, decoder_outputs
        else:   
            return src_reps, tgt_reps
    
    def extract_mol_fp(self, src):
        src_emb, encoder_outputs, src_reps = self.encoder(src)
        return encoder_outputs, src_reps

    def extract_src_token_self_attention(self, src, tgt, dec_state=None, reaction_reps=None):
        tgt = tgt[:-1]
        if reaction_reps.shape[1] == self.d_model * 2:
            reaction_reps = self.linear(reaction_reps) # (batch_size, d_model)
        src_emb, encoder_outputs, src_reps, mask = self.encoder(src, return_mask=True) # src_token_reps = encoder_outputs

        # add additional token
        inputs = torch.cat([reaction_reps.unsqueeze(0), encoder_outputs], dim=0).transpose(0, 1) # (max_len+1, batch_size, d_model)-->[batch_size x src_len x model_dim]
        w_batch, w_len = mask.shape[0], mask.shape[1]
        mask = torch.cat([torch.zeros(w_batch,1).bool().to(mask.device), mask[:, 0, :]], dim=1).unsqueeze(1).expand(w_batch, w_len+1, w_len+1)

        # substitute <UNK> token
        # encoder_outputs[0, :, :] = reaction_reps
        # inputs = encoder_outputs

        # self-attention 
        out = self.encoder_layer(inputs=inputs, mask=mask) # [batch_size x src_len x model_dim]
        out = self.layer_norm(out)
        src_reps = torch.einsum("abc,ab->ac", out, (mask[:, 0, :] == 0).float())
        return out.transpose(0, 1).contiguous()[0, :, :], src_reps  # use updated retro_fp or the whole src_reps


class ProjectNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, output_dim=None):
        super(ProjectNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, x):
        x_proj = self.proj(x)

        return x_proj


class MLP(nn.Module):
    def __init__(self, size_layer, activation='relu', output_activation=None, initial_method=None, dropout=0.0):
        super(MLP, self).__init__()
        self.hiddens = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation

        # self.linear_transform = nn.Linear(512, 512)

        for i in range(1, len(size_layer)):
            if i + 1 == len(size_layer):
                self.output = nn.Linear(size_layer[i - 1], size_layer[i])
            else:
                self.hiddens.append(nn.Linear(size_layer[i - 1], size_layer[i]))

        self.dropout = nn.Dropout(p=dropout)

        actives = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            "silu": nn.SiLU(),
            "gelu": nn.GELU(),
        }
        if not isinstance(activation, list):
            activation = [activation] * (len(size_layer) - 2)
        elif len(activation) == len(size_layer) - 2:
            pass
        else:
            raise ValueError(
                f"the length of activation function list except {len(size_layer) - 2} but got {len(activation)}!")
        self.hidden_active = []
        for func in activation:
            if callable(func):
                self.hidden_active.append(func)
            elif func.lower() in actives:
                self.hidden_active.append(actives[func])
            else:
                raise ValueError("should set activation correctly: {}".format(activation))
        if self.output_activation is not None:
            if callable(self.output_activation):
                pass
            elif self.output_activation.lower() in actives:
                self.output_activation = actives[self.output_activation]
            else:
                raise ValueError("should set activation correctly: {}".format(activation))
        # self._init_params()
    
    def _init_params(self):
        for layer in self.hiddens:
            nn.init.kaiming_normal_(layer.weight, nonlinearity=self.activation)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        #! add batchnorm
        for layer, func in zip(self.hiddens, self.hidden_active):
            x = self.dropout(func(layer(x)))
        x = self.output(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        x = self.dropout(x)
        return x


class YieldNet(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.desc_skip_connection = True
        self.fcs = []  # nn.ModuleList()
        print('dropout is {}'.format(dropout))
        self.dropout = nn.Dropout(dropout)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=16, stride=4),
            nn.ReLU(), # [32, 32, 32]
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(), # [32, 4, 4]
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=4),
            nn.ReLU(), # [32, 1, 1]
        )
        
        self.pred_head = nn.Linear(32, 1)

    def forward(self, reaction_fp):
        f = self.conv_layers(reaction_fp).reshape(reaction_fp.shape[0], -1)
        return self.pred_head(self.double(f))
        

def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class FinetunePredictor(nn.Module):
    def __init__(self, transformer, task, target, feat_dim=256, dropout=0.1, pool="mean", decoder_start_token_id=2):
        super().__init__()
    
        self.encoder = copy.deepcopy(transformer.encoder)
        self.decoder = copy.deepcopy(transformer.decoder)
        self.decoder_start_token_id = decoder_start_token_id
        self.task = task
        self.feat_dim = feat_dim
        if self.task == 'classification': # num_tasks?
            output_dim = len(target)
        elif self.task == 'regression':
            output_dim = 1
        self.pool = pool

        #! change the architecture of prediction head
        # if self.task == 'classification':
        #     self.pred_head = nn.Sequential(
        #         nn.Linear(self.feat_dim, self.feat_dim//2), 
        #         nn.Softplus(),
        #         nn.Linear(self.feat_dim//2, 2)
        #     )
        # elif self.task == 'regression':
        #     self.pred_head = nn.Sequential(
        #         nn.Linear(self.feat_dim, self.feat_dim//2), 
        #         # nn.Softplus(),
        #         nn.ReLU(),
        #         nn.Linear(self.feat_dim//2, 1)
        #     )

        self.pred_head = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim//2),
                                        nn.BatchNorm1d(self.feat_dim//2),
                                        nn.ReLU(),
                                        # nn.Softplus(),
                                        # nn.GELU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(self.feat_dim//2, output_dim))
        #TODO: initialize params
        # self.pred_head.apply(lambda module: init_params(module))

    def forward(self, x):
        _, token_emb, mol_emb = self.encoder(x.swapaxes(0,1))  #! x.shape=[batch_size, max_len]->[max_len, batch_size]

        # mean-pooling
        if self.pool == "mean":
            pred = self.pred_head(mol_emb)

        elif self.pool == "cls":
            pred = self.pred_head(token_emb.swapaxes(0,1)[:, 0])

        # use first tok of decoder (T5Chem)
        elif self.pool == "dec":
            decoder_input_ids = torch.full((1, x.size(0)),
                                            self.decoder_start_token_id,
                                            dtype=torch.long,
                                            device=x.device)
            enc_state = self.decoder.init_decoder_state(x.swapaxes(0,1), token_emb)
            decoder_output = self.decoder(decoder_input_ids, token_emb, enc_state)
            pred = self.pred_head(decoder_output[0].swapaxes(0,1)[:, 0])
        
        return pred