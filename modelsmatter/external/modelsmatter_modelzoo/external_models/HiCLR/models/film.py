#!/usr/bin/env python3

import ipdb as pdb
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import kaiming_normal, kaiming_uniform
from models.masked_layers import *


def init_modules(modules, init='uniform'):
  if init.lower() == 'normal':
    init_params = kaiming_normal
  elif init.lower() == 'uniform':
    init_params = kaiming_uniform
  else:
    return
  for m in modules:
    if isinstance(m, (nn.Conv2d, nn.Linear)):
      init_params(m.weight)


class FiLMGen(nn.Module):
  def __init__(self,
    null_token=0,
    start_token=1,
    end_token=2,
    encoder_embed=None,
    encoder_vocab_size=100,
    decoder_vocab_size=100,
    wordvec_dim=200,
    hidden_dim=512,
    rnn_num_layers=1,
    rnn_dropout=0,
    output_batchnorm=False,
    bidirectional=False,
    encoder_type='gru',
    decoder_type='linear',
    gamma_option='linear',
    gamma_baseline=1,
    num_modules=4,
    module_num_layers=1,
    module_dim=128,
    parameter_efficient=True,
    debug_every=float('inf'),
    use_encoder=False,
    kernel_size=3,
  ):
    super(FiLMGen, self).__init__()
    self.encoder_type = encoder_type
    self.decoder_type = decoder_type
    self.output_batchnorm = output_batchnorm
    self.bidirectional = bidirectional
    self.num_dir = 2 if self.bidirectional else 1
    self.gamma_option = gamma_option
    self.gamma_baseline = gamma_baseline
    self.num_modules = num_modules
    self.module_num_layers = module_num_layers
    self.module_dim = module_dim
    self.debug_every = debug_every
    self.NULL = null_token
    self.START = start_token
    self.END = end_token
    if self.bidirectional:
      if decoder_type != 'linear':
        raise(NotImplementedError)
      hidden_dim = (int) (hidden_dim / self.num_dir)

    self.func_list = {
      'linear': None,
      'sigmoid': F.sigmoid,
      'tanh': F.tanh,
      'exp': torch.exp,
    }

    self.cond_feat_size = 2 * self.module_dim * self.module_num_layers  # FiLM params per ResBlock
    if not parameter_efficient:  # parameter_efficient=False only used to load older trained models
      self.cond_feat_size = 4 * self.module_dim + 2 * self.num_modules

    self.encoder_embed = nn.Embedding(encoder_vocab_size, wordvec_dim)

    if self.encoder_type == "lstm" or self.encoder_type == "gru":
      self.encoder_rnn = init_rnn(self.encoder_type, wordvec_dim, hidden_dim, rnn_num_layers,
                                  dropout=rnn_dropout, bidirectional=self.bidirectional)
    elif self.encoder_type == "cnn":
      self.encoder_cnn = init_cnn(wordvec_dim, hidden_dim)
      self.encoder_rnn = init_rnn("gru", wordvec_dim, hidden_dim, rnn_num_layers,
                                  dropout=rnn_dropout, bidirectional=self.bidirectional)
    
    self.decoder_rnn = init_rnn(self.decoder_type, hidden_dim, hidden_dim, rnn_num_layers,
                                dropout=rnn_dropout, bidirectional=self.bidirectional)
    self.decoder_linear = nn.Linear(
      hidden_dim * self.num_dir, self.num_modules * self.cond_feat_size)
    if self.output_batchnorm:
      self.output_bn = nn.BatchNorm1d(self.cond_feat_size, affine=True)

    init_modules(self.modules())
    self.use_encoder = use_encoder

  def get_dims(self, x=None):
    V_in = self.encoder_embed.num_embeddings
    V_out = self.cond_feat_size
    D = self.encoder_embed.embedding_dim
    H = self.encoder_rnn.hidden_size
    H_full = self.encoder_rnn.hidden_size * self.num_dir
    L = self.encoder_rnn.num_layers * self.num_dir

    N = x.size(0) if x is not None else None
    T_in = x.size(1) if x is not None else None
    T_out = self.num_modules
    return V_in, V_out, D, H, H_full, L, N, T_in, T_out

  def before_rnn(self, x, replace=0):
    N, T = x.size()
    idx = torch.LongTensor(N).fill_(T - 1)

    # Find the last non-null element in each sequence.
    x_cpu = x.cpu()
    for i in range(N):
      for t in range(T - 1):
        if x_cpu.data[i, t] != self.NULL and x_cpu.data[i, t + 1] == self.NULL:
          idx[i] = t
          break
    idx = idx.type_as(x.data)
    x[x.data == self.NULL] = replace
    return x, Variable(idx)

  def encoder(self, x, mask):
   
    V_in, V_out, D, H, H_full, L, N, T_in, T_out = self.get_dims(x=x)

    if x.shape == 2:
      x, idx = self.before_rnn(x)  # Tokenized word sequences (questions), end index
      embed = self.encoder_embed(x)
    else:
      embed = x
      idx = mask.sum(1) - 1
    h0 = Variable(torch.zeros(L, N, H).type_as(embed.data))

    if self.encoder_type == 'lstm':
      c0 = Variable(torch.zeros(L, N, H).type_as(embed.data))
      out, _ = self.encoder_rnn(embed, (h0, c0))
    elif self.encoder_type == 'gru':
      out, _ = self.encoder_rnn(embed, h0)
    elif self.encoder_type == "cnn":
      out, _ = self.encoder_cnn((embed, mask))
      out = out.sum(1) / mask.sum(1)[:, None]

    # Pull out the hidden state for the last non-null value in each input
    # idx = idx.view(N, 1, 1).expand(N, 1, H_full)
    # return out.gather(1, idx).view(N, H_full)

    return out

  def decoder(self, encoded, dims, h0=None, c0=None):
    V_in, V_out, D, H, H_full, L, N, T_in, T_out = dims

    if self.decoder_type == 'linear':
      # (N x H) x (H x T_out*V_out) -> (N x T_out*V_out) -> N x T_out x V_out
      return self.decoder_linear(encoded).view(N, T_out, V_out), (None, None)

    encoded_repeat = encoded.view(N, 1, H).expand(N, T_out, H)
    if not h0:
      h0 = Variable(torch.zeros(L, N, H).type_as(encoded.data))

    if self.decoder_type == 'lstm':
      if not c0:
        c0 = Variable(torch.zeros(L, N, H).type_as(encoded.data))
      rnn_output, (ht, ct) = self.decoder_rnn(encoded_repeat, (h0, c0))
    elif self.decoder_type == 'gru':
      ct = None
      rnn_output, ht = self.decoder_rnn(encoded_repeat, h0)

    rnn_output_2d = rnn_output.contiguous().view(N * T_out, H)
    linear_output = self.decoder_linear(rnn_output_2d)
    if self.output_batchnorm:
      linear_output = self.output_bn(linear_output)

    # output_shaped = linear_output.view(N, T_out, V_out)
    output_shaped = linear_output.view(N, T_out, -1)[..., :V_out]
    return output_shaped, (ht, ct)

  def forward(self, x, mask=None):
    # input prot_emb: decoder(linear)
    # input res_rmb: encoder(gru) --> decoder(linear)
    if self.debug_every <= -2:
      pdb.set_trace()
    
    if self.use_encoder:
      x = self.encoder(x, mask)
    film_pre_mod, _ = self.decoder(x, self.get_dims(x=x))
    film = self.modify_output(film_pre_mod, gamma_option=self.gamma_option,
                              gamma_shift=self.gamma_baseline)
    gammas, betas = torch.split(film[:,:,:2*self.module_dim], self.module_dim, dim=-1)
    return gammas, betas, x
  
  def modify_output(self, out, gamma_option='linear', gamma_scale=1, gamma_shift=0,
                    beta_option='linear', beta_scale=1, beta_shift=0):
    gamma_func = self.func_list[gamma_option]
    beta_func = self.func_list[beta_option]

    gs = []
    bs = []
    for i in range(self.module_num_layers):
      gs.append(slice(i * (2 * self.module_dim), i * (2 * self.module_dim) + self.module_dim))
      bs.append(slice(i * (2 * self.module_dim) + self.module_dim, (i + 1) * (2 * self.module_dim)))

    if gamma_func is not None:
      for i in range(self.module_num_layers):
        out[:,:,gs[i]] = gamma_func(out[:,:,gs[i]])
    if gamma_scale != 1:
      for i in range(self.module_num_layers):
        out[:,:,gs[i]] = out[:,:,gs[i]] * gamma_scale
    if gamma_shift != 0:
      for i in range(self.module_num_layers):
        out[:,:,gs[i]] = out[:,:,gs[i]] + gamma_shift
    if beta_func is not None:
      for i in range(self.module_num_layers):
        out[:,:,bs[i]] = beta_func(out[:,:,bs[i]])
      out[:,:,b2] = beta_func(out[:,:,b2])
    if beta_scale != 1:
      for i in range(self.module_num_layers):
        out[:,:,bs[i]] = out[:,:,bs[i]] * beta_scale
    if beta_shift != 0:
      for i in range(self.module_num_layers):
        out[:,:,bs[i]] = out[:,:,bs[i]] + beta_shift
    return out

def init_rnn(rnn_type, hidden_dim1, hidden_dim2, rnn_num_layers,
             dropout=0, bidirectional=False):
  if rnn_type == 'gru':
    return nn.GRU(hidden_dim1, hidden_dim2, rnn_num_layers, dropout=dropout,
                  batch_first=True, bidirectional=bidirectional)
  elif rnn_type == 'lstm':
    return nn.LSTM(hidden_dim1, hidden_dim2, rnn_num_layers, dropout=dropout,
                   batch_first=True, bidirectional=bidirectional)
  elif rnn_type == 'linear':
    return None
  else:
    print('RNN type ' + str(rnn_type) + ' not yet implemented.')
    raise(NotImplementedError)

def init_cnn(input_dim, embed_dim, kernel_size=3, layernorm=True, p=0.1):
      return nn.Sequential(
            Apply(Expression(lambda x: x.permute(0, 2, 1))),  # (B,N,C) -> (B,C,N)
            # mResidualBlock(embed_dim, embed_dim, kernel_size, layernorm),
            mResidualBlock(input_dim, embed_dim, kernel_size, layernorm, p),
            mResidualBlock(embed_dim, embed_dim, kernel_size, layernorm, p),
            mResidualBlock(embed_dim, embed_dim, kernel_size, layernorm, p),
            # Apply(nn.Dropout(p=p)),
            Apply(Expression(lambda x: x.permute(0, 2, 1))),  # (B,C,N) -> (B,N,C)
        )