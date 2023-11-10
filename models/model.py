""" Onmt NMT Model base class definition """
import torch.nn as nn
import torch

from models.embedding import Embedding, TokenEmbedding
from models.module import MultiHeadedAttention
from models.encoder import TransformerEncoder
from models.decoder import TransformerDecoder

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
                 tgt_pad_idx=1, multigpu=False):
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
        if shared_vocab:
            assert vocab_size_src == vocab_size_tgt and src_pad_idx == tgt_pad_idx
            self.embedding_src = self.embedding_tgt = Embedding(vocab_size=vocab_size_src + 1, embed_size=d_model,
                                                                padding_idx=src_pad_idx)
        else:
            self.embedding_src = Embedding(vocab_size=vocab_size_src + 1, embed_size=d_model, padding_idx=src_pad_idx)
            self.embedding_tgt = Embedding(vocab_size=vocab_size_tgt + 1, embed_size=d_model, padding_idx=tgt_pad_idx)

        multihead_attn_modules_en = nn.ModuleList(
            [MultiHeadedAttention(heads, d_model, dropout=dropout)
             for _ in range(encoder_num_layers)])
        if shared_encoder:
            assert encoder_num_layers == decoder_num_layers
            multihead_attn_modules_de = multihead_attn_modules_en
        else:
            multihead_attn_modules_de = nn.ModuleList(
                [MultiHeadedAttention(heads, d_model, dropout=dropout)
                 for _ in range(decoder_num_layers)])

        self.encoder = TransformerEncoder(args, num_layers=encoder_num_layers,
                                          d_model=d_model, heads=heads,
                                          d_ff=d_ff, dropout=dropout,
                                          embeddings=self.embedding_src,
                                          attn_modules=multihead_attn_modules_en)

        self.decoder = TransformerDecoder(args, num_layers=decoder_num_layers,
                                          d_model=d_model, heads=heads,
                                          d_ff=d_ff, dropout=dropout,
                                          embeddings=self.embedding_tgt,
                                          self_attn_modules=multihead_attn_modules_de)

        self.generator = nn.Sequential(nn.Linear(d_model, vocab_size_tgt),
                                       nn.LogSoftmax(dim=-1))

        self.softmax = nn.Softmax(dim=-1)
        
        self.attn_bn = args.attn_bn
        self.attn_dim = args.attn_dim
        self.attn_mode = args.attn_mode
        if self.attn_mode == "prefix":
            self.prefix_tokens = torch.arange(self.attn_bn).long().to(args.device)
            # TODO: retro model for prefix initialization(instance-level prefix?)
            self.prefix_key = nn.Embedding(self.attn_bn, self.attn_dim)
            self.prefix_value = nn.Embedding(self.attn_bn, self.attn_dim)
            self.prefix_map = MLP(size_layer=[self.attn_dim, self.d_model], dropout=dropout)

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
            decoder_outputs, dec_state, attns, tgt_reps, tgt_template_reps = self.decoder(tgt, encoder_outputs,
                                                                                          enc_state if dec_state is None
                                                                                          else dec_state,
                                                                                          tgt_template_mask=tgt_template_mask, template_pooling=template_pooling)
        elif self.attn_mode == "prefix":
            prefix_key = self.prefix_map(self.prefix_key(self.prefix_tokens))
            prefix_value = self.prefix_map(self.prefix_value(self.prefix_tokens))
            src_emb, encoder_outputs, src_reps = self.encoder(src, src_template_mask=src_template_mask, prefix_key=prefix_key, prefix_value=prefix_value)
            enc_state = \
                self.decoder.init_decoder_state(src, encoder_outputs)
            decoder_outputs, dec_state, attns, tgt_reps = self.decoder(tgt, encoder_outputs,
                                                                       enc_state if dec_state is None
                                                                       else dec_state,
                                                                       tgt_template_mask=tgt_template_mask, 
                                                                       prefix_key=prefix_key, prefix_value=prefix_value)
        
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

    def extract_reaction_fp(self, src, tgt, dec_state=None, reaction_reps=None):
        tgt = tgt[:-1]
        if reaction_reps is not None:
            if reaction_reps.shape[1] == self.d_model * 2:
                reaction_reps = self.linear(reaction_reps)
            src_emb, encoder_outputs, src_reps = self.encoder(src, reaction_reps=reaction_reps, prompt=True)
        else:
            src_emb, encoder_outputs, src_reps = self.encoder(src)
        enc_state = \
            self.decoder.init_decoder_state(src, encoder_outputs)
        decoder_outputs, dec_state, attns, tgt_reps = self.decoder(tgt, encoder_outputs,
                                                                    enc_state if dec_state is None
                                                                    else dec_state)
        return src_reps, tgt_reps

    def extract_prompt_rep(self, src, tgt, dec_state=None, reaction_reps=None):
        tgt = tgt[:-1]
        if reaction_reps.shape[1] == self.d_model * 2:
            reaction_reps = self.linear(reaction_reps)
        src_emb, encoder_outputs, src_reps = self.encoder(src, reaction_reps=reaction_reps, prompt=True)
        return encoder_outputs[0, :, :]  # [src_len x batch_size x model_dim]

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
    def __init__(self, rep_dim):
        super(ProjectNet, self).__init__()
        self.rep_dim = rep_dim
        self.proj = nn.Sequential(
            nn.Linear(self.rep_dim, self.rep_dim),
            nn.ReLU(),
            nn.Linear(self.rep_dim, self.rep_dim)
        )

    def forward(self, x):
        x_proj = self.proj(x)

        return x_proj


class MLP(nn.Module):
    def __init__(self, size_layer, activation='relu', output_activation=None, initial_method=None, dropout=0.0):
        super(MLP, self).__init__()
        self.hiddens = nn.ModuleList()
        self.output = None
        self.output_activation = output_activation
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
        # initial_parameter(self, initial_method)

    def forward(self, x):
        for layer, func in zip(self.hiddens, self.hidden_active):
            x = self.dropout(func(layer(x)))
        x = self.output(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        x = self.dropout(x)
        return x


class AdapterMT(MolecularTransformer):
    """
    encoder(adapter=True)
    decoder(adapter=True)
    """
    pass