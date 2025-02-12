"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
import numpy as np

from models.module import PositionwiseFeedForward, MultiHeadedAttention, LayerNorm, CondAdapter
from models.petl_factory import Adapter_Layer

MAX_SIZE = 5000

    
class TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the PositionwiseFeedForward.
      dropout (float): dropout probability(0-1.0).
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, args, d_model, heads, d_ff, dropout, self_attn, context_attn, cond_attn=None):
        super(TransformerDecoderLayer, self).__init__()

        # self.self_attn_type = self_attn_type

        # if self_attn_type == "scaled-dot":
        #     self.self_attn = MultiHeadedAttention(
        #         heads, d_model, dropout=dropout)
        # elif self_attn_type == "average":
        #     self.self_attn = onmt.modules.AverageAttention(
        #         d_model, dropout=dropout)  

        self.self_attn = self_attn #! masked self-attention
        self.context_attn = context_attn  #! cross-attention (encoder-decoder attention)
        self.cond_attn = cond_attn
        self.feed_forward = PositionwiseFeedForward(args, d_model, d_ff, dropout)
        self.layer_norm_1 = LayerNorm(d_model)
        self.layer_norm_2 = LayerNorm(d_model)
        # self.layer_norm_1 = DiTAdaLayerNorm(d_model)
        # self.layer_norm_2 = DiTAdaLayerNorm(d_model)
        self.dropout = dropout
        self.drop = nn.Dropout(dropout)
        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer('mask', mask)
        
        self.layer_norm_cond = LayerNorm(d_model)
        self.adapter = CondAdapter(args, heads, d_model, d_ff, dropout)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                previous_input=None, layer_cache=None, 
                cond_emb=None, forward_adapter=False
                ):
        """
        Args:
            inputs (`FloatTensor`): `[batch_size x 1 x model_dim]`
            memory_bank (`FloatTensor`): `[batch_size x src_len x model_dim]`
            src_pad_mask (`LongTensor`): `[batch_size x 1 x src_len]` --> [batch_size x tgt_len x src_len]
            tgt_pad_mask (`LongTensor`): `[batch_size x 1 x 1]` --> [batch_size x tgt_len x tgt_len]
            cond_emb: [batch_size * num_cond * model_dim]

        Returns:
            (`FloatTensor`, `FloatTensor`, `FloatTensor`):

            * output `[batch_size x 1 x model_dim]`
            * attn `[batch_size x 1 x src_len]`
            * all_input `[batch_size x current_step x model_dim]`

        """
        dec_mask = torch.gt(tgt_pad_mask +
                            self.mask[:, :tgt_pad_mask.size(1),
                                      :tgt_pad_mask.size(1)], 0) 
        input_norm = self.layer_norm_1(inputs)

        #! self-attention
        all_input = input_norm
        if previous_input is not None:
            all_input = torch.cat((previous_input, input_norm), dim=1)
            dec_mask = None  # current_step

        query, self_attn = self.self_attn(all_input, all_input, input_norm,
                                          mask=dec_mask,
                                          layer_cache=layer_cache,
                                          type="self")

        query = self.drop(query) + inputs
        query_norm = self.layer_norm_2(query)

        #! enc-dec cross attention
        # memory_bank: (320, 8, 124, 32)
        # query_norm: (320, 8, 1, 32) --> (320, 8, 2, 32)
        # if cond_emb is not None:
        #     cond_emb_norm = self.layer_norm_cond(cond_emb)
        #     memory_bank = torch.cat((cond_emb_norm, memory_bank), dim=1)
        #     bs, tgt_len, src_len = src_pad_mask.shape
        #     src_pad_mask = torch.cat((torch.zeros(bs, tgt_len, cond_emb.shape[1]).bool().to(src_pad_mask.device),
        #                               src_pad_mask), dim=-1)

        mid, context_attn = self.context_attn(memory_bank, memory_bank, query_norm,
                                              mask=src_pad_mask,
                                              layer_cache=layer_cache,
                                              type="context")
        
        # if self.cond_attn is not None and cond_emb is not None:
        #     cond_emb_norm = self.layer_norm_cond(cond_emb)
        #     #! adapter layer with attn modules
        #     cond_mid, cond_attn = self.cond_attn(cond_emb_norm, cond_emb_norm, query_norm,
        #                                          layer_cache=layer_cache,
        #                                          type="context")
        #     mid = (mid + cond_mid) / 2

        output = self.feed_forward(self.drop(mid) + query)
        
        # forward adapter
        if cond_emb is not None and forward_adapter == True:
            output = output + self.adapter(output, cond_emb)

        return output, context_attn, all_input

    def _get_attn_subsequent_mask(self, size):
        """
        Get an attention mask to avoid using the subsequent info.

        Args:
            size: int

        Returns:
            (`LongTensor`):

            * subsequent_mask `[1 x size x size]`
        """
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask


class TransformerDecoder(nn.Module):
    """
    The Transformer decoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       dropout (float): dropout parameters
       embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
       attn_type (str): if using a seperate copy attention
    """

    def __init__(self, args, num_layers, d_model, heads, d_ff, dropout, embeddings,
                 self_attn_modules):
        super(TransformerDecoder, self).__init__()

        # Basic attributes.
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.cond_attn_layers = args.condition_attn_layers

        # Build cross-attention module
        context_attn_modules = nn.ModuleList(
            [MultiHeadedAttention(args, heads, d_model, dropout=dropout)
             for _ in range(num_layers)])
        
        # Build cross-attention module for reaction conditions
        cond_attn_modules = nn.ModuleList(
            [MultiHeadedAttention(args, heads, d_model, dropout=dropout)
             for _ in range(self.cond_attn_layers)])
        
        # Build TransformerDecoder.
        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(args, d_model, heads, d_ff, dropout,
             self_attn=self_attn_modules[i], context_attn=context_attn_modules[i])
             for i in range(num_layers-self.cond_attn_layers)] + \
            [TransformerDecoderLayer(args, d_model, heads, d_ff, dropout,
             self_attn=self_attn_modules[-1], context_attn=context_attn_modules[-1],
             cond_attn=cond_attn_modules[i])
             for i in range(self.cond_attn_layers)]
            )
        
        # self.transformer_layers = nn.ModuleList(
        #     [TransformerDecoderLayer(args, d_model, heads, d_ff, dropout,
        #      self_attn=self_attn_modules[i], context_attn=context_attn_modules[i])
        #      for i in range(num_layers)])
        
        self.layer_norm = LayerNorm(d_model)


    def forward(self, tgt, memory_bank, state, tgt_template_mask=None, step=None, cache=None, 
                template_pooling=False, cond_emb=None):
        """
        See :obj:`onmt.modules.RNNDecoderBase.forward()`
        """
        src = state.src
        # src_words = src[:, :, 0].transpose(0, 1)
        # tgt_words = tgt[:, :, 0].transpose(0, 1)
        src_words = src.transpose(0, 1)
        tgt_words = tgt.transpose(0, 1)
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()

        # Initialize return variables.
        attns = []

        # Run the forward pass of the TransformerDecoder.
        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3  # len x batch x embedding_dim
        if step is not None:
            tgt_words = tgt[-1].unsqueeze(0).transpose(0, 1)
            tgt_batch, tgt_len = tgt_words.size()

        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        padding_idx = self.embeddings.word_padding_idx
        src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(src_batch, tgt_len, src_len)

        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)
        
        cond_emb = cond_emb.transpose(0, 1) if cond_emb is not None else None

        if state.cache is None:
            saved_inputs = []

        for i in range(self.num_layers):
            prev_layer_input = None
            if state.cache is None:
                if state.previous_input is not None:
                    prev_layer_input = state.previous_layer_inputs[i]
                    
            # if i >= self.num_layers - self.cond_attn_layers:
            #     output, top_context_attn, all_input \
            #         = self.transformer_layers[i](
            #             output, src_memory_bank,
            #             src_pad_mask, tgt_pad_mask,
            #             previous_input=prev_layer_input,
            #             layer_cache=state.cache["layer_{}".format(i)]
            #             if state.cache is not None else None,
            #             cond_emb=cond_emb)
            # else:
            #     output, top_context_attn, all_input \
            #         = self.transformer_layers[i](
            #             output, src_memory_bank,
            #             src_pad_mask, tgt_pad_mask,
            #             previous_input=prev_layer_input,
            #             layer_cache=state.cache["layer_{}".format(i)]
            #             if state.cache is not None else None)
            
            output, top_context_attn, all_input \
                    = self.transformer_layers[i](
                        output, src_memory_bank,
                        src_pad_mask, tgt_pad_mask,
                        previous_input=prev_layer_input,
                        layer_cache=state.cache["layer_{}".format(i)]
                        if state.cache is not None else None,
                        cond_emb=cond_emb, forward_adapter=True if cond_emb is not None else False)
                    
            attns.append(top_context_attn)  #! head-0 context attention values (batch_size, query_len, key_len)-->(batch_size, tgt_len, src_len)
            if state.cache is None:
                saved_inputs.append(all_input)

        output = self.layer_norm(output)

        # Process the result and update the attentions.

        tgt_reps = torch.einsum("abc,ab->ac", output, (~tgt_pad_mask[:, 0, :]).float()) / torch.sum(~tgt_pad_mask[:, 0, :], dim=1).unsqueeze(1)  
    
        if state.cache is None:
            saved_inputs = torch.stack(saved_inputs)
            state = state.update_state(tgt, saved_inputs)

        if not template_pooling:
            return output.transpose(0, 1).contiguous(), state, attns, tgt_reps
        else: 
            tgt_template_reps = torch.einsum("abc,ba->ac", output, tgt_template_mask[1:, :]) / torch.sum(tgt_template_mask[1:, :], dim=0).unsqueeze(1)
            return output.transpose(0, 1).contiguous(), state, attns, tgt_reps, tgt_template_reps
 
    def forward_with_prompt(self, tgt, memory_bank, state, step=None, cache=None):
        src = state.src
        src_words = src.transpose(0, 1)
        tgt_words = tgt.transpose(0, 1)
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()

        # Initialize return variables.
        attns = []

        # Run the forward pass of the TransformerDecoder.
        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3  # len x batch x embedding_dim
        if step is not None:
            tgt_words = tgt[-1].unsqueeze(0).transpose(0, 1)
            tgt_batch, tgt_len = tgt_words.size()

        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        padding_idx = self.embeddings.word_padding_idx
        prompt_len = src_memory_bank.shape[1] - src_len
        src_words_mask = src_words.data.eq(padding_idx)
        src_pad_mask = torch.cat([torch.zeros(src_batch, prompt_len).bool().to(src_words_mask.device), src_words_mask], dim=1)
        src_pad_mask = src_pad_mask.unsqueeze(1).expand(tgt_batch, tgt_len, src_len + prompt_len)

        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)

        if state.cache is None:
            saved_inputs = []

        for i in range(self.num_layers):
            prev_layer_input = None
            if state.cache is None:
                if state.previous_input is not None:
                    prev_layer_input = state.previous_layer_inputs[i]
            output, top_context_attn, all_input \
                = self.transformer_layers[i](
                    output, src_memory_bank,
                    src_pad_mask, tgt_pad_mask,
                    previous_input=prev_layer_input,
                    layer_cache=state.cache["layer_{}".format(i)]
                    if state.cache is not None else None)
            attns.append(top_context_attn) 
            if state.cache is None:
                saved_inputs.append(all_input)


        if state.cache is None:
            saved_inputs = torch.stack(saved_inputs)

        output = self.layer_norm(output)
    
        if state.cache is None:
            state = state.update_state(tgt, saved_inputs)

        return output.transpose(0, 1).contiguous(), state, attns
    
    def init_decoder_state(self, src, memory_bank,
                           with_cache=False):
        """ Init decoder state """
        state = TransformerDecoderState(src)
        if with_cache:
            state._init_cache(memory_bank, self.num_layers)
        return state


class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.

    Modules need to implement this to utilize beam search decoding.
    """
    def detach(self):
        """ Need to document this """
        self.hidden = tuple([_.detach() for _ in self.hidden])
        self.input_feed = self.input_feed.detach()

    def beam_update(self, idx, positions, beam_size):
        """ Need to document this """
        for e in self._all:
            sizes = e.size()
            br = sizes[1]  # (beam_size * batch_size)
            if len(sizes) == 3:
                sent_states = e.view(sizes[0], beam_size, br // beam_size,
                                     sizes[2])[:, :, idx]
            elif len(sizes) == 4:
                sent_states = e.view(sizes[0], beam_size,
                                     br // beam_size,
                                     sizes[2],
                                     sizes[3])[:, :, idx]
            else:
                sent_states = e.view(sizes[0], beam_size,
                                     br // beam_size)[:, :, idx]
            try:
                sent_states.data.copy_(
                    sent_states.data.index_select(1, positions))  # reorder decoder state (finished beams are placed last)
            except:
                print("error")

    def map_batch_fn(self, fn):
        raise NotImplementedError()


class TransformerDecoderState(DecoderState):
    """ Transformer Decoder state base class """

    def __init__(self, src):
        """
        Args:
            src (FloatTensor): a sequence of source words tensors
                    with optional feature tensors, of size (len x batch).
        """
        self.src = src
        self.previous_input = None
        self.previous_layer_inputs = None
        self.cache = None

    @property
    def _all(self):
        """
        Contains attributes that need to be updated in self.beam_update().
        """
        if (self.previous_input is not None
                and self.previous_layer_inputs is not None):
            return (self.previous_input,
                    self.previous_layer_inputs,
                    self.src)
        else:
            return (self.src,)

    def detach(self):
        if self.previous_input is not None:
            self.previous_input = self.previous_input.detach()
        if self.previous_layer_inputs is not None:
            self.previous_layer_inputs = self.previous_layer_inputs.detach()
        self.src = self.src.detach()

    def update_state(self, new_input, previous_layer_inputs):
        state = TransformerDecoderState(self.src)
        state.previous_input = new_input.clone()
        state.previous_layer_inputs = previous_layer_inputs
        return state

    def _init_cache(self, memory_bank, num_layers, self_attn_type):
        self.cache = {}
        batch_size = memory_bank.size(1)
        depth = memory_bank.size(-1)

        for l in range(num_layers):
            layer_cache = {
                "memory_keys": None,
                "memory_values": None,
                "self_keys": None,
                "self_values": None
            }
            self.cache["layer_{}".format(l)] = layer_cache

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.src = self.src.data.repeat(1, beam_size, 1)

    def map_batch_fn(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.src = fn(self.src, 1)
        if self.cache is not None:
            _recursive_map(self.cache)