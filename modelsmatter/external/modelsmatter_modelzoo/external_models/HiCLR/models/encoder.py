"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn
import torch
import numpy as np

from models.module import PositionwiseFeedForward, MultiHeadedAttention, LayerNorm, CondAdapter
from models.petl_factory import Adapter_Layer


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)
 
    
class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, args, d_model, heads, d_ff, dropout, attn):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = attn
        self.feed_forward = PositionwiseFeedForward(args, d_model, d_ff, dropout)
        self.layer_norm = LayerNorm(d_model)
        # self.layer_norm = DiTAdaLayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.adapter = CondAdapter(args, heads, d_model, d_ff, dropout)

    def forward(self, inputs, mask, cond_emb=None, forward_adapter=False):
        """
        Transformer Encoder Layer definition.

        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`

        Returns:
            (`FloatTensor`):

            * outputs `[batch_size x src_len x model_dim]`
        """
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm, mask=mask) 
        out = self.dropout(context) + inputs
        out = self.feed_forward(out) # input layer_norm within FFN
        
        # cond adapter (similar to structure adapter)
        if cond_emb is not None and forward_adapter == True:
            out = out + self.adapter(out, cond_emb)
        
        return out


class TransformerEncoder(nn.Module):
    """
    The Transformer encoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings

    Returns:
        (`FloatTensor`, `FloatTensor`):

        * embeddings `[src_len x batch_size x model_dim]`
        * memory_bank `[src_len x batch_size x model_dim]`
    """

    def __init__(self, args, num_layers, d_model, heads, d_ff,
                 dropout, embeddings, attn_modules, prompt_embeddings=None):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(args, d_model, heads, d_ff, dropout, attn_modules[i])
             for i in range(num_layers)])
        self.layer_norm = LayerNorm(d_model)
        
        if args.prompt:
            self.prompt_embeddings = prompt_embeddings
        if args.input_prompt_attn:
            self.input_proj = nn.Sequential(
                nn.Linear(d_model, 100),
                nn.SiLU(),
                nn.Linear(100, d_model)
            )
            self.softmax_temp = d_model * np.exp(1)


    def forward(self, src, src_template_mask=None, template_pooling=False, cond_emb=None): 
        """ See :obj:`EncoderBase.forward()`"""
        
        emb = self.embeddings(src)  # (max_len, batch_size, emb_dim)
        
        words = src.transpose(0, 1)
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1).expand(w_batch, w_len, w_len)
        
        out = emb.transpose(0, 1).contiguous()
        cond_emb = cond_emb.transpose(0, 1) if cond_emb is not None else None

        # Run the forward pass of every layer of the transformer.
        for i in range(self.num_layers):
            out = self.transformer[i](out, mask, cond_emb, forward_adapter=False)
        out = self.layer_norm(out)  
        # *out: [batch_size x src_len x model_dim]
        # *mask[:,0,:]: [batch_size, src_len]
        # *src_template_mask: [src_len, batch_size]

        src_reps = torch.einsum("abc,ab->ac", out, (~mask[:, 0, :]).float()) / torch.sum(~mask[:, 0, :], dim=1).unsqueeze(1)
        #* sum or mean (reactions with different length?)

        if template_pooling:
            src_template_reps = torch.einsum("abc,ba->ac", out, src_template_mask) / torch.sum(src_template_mask, dim=0).unsqueeze(1)
            return emb, out.transpose(0, 1).contiguous(), src_reps, src_template_reps
        else:
            return emb, out.transpose(0, 1).contiguous(), src_reps

    def forward_with_given_prompt(self, src, prompt):
        src_emb = self.embeddings(src)  # (max_len, batch_size, emb_dim)
        prompt_emb = self.prompt_embeddings(prompt) # (prompt_len, batch_size, emb_dim)
        emb = torch.cat([prompt_emb, src_emb], dim=0)
        out = emb.transpose(0, 1).contiguous()
        
        words = src.transpose(0, 1)
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx
        words_mask = words.data.eq(padding_idx)
        prompt_len = prompt.size(0)
        mask = torch.cat([torch.zeros(w_batch, prompt_len).bool().to(words_mask.device), words_mask], dim=1)
        mask = mask.unsqueeze(1).expand(w_batch, w_len+prompt_len, w_len+prompt_len)
        
        # Run the forward pass of every layer of the transformer.
        for i in range(self.num_layers):
            out = self.transformer[i](out, mask)
        out = self.layer_norm(out) 
        
        return emb, out.transpose(0, 1).contiguous()
        
    def forward_with_pred_prompt(self):
        pass
    
        #     prompt_emb_all = self.prompt_proj(self.embedding_prompt.weight).to(emb.device)  # (num_proto, d_model)
        #     emb_sum = torch.einsum("abc,ab->abc", emb.transpose(0, 1), (~mask[:,0,:])).sum(dim=1)  # (w_batch, d_model)
        #     # emb_sum = torch.einsum("abc,ab->ac", emb.transpose(0, 1), (~mask[:,0,:]).float())
        #     emb_sum = self.layer_norm(self.x_proj(emb_sum))
        #     prompt_attn = torch.matmul(emb_sum, prompt_emb_all.transpose(0,1)) #* (w_batch, d_model)*(d_model, num_proto)-->(w_batch, num_proto)
        #     prompt_attn = (torch.exp(prompt_attn) / self.softmax_temp) / (torch.exp(prompt_attn) / self.softmax_temp).sum(dim=1).unsqueeze(1)
        #     #? converge to a single prompt
        #     #? dot product v.s. cosine similarity
        #     prompt_target = torch.matmul(prompt_attn, prompt_emb_all) #* (w_batch, num_proto)*(num_proto, d_model)-->(w_batch, d_model)
        #
        #     if self.prompt_version == "v1":
        #         emb = torch.cat([prompt_target.unsqueeze(1).transpose(0, 1), emb], dim=0) #* (1, w_batch, d_model)
        #         mask = torch.cat([torch.zeros(w_batch,1).bool().to(mask.device), words.data.eq(padding_idx)],dim=1)
        #         mask = mask.unsqueeze(1).expand(-1, w_len+1, w_len+1)
        #     else:
        #         emb[0, :, :] += prompt_target