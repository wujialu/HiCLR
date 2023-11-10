"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn
import torch

from models.module import PositionwiseFeedForward, MultiHeadedAttention, LayerNorm
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
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask, prefix_key=None, prefix_value=None):
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
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask, prefix_key=prefix_key, prefix_value=prefix_value) 
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


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
                 dropout, embeddings, attn_modules):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.embeddings = embeddings
        self.prompt_embeddings = nn.Embedding(num_embeddings=1, embedding_dim=d_model)
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(args, d_model, heads, d_ff, dropout, attn_modules[i])
             for i in range(num_layers)])
        self.layer_norm = LayerNorm(d_model)


    def forward(self, src, src_template_mask=None, reaction_reps=None, template_pooling=False, prompt=False, prefix_key=None, prefix_value=None): 
        """ See :obj:`EncoderBase.forward()`"""
        
        emb = self.embeddings(src)  # torch.tensor(max_len, batch_size, emb_dim)
        # if prompt:
        #     prompt_emb = self.prompt_embeddings(torch.zeros(1).int().to(src.device))  # (1, emb_dim)
        #     emb = torch.cat([emb, prompt_emb.unsqueeze(1).expand(-1, emb.shape[1], -1)], dim=0) # (max_len+1, batch_size, emb_dim)
        
        if reaction_reps is not None and prompt:
            prompt_emb = reaction_reps  # (batch_size, emb_dim)
            emb[0, :, :] = prompt_emb  
        
        out = emb.transpose(0, 1).contiguous()

        # words = src[:, :, 0].transpose(0, 1)
        words = src.transpose(0, 1)
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1).expand(w_batch, w_len, w_len)
        
        # if not prompt:
        #     mask = words.data.eq(padding_idx).unsqueeze(1).expand(w_batch, w_len, w_len)
        # else:
        #     mask = words.data.eq(padding_idx)
        #     mask = torch.cat([torch.zeros(mask.shape[0],1).bool().to(mask.device),mask],dim=1)
        #     mask = mask.unsqueeze(1).expand(w_batch, w_len+1, w_len+1)

        # Run the forward pass of every layer of the transformer.
        for i in range(self.num_layers):
            out = self.transformer[i](out, mask, prefix_key=prefix_key, prefix_value=prefix_value)
        out = self.layer_norm(out)  
        # *out: [batch_size x src_len x model_dim]
        # *mask[:,0,:]: [batch_size, src_len]
        # *src_template_mask: [src_len, batch_size]

        src_reps = torch.einsum("abc,ab->ac", out, (mask[:, 0, :]==0).float()) 

        if not template_pooling:
            return emb, out.transpose(0, 1).contiguous(), src_reps
        else: 
            src_template_reps = torch.einsum("abc,ba->ac", out, src_template_mask)
            return emb, out.transpose(0, 1).contiguous(), src_reps, src_template_reps
