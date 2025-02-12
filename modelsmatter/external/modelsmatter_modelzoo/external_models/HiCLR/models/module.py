import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.petl_factory import Adapter_Layer
import numpy as np


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

        Args:
            d_model (int): the size of input for the first-layer of the FFN.
            d_ff (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, args, d_model, d_ff, dropout=0.1, use_adapter=False):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)
        self.use_adapter = use_adapter
        if self.use_adapter:
            self.ffn_adapter = Adapter_Layer(d_model=d_model, bottleneck=args.ffn_bn, dropout=dropout,
                                            init_option=args.ffn_adapter_init_option, 
                                            adapter_scalar=args.ffn_adapter_scalar,
                                            adapter_layernorm_option=args.ffn_adapter_layernorm_option)
        self.ffn_option = args.ffn_option

    def forward(self, x):
        """
        Layer definition.

        Args:
            input: [ batch_size, input_len, model_dim ]


        Returns:
            output: [ batch_size, input_len, model_dim ]
        """
        #! input layer_norm in FFN layer
        if self.use_adapter:
            if self.ffn_option == "sequential":
                inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
                output = self.dropout_2(self.w_2(inter))
                output = self.ffn_adapter(output)
            elif self.ffn_option == "parallel":
                x_norm = self.layer_norm(x)
                inter = self.dropout_1(self.relu(self.w_1(x_norm)))
                output = self.dropout_2(self.w_2(inter))
                adpater_output = self.ffn_adapter(x_norm, add_residual=False)
                output += adpater_output
        else:
            inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
            output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, args, head_count, model_dim, dropout=0.1, use_adapter=False):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

        self.use_adapter = use_adapter
        self.attn_mode = args.attn_mode
        if self.use_adapter:
            self.attn_adapter = Adapter_Layer(d_model=model_dim, bottleneck=args.ffn_bn, dropout=dropout,
                                              init_option=args.ffn_adapter_init_option, 
                                              adapter_scalar=args.ffn_adapter_scalar,
                                              adapter_layernorm_option=args.ffn_adapter_layernorm_option)


    def forward(self, key, value, query, mask=None,
                layer_cache=None, type=None, prefix_key=None, prefix_value=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :

           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """

        # CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # aeq(self.model_dim % 8, 0)
        # if mask is not None:
        #    batch_, q_len_, k_len_ = mask.size()
        #    aeq(batch_, batch)
        #    aeq(k_len_, k_len)
        #    aeq(q_len_ == q_len)
        # END CHECKS

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if type == "self":
                query, key, value = self.linear_query(query),\
                                    self.linear_keys(query),\
                                    self.linear_values(query)

                key = shape(key)
                value = shape(value)

                if layer_cache is not None:
                    device = key.device
                    if layer_cache["self_keys"] is not None:
                        key = torch.cat(
                            (layer_cache["self_keys"].to(device), key),
                            dim=2)
                    if layer_cache["self_values"] is not None:
                        value = torch.cat(
                            (layer_cache["self_values"].to(device), value),
                            dim=2)
                    layer_cache["self_keys"] = key
                    layer_cache["self_values"] = value
            elif type == "context":
                query = self.linear_query(query)
                if layer_cache is not None:
                    if layer_cache["memory_keys"] is None:
                        key, value = self.linear_keys(key),\
                                     self.linear_values(value)
                        key = shape(key)
                        value = shape(value)
                    else:
                        key, value = layer_cache["memory_keys"],\
                                   layer_cache["memory_values"]
                    layer_cache["memory_keys"] = key
                    layer_cache["memory_values"] = value
                else:
                    key, value = self.linear_keys(key),\
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        query = shape(query)

        key_len = key.size(2)
        query_len = query.size(2)
        
        # concat prefix to original keys and values
        if prefix_key is not None and prefix_value is not None:
            prefix_key = shape(prefix_key.unsqueeze(0).expand(batch_size, -1, -1))
            prefix_value = shape(prefix_value.unsqueeze(0).expand(batch_size, -1, -1))
            key = torch.cat([prefix_key, key], dim=2)
            value = torch.cat([prefix_value, value], dim=2)
            
            key_len = key.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))  #! (br, n_heads, len_seq, len_seq)

        if mask is not None:
            if prefix_key is not None and prefix_value is not None:
                prefix_mask = torch.zeros(prefix_key.shape[0]).expand(mask.size(0), mask.size(1), -1).bool().to(mask.device)
                mask = torch.cat([prefix_mask, mask], dim=-1)
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18) #! if mask=1, fill with value 1e-18

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = unshape(torch.matmul(drop_attn, value))

        output = self.final_linear(context)
        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        # Return one-head attention
        top_attn = attn \
            .view(batch_size, head_count,
                  query_len, key_len)[:, 0, :, :] \
            .contiguous() 

        all_attn = attn \
            .view(batch_size, head_count,
                  query_len, key_len) \
            .contiguous() 
        
        if self.use_adapter:
            output = self.attn_adapter(output)
            
        return output, top_attn


class CondAdapter(nn.Module):
    def __init__(self, args, heads, d_model, d_ff, dropout):
        super(CondAdapter, self).__init__()
        self.args = args
        self.layer_norm_1 = LayerNorm(d_model)
        self.layer_norm_2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.context_attn = MultiHeadedAttention(args, heads, d_model, dropout=dropout)
        # 256 --> 64
        self.feed_forward = Adapter_Layer(d_model=d_model, bottleneck=args.ffn_bn, dropout=dropout,
                                            init_option=args.ffn_adapter_init_option, 
                                            adapter_scalar=args.ffn_adapter_scalar,
                                            adapter_layernorm_option=args.ffn_adapter_layernorm_option)

    def forward(self, x, cond):
        #! the dimension of conditions is fixed, no need for mask
        # x_norm = self.layer_norm(x)
        # context, attn = self.context_attn(cond, cond, x_norm, type="context")
        # context = self.feed_forward(x, add_residual)
        
        residual = x
        x = self.layer_norm_1(x)
        x, attn = self.context_attn(cond, cond, x, type="context")
        x = self.dropout(x)
        x = residual + x
        
        residual = x
        x = self.layer_norm_2(x)
        #! bottleneck adapter
        x = self.feed_forward(x, add_residual=False)
        x = self.dropout(x)
        x = residual + x   
        # x = self.feed_forward(self.dropout(x), add_residual=True)
        
        return x


class LayerNorm(nn.Module):
    """
        Layer Normalization class
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class DiTAdaLayerNorm(nn.Module):
    def __init__(self, feature_dim, epsilon=1e-6):
        super(DiTAdaLayerNorm, self).__init__()
        self.epsilon = epsilon
        # self.weight = nn.Parameter(torch.rand(feature_dim, feature_dim * 2))
        self.linear = nn.Linear(feature_dim, feature_dim * 2)
        self.feature_dim = feature_dim
        
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)
        
    def __call__(self, x, condition):
        """
        Args:
            x: shape: (batch_size, sequence_length, feature_dim)
            condition: shape: (batch_size, 1, feature_dim)
            Ps: condition = time_cond_embedding + class_cond_embedding
        return:
            x_layer_norm: shape: (batch_size, sequence_length, feature_dim)
        """
        # affine = condition @ self.weight  # shape: (batch_size, 1, feature_dim * 2)
        affine = self.linear(condition)
        gamma, beta = torch.split(affine, self.feature_dim, dim=-1)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x_layer_norm = gamma * (x - mean) / (std + self.epsilon) + beta
        return x_layer_norm