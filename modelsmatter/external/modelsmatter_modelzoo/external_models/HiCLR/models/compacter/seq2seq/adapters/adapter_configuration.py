"""Implements the adapters and other parameter-efficient finetuning methods' configurations."""

from collections import OrderedDict
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class AdapterConfig(object):
    """Implements the adapter configuration proposed by Houlsby et. al, 2019
    in https://arxiv.org/abs/1902.00751.
    We additionally pass all the configuration of parameter-efficient finetuning
    methods with this config."""
    add_layer_norm_before_adapter: bool = False
    add_layer_norm_after_adapter: bool = True
    non_linearity: str = "swish"
    task_reduction_factor: int = 16
    add_adapter_in_feed_forward = True
    add_adapter_in_self_attention = True
    hidden_dim = 128
    task_adapter_layers_encoder = None
    task_adapter_layers_decoder = None
    task_adapter_in_decoder = True
    intrinsic_dim = 100
    normalize_intrinsic_projections = False
    # This can be either random, or fastfood.
    intrinsic_projection = "random"

    # Hypercomplex adapters parameters 
    hypercomplex_adapters = False
    hypercomplex_division = 8
    learn_phm = True
    hypercomplex_nonlinearity="glorot-uniform"
    shared_phm_rule = False 
    factorized_phm = False 
    shared_W_phm = False
    factorized_phm_rule = False 
    phm_c_init = "normal"
    phm_rank = 1
    phm_init_range=0.01

    # prefix-tuning parameters.
    prefix_dim = 100
    init_prefix_from_vocab = False 
    kronecker_prod = False  

    # BitFit configuration.
    bitfit = False

    # Low-rank adapters.
    low_rank_adapters = False
    low_rank_w_init = "glorot-uniform"
    low_rank_rank = 1


ADAPTER_CONFIG_MAPPING = OrderedDict(
    [("adapter", AdapterConfig)])


class AutoAdapterConfig(nn.Module):
    """Generic Adapter config class to instantiate different adapter configs."""

    @classmethod
    def get(cls, config_name: str):
        if config_name in ADAPTER_CONFIG_MAPPING:
            return ADAPTER_CONFIG_MAPPING[config_name]()
        raise ValueError(
            "Unrecognized adapter config type identifier: {}. Should contain one of {}"
                .format(config_name, ", ".join(ADAPTER_CONFIG_MAPPING.keys())))

@dataclass
class AdapterTrainingArguments:
    """Defines the adapters parameters."""
    train_task_adapters: Optional[bool] = field(default=False,
                                                metadata={"help": "If set, adds task adapters in the model."})
    adapter_config_name: Optional[str] = field(
        default="adapter", metadata={"help": "config name for the adapter layers, should be selected "
        f"in {sorted(ADAPTER_CONFIG_MAPPING.keys())}."}
    )
    add_layer_norm_before_adapter: Optional[bool] = field(default=False, metadata={
        "help": "whether to have layer-norm before adapter."})
    add_layer_norm_after_adapter: Optional[bool] = field(default=True,
        metadata={"help": "whether to have layer-norm after adapter."})
    hidden_dim: Optional[int] = field(default=128, metadata={"help": "defines the default hidden dimension for "
        "adapter layers."})
    task_reduction_factor: Optional[int] = field(default=16, metadata={"help": "defines the default reduction factor for "
        "adapter layers."})
    non_linearity: Optional[str] = field(default="swish", metadata={"help": "Defines nonlinearity for adapter layers."})
    unfreeze_lm_head: bool = field(default=False, metadata={"help": "If set unfreeze the last linear layer."})
    unfreeze_layer_norms: bool = field(default=False, metadata={"help": "If set, unfreezes the layer norms."})
    task_adapter_layers_encoder: Optional[List[int]] = field(default=None, metadata={"help": "Defines the layers id"
                                                                                      "in which task adapters is"
                                                                                      "added in the encoder."})
    task_adapter_layers_decoder: Optional[List[int]] = field(default=None, metadata={"help": "Defines the layers id"
                                                                                      "in which task adapters is"
                                                                                      "added in the decoder."})
    task_adapter_in_decoder: Optional[bool] = field(default=True, metadata={"help": "If set to false, do not include"
                                                                                    "task adapters in the decoder."})
    hypercomplex_adapters: Optional[bool] = field(default=False, metadata={"help": "If set, uses the hypercomplex layers"
                                                                                "for adapters."})
    hypercomplex_division: Optional[int] = field(default=8, metadata={"help": "Defines the number to divide the dimensions"
                                                                              "of the linear layer by it."})
    intrinsic_model: Optional[bool] = field(default=False, metadata={"help": "If set, computes all parameters of the "
                                                                             "model with an intrinsic vector."})
    intrinsic_said: Optional[bool] = field(default=False, metadata={"help": "If set, computes the SAID version of the"
                                                                            "model with intrinsic vector."})
    intrinsic_dim: Optional[int] = field(default=100, metadata={"help": "Defines the intrinsic dimensionality."})
    normalize_intrinsic_projections: Optional[bool] = field(default=False, metadata={"help": "If set, normalizes "
        "the intrinsic projection matrices."})
    intrinsic_projection: Optional[str] = field(default="fastfood", metadata={"help": "Defines the type of projection"
        "for intrinsic adapters, it can be random or fastfood."})
    learn_phm: Optional[bool] = field(default=True, metadata={"help": "If set, learns the phm rules in Hypercomplex adapters."})
    normalize_phm_weight: Optional[bool] = field(default=False, metadata={"help": "Weather to normalize the weights of"
                                                                                  "the PHM layer."})
    intrinsic_layer_norms: Optional[bool] = field(default=False, metadata={"help": "If selected, then in case of unfreezing"
        " layernorms for intrinsic_adapters case, it also adds the layernorms parameters inside the parameters given for"
        " the intrinsic projection, and if this is not set, those parameters are not projected with intrinsic vector."})
    hypercomplex_nonlinearity: Optional[str] = field(default="glorot-uniform", metadata={"help": "Defines the nonlinearity for the"
        " hypercomplex adapter layers."})
    shared_phm_rule: Optional[bool] = field(default=False, metadata={"help": "If set, uses a shared phm rules for all"
        " hypercomplex adapter layers."})
    factorized_phm: Optional[bool] = field(default=False, metadata={"help": "If set, it factorizes the weights for the W in"
        " hypercomplex adapters."})
    shared_W_phm: Optional[bool] = field(default=False, metadata={"help": "If set, shares the W in phm adapter layers between all adapters."})
    factorized_phm_rule: Optional[bool] = field(default=False, metadata={"help": "If set, it factorizes the shared weights for the W in"
        " hypercomplex adapters."})
    phm_c_init: Optional[str] = field(default="normal", metadata={"help": "Initialization for the phm rules."})
    phm_rank: Optional[int] = field(default=1, metadata={"help":"sets the rank for the phm decomposition."})
    phm_init_range: Optional[float] = field(default=0.01, metadata={"help": "defines the phm init range."})
    add_adapter_in_feed_forward: Optional[bool] = field(default=True, metadata={"help": "If set, adds adapters in the feedforward."})
    add_adapter_in_self_attention: Optional[bool] = field(default=True, metadata={"help": "If set, adds adapters in the selfattention"})
    prefix_tuning: Optional[bool] = field(default=False, metadata={"help": "If set, uses prefix tuning."})
    prefix_dim: Optional[int] = field(default=100, metadata={"help": "Specifies the prefix embedding dimension."})
    init_prefix_from_vocab: Optional[bool] = field(default=False, metadata={"help": "Initialize prefix from the tokens of pretrained t5-base model."})
    kronecker_prod: Optional[bool] = field(default=False, metadata={"help": "If set, compute the kronecker using another version."})
    bitfit: Optional[bool] = field(default=False, metadata={"help": "If set, we train the bitfit model."})
    freeze_bitfit_lm_head: Optional[bool] = field(default=False, metadata={"help": "If set, freezes the classifier in bitfit."})
    freeze_bitfit_lm_head_all: Optional[bool] = field(default=False, metadata={"help": "If set, freezes the classifier in bitfit."})
    # Low-rank adapters.
    low_rank_adapters: Optional[bool] = field(default=False, metadata={"help": "If set, uses the low-rank adapters."})
    low_rank_w_init: Optional[str] = field(default="glorot-uniform", metadata={"help": "Defines the initialization for low-rank adapters."})
    low_rank_rank: Optional[int] = field(default=1, metadata={"help": "Defines the rank of low-rank adapters."})
