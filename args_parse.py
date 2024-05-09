from transformers import HfArgumentParser
from models.adapter_configuration import AdapterTrainingArguments

def args_parser(params=None):
    parser = HfArgumentParser(AdapterTrainingArguments)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--encoder', type=str, default="ReactionPCL", choices=["ReactionPCL", "Rxnfp"], help='the encoder to extract reaction reps')
    parser.add_argument('--decoder', type=str, default="MLP", help='the type of reaction classifier')
    parser.add_argument('--decoder_hidden_size', type=int, nargs='+', default=[]) 
    parser.add_argument('--num_class', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')  
    parser.add_argument('--encoder_lr', type=float, default=1e-3)  
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_runs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0, help="Seed for splitting dataset.")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num_train_samples', type=int, default=8) 
    parser.add_argument('--batch_size_trn', type=int, default=32, help='raw train batch size')
    parser.add_argument('--batch_size_val', type=int, default=32, help='val/test batch size')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers for dataset loading')
    parser.add_argument('--data_dir', type=str, default='./data/buchward/az/30_random_splits')
    parser.add_argument('--exp_dir', type=str, default='./result/buchward/az/30_random_splits')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint model file')

    parser.add_argument('--encoder_num_layers', type=int, default=4, help='number of layers of transformer')
    parser.add_argument('--decoder_num_layers', type=int, default=4, help='number of layers of transformer')
    parser.add_argument('--condition_attn_layers', type=int, default=0, help='number of layers of transformer decoder with cond_attn')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model representation')
    parser.add_argument('--heads', type=int, default=8, help='number of heads of multi-head attention')
    parser.add_argument('--d_ff', type=int, default=2048, help='')
    parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
    parser.add_argument('--known_class', action="store_true", default=False)
    parser.add_argument('--shared_vocab', action="store_true", default=False)
    parser.add_argument('--shared_encoder', action="store_true", default=False)
    parser.add_argument('--mode', type=str, default='forward', choices=["forward", "backward"])

    # finetuning
    parser.add_argument('--fix_encoder', action="store_true", default=False)
    parser.add_argument('--ft_mode', type=str, default='petl', choices=["none", "petl", "full"]) 
    parser.add_argument('--decoder_type', type=str, default='linear', choices=["linear", "non-linear"])
    parser.add_argument('--ffn_mode', type=str, default='none', choices=["none", "adapter", "prefix"])
    parser.add_argument('--ffn_option', type=str, default="none", choices=["none", "parallel", "sequential"])
    parser.add_argument('--ffn_bn', type=int, default=64)
    parser.add_argument('--ffn_adapter_scalar', type=str, default="1")  # learnable or fixed 
    parser.add_argument('--ffn_adapter_init_option', type=str, default="lora", choices=["bert", "lora"])
    parser.add_argument('--ffn_adapter_layernorm_option', type=str, default="none", choices=["none", "in", "out"]) 
    parser.add_argument('--attn_mode', type=str, default='none', choices=["none", "prefix", "lora", "adapter"])
    parser.add_argument('--attn_bn', type=int, default=32) # bottleneck dim
    parser.add_argument('--attn_dim', type=int, default=32) 
    parser.add_argument('--update_layernorm', action="store_true", default=False) 

    # prompt    
    parser.add_argument('--prompt', action="store_true", default=False)
    parser.add_argument('--input_prompt_attn', action="store_true", default=False)
    parser.add_argument('--proto_hierarchy', type=int, default=3, help='the number of hierarchy')
    parser.add_argument('--proto_path', type=str, default='./result/ecreact')
    parser.add_argument('--proto_version', type=str, default="top", choices=["bottom", "middle", "top", "hierarchy", "namerxn"])
    parser.add_argument('--freeze_proto', action="store_true", default=False)

    # imbalanced regression
    parser.add_argument('--num_bins', type=int, default=5, help='')
    parser.add_argument('--criterion', type=str, default="mse", choices=["mse", "weighted_mse", "weighted_focal_mse", "focal_mse", "bmc"])

    # downstream tasks
    parser.add_argument('--do_train', action="store_true", default=False)
    parser.add_argument('--do_pretrain', action="store_true", default=False)
    parser.add_argument('--pretrain_epochs', type=int, default=10, help='')

    args = parser.parse_args(params)
    
    return args