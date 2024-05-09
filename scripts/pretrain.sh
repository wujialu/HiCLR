source activate hiclr

type=uspto_1K_TPL_backward
tau=0.1
contrastive_coeff=0.2
token_coeff=0.8
num_layer=4
d_ff=2048
contrastive_loss_type=hmlc
hmlc_loss_type=hmce
supcon_level=rxn
layer_penalty=pow2
batch_size_token=32768
task=model_size_scaling/tau${tau}_coeff${contrastive_coeff}_${contrastive_loss_type}_${num_layer}layer_d_ff_${d_ff}
sp_gamma_min=10
sp_gamma_max=50 # for self-paced learning

python -u pretrain_hierar.py \
	--enable_tensorboard --shared_vocab \
	--data_dir ./data/${type} --exp_dir ./result/${type}/${task}/ \
	--max_pretrain_step 100000 \
	--report_per_step 100 --save_per_step 2000 --val_per_step 100 \
	--token_loss_coeff ${token_coeff} \
	--contrast_loss_coeff ${contrastive_coeff} \
	--contrastive_loss_type ${contrastive_loss_type} \
	--supervised_scale reaction \
	--hmlc_loss_type ${hmlc_loss_type} \
	--supcon_level ${supcon_level} \
	--sp_gamma_min ${sp_gamma_min} --sp_gamma_max ${sp_gamma_max} \
	--tau ${tau}  \
	--mode backward \
	--encoder_num_layers ${num_layer} --decoder_num_layers ${num_layer} --d_ff ${d_ff} \
	--batch_size_trn 4 --batch_size_token ${batch_size_token} \
	--lr 1e-3 \
	--reaction_rep_mode concat \
	--proj_mode non-linear \
	--hierar_sampling \
