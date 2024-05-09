source activate hiclr

num_layer=4
d_ff=2048

python -u finetune_retro.py \
	--augment \
	--enable_tensorboard --shared_vocab \
	--data_dir ./data/uspto_50k_typed --exp_dir ./result/uspto_50k_untyped \
	--batch_size_trn 2 --batch_size_token 4096 --batch_size_val 8 \
	--max_train_step 50000 \
	--disp_interval 1000 \
	--learning_rate 1. --dropout 0.1 \
	--encoder_num_layers ${num_layer} --decoder_num_layers ${num_layer} --d_ff ${d_ff} \
	--checkpoint ./checkpoint/supcon_hierar/model_pretrain_best_mAP.pt \
