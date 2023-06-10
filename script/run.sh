#!bin/sh

CUDA_VISIBLE_DEVICES=0 python train_nerv_all.py  --outf 1120  --data_path data/bunny --vid bunny   \
   --conv_type convnext pshuffel --act gelu --norm none  --crop_list 640_1280  \
   --resize_list -1 --loss L2  --enc_strds 5 4 4 2 2 --enc_dim 64_16 \
   --dec_strds 5 4 4 2 2 --ks 0_1_5 --reduce 1.2  \
   --modelsize 1.5  -e 300 --eval_freq 30  --lower_width 12 -b 2 --lr 0.001 \
   --eval_only --weight checkpoints/hnerv-1.5m-e300.pth \
   --quant_model_bit 8 --quant_embed_bit 6 \
   --dump_images --dump_videos