#CUDA_VISIBLE_DEVICES=0 python train.py --do_train

python -m torch.distributed.launch --nproc_per_node=8 \
train.py --do_train --use_bart --tokenizer facebook/bart-large-cnn --load_pretrained \
--shapenet_pic_dir ~/data/nrrd_256_filter_div_64_solid/ \
--train_batch_size 12 --save_interval_step 800 --num_train_epochs 30 --output_dir ./output_scratch --lr 5e-5 --max_length 128
#python train.py --do_train --use_bart --tokenizer facebook/bart-large-cnn --shapenet_pic_dir ~/data/nrrd_256_filter_div_64_solid/ \
