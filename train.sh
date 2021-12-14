#CUDA_VISIBLE_DEVICES=0 python train.py --do_train

python train.py --do_train --use_bart --tokenizer \
facebook/bart-large-cnn  --train_batch_size 80 --save_interval_step 400 --num_train_epochs 60 --output_dir ./output_scratch