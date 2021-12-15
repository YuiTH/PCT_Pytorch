ckpt_name=$1
#ckpt_name=checkpoint-30000-0.2412
CUDA_VISIBLE_DEVICES=$2
python test.py --use_bart --tokenizer \
facebook/bart-large-cnn  --test_batch_size 32 --output_dir ./output_scratch/$ckpt_name \
--load_finetune_path ./output_scratch/$ckpt_name/pytorch_model.bin

