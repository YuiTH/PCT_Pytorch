import argparse
import logging
import os
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from tqdm import tqdm
from transformers import ProphetNetConfig, set_seed, AdamW, get_linear_schedule_with_warmup, BartConfig, \
    BartForConditionalGeneration

from data import ArgMock, Text2Cap
# from evaluate import evaluate
from model.pcl_bart import MyBartForConditionalGeneration
from model.pct_model import PctEncoder
from model.pcl_prophet import MyProphetNetForConditionalGeneration

# logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# parse args
parser = argparse.ArgumentParser(description='Point Cloud Recognition')
parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--max_length', type=int, default=128,
                    help='Max length of caption tokens')
# parser.add_argument('--num_train_epochs', type=int, default=10, metavar='N',
#                     help='number of episode to train ')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 0.001, 0.1 if using sgd)')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help='Number of updates steps to accumulate before performing a backward/update pass.')
parser.add_argument('--save_interval_step', type=int, default=500,
                    help='save model every save_interval_step')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--eval', type=bool, default=False,
                    help='evaluate the model')
parser.add_argument('--num_points', type=int, default=1024,
                    help='num of points to use')
parser.add_argument('--model_path', type=str, default='', metavar='N',
                    help='Pretrained model path')
parser.add_argument('--tokenizer', type=str, default='microsoft/prophetnet-large-uncased',
                    help='tokenizer name for decoder')
parser.add_argument('--text2shape_csv', type=str, default='/mnt/finetune/text2shape/captions.tablechair.csv',
                    help='tokenizer name for decoder')
parser.add_argument('--shapenet_dir', type=str, default='/mnt/finetune/text2shape/shapenetcorev2_hdf5_2048/',
                    help='tokenizer name for decoder')
parser.add_argument('--output_dir', type=str, default='./output',
                    help='tokenizer name for decoder')
parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
parser.add_argument('--load_model_path', type=str, default=None)
parser.add_argument('--load_finetune_path', type=str, default="")
parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
# parser.add_argument("--train_steps", default=-1, type=int, help="")
parser.add_argument("--use_bart", action='store_true', help="Whether to use bart.")
parser.add_argument("--load_pretrained", action='store_true', help="Whether to use bart.")

args = parser.parse_args()


def main(args):
    # sample_pcl, sample_cap, sample_attn_mask = sample_pcl.cuda(), sample_cap.cuda(), sample_attn_mask.cuda()
    # model = model.cuda()
    # loss = model(input_pcl=sample_pcl, labels=sample_cap, decoder_attention_mask=sample_attn_mask)

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    # Set seed
    set_seed(args.seed)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    # budild model

    # Get Model
    pct_encoder = PctEncoder(args)
    if not args.use_bart:
        model = MyProphetNetForConditionalGeneration(
            config=ProphetNetConfig.from_pretrained('microsoft/prophetnet-large-uncased'), encoder=pct_encoder)
    else:
        model = MyBartForConditionalGeneration(
            config=BartConfig.from_pretrained('facebook/bart-large-cnn'), encoder=pct_encoder)
        if args.load_pretrained:
            pre_trained_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
            decoder_state_dict = pre_trained_model.get_decoder().state_dict()
            lm_head_state_dict = pre_trained_model.lm_head.state_dict()
            model.get_decoder().load_state_dict(decoder_state_dict)
            model.lm_head.load_state_dict(lm_head_state_dict)
            del pre_trained_model
            del decoder_state_dict
            del lm_head_state_dict
        elif args.load_finetune_path:
            model.load_state_dict(torch.load(args.load_finetune_path))

    if args.local_rank == 0 or args.local_rank == -1:
        print(model)

    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)

    # Prepare training data loader
    # dataset
    test_dataset = Text2Cap(args, partition='test')

    test_sampler = RandomSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, num_workers=24,
                                 batch_size=args.test_batch_size, sampler=test_sampler, drop_last=False)
    val_dataloader = None

    # num_train_optimization_steps = args.train_steps

    # Prepare optimizer and schedule (linear warmup and decay)

    # Start training
    logger.info("***** Running testing *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.test_batch_size)

    model.eval()
    # main training loop
    results = []
    bar = tqdm(test_dataloader, total=len(test_dataloader))
    for batch in bar:
        batch = tuple(t.to(device) for t in batch)
        sample_pcl, sample_cap, sample_attn_mask = batch
        preds = model.predict(sample_pcl, test_dataset.tokenizer, beam_size=1, max_length=80)
        for idx, pred in enumerate(preds):
            pred_text = test_dataset.tokenizer.decode(pred.cpu(), skip_special_tokens=True)
            gold_text = test_dataset.tokenizer.decode(sample_cap[idx].cpu(), skip_special_tokens=True)
            results.append((pred_text, gold_text))
        # if len(results) >= 40:
        #     break

    def format_prediction(prediction):
        if prediction[0] == 'A':
            prediction = prediction[1:]
        return prediction.strip("<s>").strip("</s>")

    with open(os.path.join(args.output_dir, "test_{}.output".format(str(0))), 'w') as f, open(
            os.path.join(args.output_dir, "test_{}.gold".format(str(0))), 'w') as f1:
        for idx, (pred, gold) in enumerate(results):
            shape_id = str(test_dataset.get_shape_id(idx))
            f.write(shape_id + '\t' + format_prediction(pred) + '\n')
            f1.write(shape_id + '\t' + gold + '\n')


if __name__ == '__main__':
    main(args)
