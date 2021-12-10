import argparse
import logging
import os
import shutil

import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from tqdm import tqdm
from transformers import ProphetNetConfig, set_seed, AdamW, get_linear_schedule_with_warmup, BartConfig, \
    BartForConditionalGeneration

from data import ArgMock, Text2Cap
from model.pcl_bart import MyBartForConditionalGeneration
from model.pct_model import PctEncoder
from model.trans import MyProphetNetForConditionalGeneration

# logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# parse args
parser = argparse.ArgumentParser(description='Point Cloud Recognition')
parser.add_argument('--train_batch_size', type=int, default=48, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--max_length', type=int, default=256,
                    help='Max length of caption tokens')
parser.add_argument('--num_train_epochs', type=int, default=10, metavar='N',
                    help='number of episode to train ')
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
parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
parser.add_argument("--train_steps", default=-1, type=int, help="")
parser.add_argument("--use_bart", action='store_true', help="Whether to use bart.")
parser.add_argument("--load_pretrained", action='store_true', help="Whether to use bart.")

args = parser.parse_args()


def main(args):
    # sample_pcl, sample_cap, sample_attn_mask = sample_pcl.cuda(), sample_cap.cuda(), sample_attn_mask.cuda()
    # model = model.cuda()
    # loss = model(input_pcl=sample_pcl, labels=sample_cap, decoder_attention_mask=sample_attn_mask)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
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

    if args.local_rank == 0 or args.local_rank == -1:
        print(model)

    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)
    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        # Prepare training data loader
        # dataset
        train_dataset = Text2Cap(args, partition='train')

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, num_workers=24,
                                      batch_size=args.train_batch_size, sampler=train_sampler, drop_last=True)

        # num_train_optimization_steps = args.train_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total * 0.1),
                                                    num_training_steps=t_total)

        # Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", args.num_train_epochs)

        model.train()
        dev_dataset = {}
        nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss = 0, 0, 0, 0, 0, 1e6
        for epoch in range(args.num_train_epochs):
            bar = tqdm(train_dataloader, total=len(train_dataloader))
            for batch in bar:
                batch = tuple(t.to(device) for t in batch)
                sample_pcl, sample_cap, sample_attn_mask = batch
                model_output = model(input_pcl=sample_pcl, labels=sample_cap, decoder_attention_mask=sample_attn_mask)
                loss = model_output.loss

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                bar.set_description("epoch {} loss {}".format(epoch, train_loss))
                nb_tr_examples += sample_cap.size(0)
                nb_tr_steps += 1
                loss.backward()

                if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                if global_step % args.save_interval_step == 0:
                    # save last checkpoint
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    current_output_dir = os.path.join(args.output_dir,
                                                      'checkpoint-{}-{}'.format(global_step, train_loss))
                    print("Saving model checkpoint to {}".format(current_output_dir))
                    shutil.copytree(last_output_dir, current_output_dir)

            # if args.do_eval:
            # Eval model with dev dataset
            #     tr_loss = 0
            #     nb_tr_examples, nb_tr_steps = 0, 0
            #     eval_flag = False
            #     if 'dev_loss' in dev_dataset:
            #         eval_examples, eval_data = dev_dataset['dev_loss']
            #     else:
            #         eval_examples = read_examples(args.dev_filename)
            #         eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='dev')
            #         all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            #         all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
            #         all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
            #         all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)
            #         eval_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)
            #         dev_dataset['dev_loss'] = eval_examples, eval_data
            #     eval_sampler = SequentialSampler(eval_data)
            #     eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
            #
            #     logger.info("\n***** Running evaluation *****")
            #     logger.info("  Num examples = %d", len(eval_examples))
            #     logger.info("  Batch size = %d", args.eval_batch_size)
            #
            #     # Start Evaling model
            #     model.eval()
            #     eval_loss, tokens_num = 0, 0
            #     for batch in eval_dataloader:
            #         batch = tuple(t.to(device) for t in batch)
            #         source_ids, source_mask, target_ids, target_mask = batch
            #
            #         with torch.no_grad():
            #             _, loss, num = model(source_ids=source_ids, source_mask=source_mask,
            #                                  target_ids=target_ids, target_mask=target_mask)
            #         eval_loss += loss.sum().item()
            #         tokens_num += num.sum().item()
            #     # Pring loss of dev dataset
            #     model.train()
            #     eval_loss = eval_loss / tokens_num
            #     result = {'eval_ppl': round(np.exp(eval_loss), 5),
            #               'global_step': global_step + 1,
            #               'train_loss': round(train_loss, 5)}
            #     for key in sorted(result.keys()):
            #         logger.info("  %s = %s", key, str(result[key]))
            #     logger.info("  " + "*" * 20)
            #
            #     # save last checkpoint
            #     last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
            #     if not os.path.exists(last_output_dir):
            #         os.makedirs(last_output_dir)
            #     model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            #     output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
            #     torch.save(model_to_save.state_dict(), output_model_file)
            #     if eval_loss < best_loss:
            #         logger.info("  Best ppl:%s", round(np.exp(eval_loss), 5))
            #         logger.info("  " + "*" * 20)
            #         best_loss = eval_loss
            #         # Save best checkpoint for best ppl
            #         output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
            #         if not os.path.exists(output_dir):
            #             os.makedirs(output_dir)
            #         model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            #         output_model_file = os.path.join(output_dir, "pytorch_model.bin")
            #         torch.save(model_to_save.state_dict(), output_model_file)
            #
            #     # Calculate bleu
            #     if 'dev_bleu' in dev_dataset:
            #         eval_examples, eval_data = dev_dataset['dev_bleu']
            #     else:
            #         eval_examples = read_examples(args.dev_filename)
            #         eval_examples = random.sample(eval_examples, min(1000, len(eval_examples)))
            #         eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
            #         all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            #         all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
            #         eval_data = TensorDataset(all_source_ids, all_source_mask)
            #         dev_dataset['dev_bleu'] = eval_examples, eval_data
            #
            #     eval_sampler = SequentialSampler(eval_data)
            #     eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
            #
            #     model.eval()
            #     p = []
            #     for batch in eval_dataloader:
            #         batch = tuple(t.to(device) for t in batch)
            #         source_ids, source_mask = batch
            #         with torch.no_grad():
            #             preds = model(source_ids=source_ids, source_mask=source_mask)
            #             for pred in preds:
            #                 t = pred[0].cpu().numpy()
            #                 t = list(t)
            #                 if 0 in t:
            #                     t = t[:t.index(0)]
            #                 text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
            #                 p.append(text)
            #     model.train()
            #     predictions = []
            #     with open(os.path.join(args.output_dir, "dev.output"), 'w') as f, open(
            #             os.path.join(args.output_dir, "dev.gold"), 'w') as f1:
            #         for ref, gold in zip(p, eval_examples):
            #             predictions.append(str(gold.idx) + '\t' + ref)
            #             f.write(str(gold.idx) + '\t' + ref + '\n')
            #             f1.write(str(gold.idx) + '\t' + gold.target + '\n')
            #
            #     (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "dev.gold"))
            #     dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
            #     logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
            #     logger.info("  " + "*" * 20)
            #     if dev_bleu > best_bleu:
            #         logger.info("  Best bleu:%s", dev_bleu)
            #         logger.info("  " + "*" * 20)
            #         best_bleu = dev_bleu
            #         # Save best checkpoint for best bleu
            #         output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
            #         if not os.path.exists(output_dir):
            #             os.makedirs(output_dir)
            #         model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            #         output_model_file = os.path.join(output_dir, "pytorch_model.bin")
            #         torch.save(model_to_save.state_dict(), output_model_file)
            # else:
            #     if epoch % args.save_interval == 0:
            #         output_model_file = os.path.join(args.output_dir, f"checkpoint{epoch}.bin")
            #         model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            #         torch.save(model_to_save.state_dict(), output_model_file)


if __name__ == '__main__':
    main(args)
