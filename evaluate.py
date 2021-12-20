import logging
import os

import numpy as np
import torch
from torch.utils.data import RandomSampler, DistributedSampler, DataLoader, SequentialSampler
from transformers import BartConfig

from data import Text2Cap, ArgMock
from model.pcl_bart import MyBartForConditionalGeneration
from model.pct_model import PctEncoder


@torch.no_grad()
def evaluate(args, model, logger, device, best_loss=1e6, val_dataloader=None):
    # dataset
    if val_dataloader is None:
        val_dataset = Text2Cap(args, partition='val')

        val_sampler = SequentialSampler(val_dataset)
        val_dataloader = DataLoader(val_dataset, num_workers=24,
                                    batch_size=args.val_batch_size, sampler=val_sampler, drop_last=True)
    # Eval model with dev dataset
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    eval_flag = False
    #
    logger.info("\n***** Running evaluation *****")
    logger.info("  Num examples = %d", len(val_dataloader.dataset))
    logger.info("  Batch size = %d", args.val_batch_size)

    #     # Start Evaling model
    model.eval()
    eval_loss, tokens_num = 0, 0
    for batch in val_dataloader:
        batch = tuple(t.to(device) for t in batch)
        sample_pcl, sample_cap, sample_attn_mask = batch
        with torch.no_grad():
            model_output = model(input_pcl=sample_pcl, labels=sample_cap, decoder_attention_mask=sample_attn_mask)
            loss = model_output.loss
            eval_loss += loss.sum().item()
            # tokens_num += num.sum().item()
    #     # Pring loss of dev dataset
    model.train()
    eval_loss = eval_loss / tokens_num
    result = {'eval_ppl': round(np.exp(eval_loss), 5),
              # 'global_step': global_step + 1,
              # 'train_loss': round(train_loss, 5)
              }
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
    logger.info("  " + "*" * 20)

    # with torch.no_grad():
    #     p = []
    #     model_output = model(input_pcl=sample_pcl)
    #     # TODO: test preds here.
    #     preds = model_output
    #     for pred in preds:
    #         t = pred[0].cpu().numpy()
    #         t = list(t)
    #         if 0 in t:
    #             t = t[:t.index(0)]
    #         text = val_dataloader.dataset.tokenizer.decode(t, clean_up_tokenization_spaces=False)
    #         p.append(text)
    #     with open(os.path.join(args.output_dir, "dev.output"), 'w') as f, open(
    #             os.path.join(args.output_dir, "dev.gold"), 'w') as f1:
    #         for ref, gold in zip(p, val_dataloader.dataset):
    #             predictions.append(str(gold.idx) + '\t' + ref)
    #             f.write(str(gold.idx) + '\t' + ref + '\n')
    #             f1.write(str(gold.idx) + '\t' + gold.target + '\n')
    #
    #     (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "dev.gold"))
    #     dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
    #     logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
    #     logger.info("  " + "*" * 20)

    if eval_loss < best_loss:
        best_loss = eval_loss
        # save last checkpoint
        current_output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        print("Saving model checkpoint to {}".format(current_output_dir))
        torch.save(model_to_save.state_dict(), current_output_dir)

    return val_dataloader, best_loss

if __name__ == '__main__':
    args = ArgMock()
    args.tokenizer = 'facebook/bart-large-cnn'
    args.output_dir = './test/'
    pct_encoder = PctEncoder(args)
    model = MyBartForConditionalGeneration(
        config=BartConfig.from_pretrained('facebook/bart-large-cnn'), encoder=pct_encoder)
    model = model.to("cuda:7")
    logger = logging.getLogger(__name__)
    evaluate(args, model, logger, "cuda:7", best_loss=1e6)
