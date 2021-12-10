import copy
# from typing import Optional, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import ProphetNetPreTrainedModel, ProphetNetEncoder, ProphetNetDecoder, ProphetNetConfig
from transformers.models.prophetnet.modeling_prophetnet import ProphetNetSeq2SeqModelOutput, ProphetNetSeq2SeqLMOutput

from data import ArgMock, Text2Cap
from .pct_model import PctEncoder


class MyProphetNetModel(ProphetNetPreTrainedModel):
    def __init__(self, config, encoder: PctEncoder):
        super().__init__(config)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_encoder_decoder = False
        encoder_config.use_cache = False
        self.encoder = encoder

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.use_cache = False
        self.decoder = ProphetNetDecoder(decoder_config, self.word_embeddings)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value
        # self.encoder.word_embeddings = self.word_embeddings
        self.decoder.word_embeddings = self.word_embeddings

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
            self,
            input_pcl=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        Returns:

        Example::

            >>> from transformers import ProphetNetTokenizer, ProphetNetModel

            >>> tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased')
            >>> model = ProphetNetModel.from_pretrained('microsoft/prophetnet-large-uncased')

            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

            >>> last_hidden_states = outputs.last_hidden_state  # main stream hidden states
            >>> last_hidden_states_ngram = outputs.last_hidden_state_ngram  # predict hidden states
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if encoder_outputs is None:
        #     encoder_outputs = self.encoder(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         head_mask=head_mask,
        #         inputs_embeds=inputs_embeds,
        #         output_attentions=output_attentions,
        #         output_hidden_states=output_hidden_states,
        #         return_dict=return_dict,
        #     )
        encoder_outputs_hidden_states = self.encoder(input_pcl)

        # only encoder_outputs.last_hidden_states available:[1,38,1024]
        # decoder outputs consists of (dec_features, past_key_values, dec_hidden, dec_attn)
        # TODO: encoder_attention_mask should be all ones.
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs_hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs_hidden_states  # TODO: WTF
        return ProphetNetSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            last_hidden_state_ngram=decoder_outputs.last_hidden_state_ngram,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_ngram_hidden_states=decoder_outputs.hidden_states_ngram,
            decoder_attentions=decoder_outputs.attentions,
            decoder_ngram_attentions=decoder_outputs.ngram_attentions,
            cross_attentions=decoder_outputs.cross_attentions,
        )


class MyProphetNetForConditionalGeneration(ProphetNetPreTrainedModel):
    def __init__(self, config: ProphetNetConfig, encoder):
        super().__init__(config)
        config.use_cache = False
        self.prophetnet = MyProphetNetModel(config, encoder)
        self.padding_idx = config.pad_token_id
        self.disable_ngram_loss = config.disable_ngram_loss

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.prophetnet.word_embeddings

    def forward(
            self,
            input_pcl=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``

        Returns:

        Example::

            >>> from transformers import ProphetNetTokenizer, ProphetNetForConditionalGeneration

            >>> tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased')
            >>> model = ProphetNetForConditionalGeneration.from_pretrained('microsoft/prophetnet-large-uncased')

            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

            >>> logits_next_token = outputs.logits  # logits to predict next token as usual
            >>> logits_ngram_next_tokens = outputs.logits_ngram  # logits to predict 2nd, 3rd, ... next tokens
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        outputs = self.prophetnet(
            input_pcl=input_pcl,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        batch_size, sequence_length = (
            decoder_input_ids.shape if decoder_input_ids is not None else decoder_inputs_embeds.shape[:2]
        )

        predicting_streams = outputs[1].view(batch_size, self.config.ngram, sequence_length, -1)
        predict_logits = self.lm_head(predicting_streams)

        logits = predict_logits[:, 0]
        logits_ngram = predict_logits[:, 1:] if self.config.ngram > 1 else None

        # To use .view in loss computation, make sure that logits is contiguous.
        if not logits.is_contiguous():
            logits = logits.contiguous()

        loss = None
        if labels is not None:
            loss = self._compute_loss(predict_logits, labels)

        if not return_dict:
            all_logits = tuple(v for v in [logits, logits_ngram] if v is not None)
            return (loss,) + all_logits + outputs[2:] if loss is not None else all_logits + outputs[2:]
        else:
            return ProphetNetSeq2SeqLMOutput(
                loss=loss,
                logits=logits,
                logits_ngram=logits_ngram,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_ngram_hidden_states=outputs.decoder_ngram_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                decoder_ngram_attentions=outputs.decoder_ngram_attentions,
                cross_attentions=outputs.cross_attentions,
            )

    def _compute_loss(self, logits, labels, ignore_index=-100):
        expend_targets = labels.new_zeros(self.config.ngram, labels.size(0), labels.size(1)).fill_(ignore_index)

        for i in range(self.config.ngram):
            if i > 0 and self.disable_ngram_loss:
                break
            expend_targets[i, :, :] = labels

        lprobs = nn.functional.log_softmax(
            logits.view(-1, logits.size(-1)),
            dim=-1,
            dtype=torch.float32,
        )

        loss = nn.functional.nll_loss(lprobs, expend_targets.view(-1), reduction="mean")

        if self.config.eps > 0.0:
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
            non_masked_tokens = expend_targets.ne(ignore_index).view(-1)
            smooth_loss = smooth_loss[non_masked_tokens]
            smooth_loss = smooth_loss.mean()

            eps_i = self.config.eps / lprobs.size(-1)
            loss = (1.0 - self.config.eps) * loss + eps_i * smooth_loss

        return loss

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs,
    ):
        assert encoder_outputs is not None, "`encoder_outputs` have to be passed for generation."

        if past:
            decoder_input_ids = decoder_input_ids[:, -1:]
        # first step, decoder_cached_states are empty
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    @staticmethod
    # Copied from transformers.models.bart.modeling_bart.BartForConditionalGeneration._reorder_cache
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    def get_encoder(self):
        return self.prophetnet.encoder

    def get_decoder(self):
        return self.prophetnet.decoder


if __name__ == '__main__':
    args = ArgMock()
    train_dataset = Text2Cap(args, partition='train')
    train_loader = DataLoader(train_dataset, num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    sample_pcl, sample_cap, sample_attn_mask = next(iter(train_loader))
    pct_encoder = PctEncoder(args)
    model = MyProphetNetForConditionalGeneration(
        config=ProphetNetConfig.from_pretrained('microsoft/prophetnet-large-uncased'), encoder=pct_encoder)
    sample_pcl, sample_cap, sample_attn_mask = sample_pcl.cuda(), sample_cap.cuda(), sample_attn_mask.cuda()
    model = model.cuda()
    loss = model(input_pcl=sample_pcl, labels=sample_cap, decoder_attention_mask=sample_attn_mask)
    print(model)
