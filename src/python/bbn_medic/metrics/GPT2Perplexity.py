import os

import torch
from accelerate.test_utils.testing import get_backend
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from bbn_medic.common.Patient import AtomicPatientExpression
from bbn_medic.common.Prompt import Prompt


class GPT2Perplexity:
    '''
    Based on code from https://huggingface.co/docs/transformers/en/perplexity
    We have found that it works better (returns lower perplexities) than the alternative code
    of https://huggingface.co/spaces/evaluate-metric/perplexity/blob/main/perplexity.py
    '''

    def __init__(self, model_name="openai-community/gpt2-large", device=None):
        if device is None:
            env_device = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
            try:
                if isinstance(device, str):
                    # If CUDA_VISIBLE_DEVICES is a list like "1,2", take the first available device
                    device = int(env_device.split(',')[0])
                else:
                    device = int(env_device)
            except ValueError:
                device = 0  # Fallback to CPU if parsing fails
        self.device = device
        self.model_name = model_name
        self.device = device

        if not self.device:
            self.device, _, _ = get_backend()  # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name).to(device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

    def calculate_perplexity(self, inputs: list[Prompt | AtomicPatientExpression | str], stride=512):
        ppls = []
        for item in inputs:
            if isinstance(item, str):
                text = item
            elif hasattr(item, 'text'):
                text = item.text
            else:
                raise ValueError(f"Item {item} should only be of type of str, or has text attribute.")
            encodings = self.tokenizer(text, return_tensors="pt")
            max_length = self.model.config.n_positions
            seq_len = encodings.input_ids.size(1)

            nll_sum = 0.0
            n_tokens = 0
            prev_end_loc = 0
            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + max_length, seq_len)
                trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
                input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                with torch.no_grad():
                    outputs = self.model(input_ids, labels=target_ids)

                    # loss is calculated using CrossEntropyLoss which averages over valid labels
                    # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                    # to the left by 1.
                    neg_log_likelihood = outputs.loss

                # Accumulate the total negative log-likelihood and the total number of tokens
                num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
                batch_size = target_ids.size(0)
                num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
                nll_sum += neg_log_likelihood * num_loss_tokens
                n_tokens += num_loss_tokens

                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break
            if n_tokens > 0:
                avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
                ppl = torch.exp(avg_nll)
            else:
                ppl = torch.tensor(1.0)
            ppls.append(ppl)
        return ppls
