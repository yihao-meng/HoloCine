from .base_prompter import BasePrompter
from ..models.wan_video_text_encoder import WanTextEncoder
from transformers import AutoTokenizer
import os, torch
import ftfy
import html
import string

import regex as re
import torch.nn.functional as F

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def canonicalize(text, keep_punctuation_exact_string=None):
    text = text.replace('_', ' ')
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(str.maketrans('', '', string.punctuation))
            for part in text.split(keep_punctuation_exact_string))
    else:
        text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


class HuggingfaceTokenizer:

    def __init__(self, name, seq_len=None, clean=None, **kwargs):
        assert clean in (None, 'whitespace', 'lower', 'canonicalize')
        self.name = name
        self.seq_len = seq_len
        self.clean = clean

        # init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs, use_fast=True)
        self.vocab_size = self.tokenizer.vocab_size

    def __call__(self, sequence, **kwargs):
        # --- MODIFICATION START ---

        return_mask = kwargs.pop('return_mask', False)
        return_offsets_mapping = kwargs.pop('return_offsets_mapping', False)

        # arguments
        _kwargs = {'return_tensors': 'pt'}
        if self.seq_len is not None:
            _kwargs.update({
                'padding': 'max_length',
                'truncation': True,
                'max_length': self.seq_len
            })
 
        _kwargs['return_offsets_mapping'] = return_offsets_mapping
        _kwargs.update(**kwargs)

        # tokenization
        if isinstance(sequence, str):
            sequence = [sequence]
        if self.clean:
            sequence = [self._clean(u) for u in sequence]
        

        encoding = self.tokenizer(sequence, **_kwargs)

        output = {'input_ids': encoding.input_ids}
        if return_mask:
            output['attention_mask'] = encoding.attention_mask
        if return_offsets_mapping:

            output['offset_mapping'] = encoding.offset_mapping[0] 
        
        return output
        # --- MODIFICATION END ---

    def _clean(self, text):
        if self.clean == 'whitespace':
            text = whitespace_clean(basic_clean(text))
        elif self.clean == 'lower':
            text = whitespace_clean(basic_clean(text)).lower()
        elif self.clean == 'canonicalize':
            text = canonicalize(basic_clean(text))
        return text


class WanPrompter(BasePrompter):

    def __init__(self, tokenizer_path=None, text_len=512):
        super().__init__()
        self.text_len = text_len
        self.text_encoder = None
        self.fetch_tokenizer(tokenizer_path)
        
    def fetch_tokenizer(self, tokenizer_path=None):
        if tokenizer_path is not None:
            self.tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=self.text_len, clean='whitespace')

    def fetch_models(self, text_encoder: WanTextEncoder = None):
        self.text_encoder = text_encoder


    # def encode_prompt(self, prompt, positive=True, device="cuda"):
#     prompt = self.process_prompt(prompt, positive=positive)
    
#     ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
#     ids = ids.to(device)
#     mask = mask.to(device)
#     seq_lens = mask.gt(0).sum(dim=1).long()
#     prompt_emb = self.text_encoder(ids, mask)
#     for i, v in enumerate(seq_lens):
#         prompt_emb[:, v:] = 0
#     return prompt_emb

    def encode_prompt(self, prompt, positive=True, device="cuda"):


        
        cleaned_prompt = prompt = self.process_prompt(prompt, positive=positive)

 
        char_spans = []
        global_match = re.search(r'\[global caption\]', cleaned_prompt)
        per_shot_match = re.search(r'\[per shot caption\]', cleaned_prompt)
        shot_cut_matches = list(re.finditer(r'\[shot cut\]', cleaned_prompt))

   
        if global_match:
            start_char = global_match.start()  

            end_char = per_shot_match.start() if per_shot_match else len(cleaned_prompt)
            char_spans.append({'id': -1, 'start': start_char, 'end': end_char})

     
        if per_shot_match:
           
            current_start_pos = per_shot_match.start()
            shot_id = 0

  
            for shot_cut_match in shot_cut_matches:
        
                end_char = shot_cut_match.end()
                
             
                char_spans.append({'id': shot_id, 'start': current_start_pos, 'end': end_char})
                
  
                current_start_pos = shot_cut_match.end()
                shot_id += 1

      
            if current_start_pos < len(cleaned_prompt):
                char_spans.append({
                    'id': shot_id, 
                    'start': current_start_pos, 
                    'end': len(cleaned_prompt)
                })

        enc_output = self.tokenizer(
            prompt,
            return_mask=True,
            return_offsets_mapping=True,
            add_special_tokens=True
        )
        ids = enc_output['input_ids']
        mask = enc_output['attention_mask']
        offsets = enc_output['offset_mapping']


        token_shot_ids = torch.full((ids.shape[1],), fill_value=-2, dtype=torch.long)
        for i, (token_start, token_end) in enumerate(offsets):
            if token_start == token_end:
                continue
            for span in char_spans:
                if not (token_end <= span['start'] or token_start >= span['end']):
                    token_shot_ids[i] = span['id']
                    break
        

        positions = {"global": None, "shots": []}
        global_indices = torch.where(token_shot_ids == -1)[0]
        if len(global_indices) > 0:
            positions["global"] = [global_indices.min().item(), global_indices.max().item() + 1]

        max_shot_id = token_shot_ids.max().item()
        for i in range(max_shot_id + 1):
            shot_indices = torch.where(token_shot_ids == i)[0]
            if len(shot_indices) > 0:
                positions["shots"].append([shot_indices.min().item(), shot_indices.max().item() + 1])


        ids = ids.to(device)
        mask = mask.to(device)
        
        if self.text_encoder is None:
            raise ValueError("Text encoder has not been fetched. Call fetch_models() first.")

        prompt_emb = self.text_encoder(ids, mask)
        
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        return prompt_emb, positions
    



    def encode_prompt_separately(self, prompt, positive=True, device="cuda"):

        
        cleaned_prompt = self.process_prompt(prompt, positive=positive)
        
        prompt_parts = []
        

        global_match = re.search(r'\[global caption\]', cleaned_prompt)
        per_shot_match = re.search(r'\[per shot caption\]', cleaned_prompt)
        shot_cut_matches = list(re.finditer(r'\[shot cut\]', cleaned_prompt))

        if global_match is None:
            output = self.tokenizer(cleaned_prompt, return_mask=True, add_special_tokens=True)
            
            ids = output['input_ids'].to(device)
            mask = output['attention_mask'].to(device)
            seq_lens = mask.gt(0).sum(dim=1).long()
            prompt_emb = self.text_encoder(ids, mask)
            for i, v in enumerate(seq_lens): 
                prompt_emb[:, v:] = 0
            return prompt_emb, {"global": None, "shots": []}


        if global_match:
            start_pos = global_match.start()
            end_pos = per_shot_match.start() if per_shot_match else len(cleaned_prompt)
            global_text = cleaned_prompt[start_pos:end_pos].strip()
            if global_text:
                prompt_parts.append({'id': -1, 'text': global_text})


        if per_shot_match:
            current_start_pos = per_shot_match.start()
            shot_id = 0


            for shot_cut_match in shot_cut_matches:
                end_pos = shot_cut_match.start()
                shot_text = cleaned_prompt[current_start_pos:end_pos].strip()
                if shot_text:
                    prompt_parts.append({'id': shot_id, 'text': shot_text})
                
                current_start_pos = shot_cut_match.start()
                shot_id += 1


            last_shot_text = cleaned_prompt[current_start_pos:].strip()
            if last_shot_text:
                prompt_parts.append({'id': shot_id, 'text': last_shot_text})


        
        if self.text_encoder is None:
            raise ValueError("Text encoder has not been fetched. Call fetch_models() first.")

        embeddings_list = []
        positions = {"global": None, "shots": {}}
        current_token_idx = 0

        

        for part in prompt_parts:
            text = part['text']
            shot_id = part['id']


            enc_output = self.tokenizer(
                text,
                return_mask=True,
                add_special_tokens=True,
                return_tensors="pt"
            )
            ids = enc_output['input_ids'].to(device)
            mask = enc_output['attention_mask'].to(device)
            

            part_emb = self.text_encoder(ids, mask) # shape: (1, seq_len, hidden_dim)

    
            seq_len = mask.sum().item()
            
     
            start_idx = current_token_idx
            end_idx = current_token_idx + seq_len
            
            if shot_id == -1: # Global prompt
                positions["global"] = [start_idx, end_idx]
            else: # Per-shot prompt
                positions["shots"][shot_id] = [start_idx, end_idx]

            embeddings_list.append(part_emb[0, :seq_len, :])
    
            current_token_idx += seq_len

   

        if not embeddings_list:
        
            return torch.zeros(1, self.text_len, self.text_encoder.config.hidden_size, device=device), {"global": None, "shots": []}

      
        concatenated_emb = torch.cat(embeddings_list, dim=0) # shape: (total_seq_len, hidden_dim)
        
       
        total_len = concatenated_emb.shape[0]
        if total_len > self.text_len:

            print(f"Warning: Concatenated prompt length ({total_len}) exceeds max length ({self.text_len}). Truncating.")
            concatenated_emb = concatenated_emb[:self.text_len, :]
            total_len = self.text_len

        pad_len = self.text_len - total_len
        

        prompt_emb = F.pad(concatenated_emb, (0, 0, 0, pad_len), 'constant', 0)
        
    
        prompt_emb = prompt_emb.unsqueeze(0)

  
        final_positions = {"global": positions["global"], "shots": []}
        if positions["shots"]:
            
            sorted_shots = sorted(positions["shots"].items())
        
            max_shot_id = sorted_shots[-1][0]
            shot_map = dict(sorted_shots)
            for i in range(max_shot_id + 1):
                final_positions["shots"].append(shot_map.get(i, None)) 

        return prompt_emb, final_positions