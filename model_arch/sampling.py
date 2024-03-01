import numpy as np
import torch
from utils.data import load_data_text
from utils import dist_util
from functools import partial
import json
import os
from datetime import datetime

device = dist_util.dev()

def sampling(
    model, 
    diffusion, 
    tokenizer, 
    device=device, 
    batch_size=16, 
    seq_len=128, 
    data_dir='data/',
    split='test',
    clip_denoised=False, 
    model_kwargs={}, 
    top_p=0, 
    step_gap=1,
    clamp_step=0,
    show_intermediate_results=True,
    inter_steps=[0.25, 0.50, 0.75]):
    
    # ---- putting the model into eval mode ----
    model.eval().requires_grad_(False).to(device)
    
    hidden_dim = model.word_embedding.embedding_dim

    model_emb = torch.nn.Embedding(
            num_embeddings=tokenizer.vocab_size, 
            embedding_dim=hidden_dim, 
            _weight=model.word_embedding.weight.clone().cpu()
        ).eval().requires_grad_(False)
    
    # ---- getting test data ----

    data_test = load_data_text(
            batch_size=batch_size,
            seq_len=seq_len,
            deterministic=True,
            data_dir=data_dir,
            split=split,
            loaded_vocab=tokenizer,
            model_emb=model_emb.cpu(),  # using the same embedding wight with tranining data
            loop=False
        )

    all_test_data = []

    try:
        while True:
            batch, cond = next(data_test)
            # print(batch.shape)
            all_test_data.append(cond)

    except StopIteration:
        print('### End of reading iteration...')

    model_emb.to(device)
    
    # ---- iterating through the test data to generate sequences ----
    iterator = iter(all_test_data)
    word_lst_recover = []
    word_lst_ref = []
    word_lst_source = []

    for cond in iterator:

        input_ids_x = cond.pop('input_ids').to(device)
        x_start = model.get_embeds(input_ids_x)
        input_ids_mask = cond.pop('input_mask')
        input_ids_mask_ori = input_ids_mask

        noise = torch.randn_like(x_start)
        input_ids_mask = torch.broadcast_to(input_ids_mask.unsqueeze(dim=-1), x_start.shape).to(device)
        x_noised = torch.where(input_ids_mask == 0, x_start, noise)

        model_kwargs = {}
        sample_fn = diffusion.p_sample_loop
        sample_shape = (x_start.shape[0], seq_len, hidden_dim)

        samples = sample_fn(
            model,
            sample_shape,
            noise=x_noised,
            clip_denoised=clip_denoised,
            denoised_fn=partial(denoised_fn_round, model_emb),
            model_kwargs=model_kwargs,
            top_p=top_p,
            clamp_step=clamp_step,
            clamp_first=True,
            mask=input_ids_mask,
            x_start=x_start,
            gap=step_gap
        )

#         print(len(samples)) # samples for each step

        sample = samples[-1] # sample at last step

        # print('decoding for seq2seq', )
        # print(sample.shape)

        logits = model.get_logits(sample)  # bsz, seqlen, vocab
        cands = torch.topk(logits, k=1, dim=-1)

        for seq, input_mask in zip(cands.indices, input_ids_mask_ori):
            len_x = seq_len - sum(input_mask).tolist()
            tokens = tokenizer.decode_token(seq[len_x:])
            word_lst_recover.append(tokens)

        for seq, input_mask in zip(input_ids_x, input_ids_mask_ori):
            # tokens = tokenizer.decode_token(seq)
            len_x = seq_len - sum(input_mask).tolist()
            word_lst_source.append(tokenizer.decode_token(seq[:len_x]))
            word_lst_ref.append(tokenizer.decode_token(seq[len_x:]))
            
        inter_lst_recover = []  
        
        if show_intermediate_results:
            total_steps = len(samples)
            all_intermediate = []
            
            for i in inter_steps:
                all_intermediate.append(samples[int(total_steps*i)])
            
            
            
            for inter in all_intermediate:
                logits = model.get_logits(inter)  # bsz, seqlen, vocab
                cands = torch.topk(logits, k=1, dim=-1)
                
                temp_inter_lst_recover = []

                for seq, input_mask in zip(cands.indices, input_ids_mask_ori):
                    len_x = seq_len - sum(input_mask).tolist()
                    tokens = tokenizer.decode_token(seq[len_x:])
                    temp_inter_lst_recover.append(tokens)
                inter_lst_recover.append(temp_inter_lst_recover)
    
    return word_lst_source, word_lst_recover, word_lst_ref, inter_lst_recover

def sampling_progressive(
    model, 
    diffusion, 
    tokenizer, 
    # sampling_steps,
    device=device, 
    batch_size=16, 
    seq_len=128, 
    data_dir='data/',
    split='test',
    clip_denoised=False, 
    model_kwargs={}, 
    top_p=0, 
    step_gap=1,
    clamp_step=0,
    # show_intermediate_results=True,
    # inter_steps=[0.25, 0.50, 0.75]
    ):
    
    # ---- putting the model into eval mode ----
    model.eval().requires_grad_(False).to(device)
    
    hidden_dim = model.word_embedding.embedding_dim

    model_emb = torch.nn.Embedding(
            num_embeddings=tokenizer.vocab_size, 
            embedding_dim=hidden_dim, 
            _weight=model.word_embedding.weight.clone().cpu()
        ).eval().requires_grad_(False)
    
    # ---- getting test data ----

    data_test = load_data_text(
            batch_size=batch_size,
            seq_len=seq_len,
            deterministic=True,
            data_dir=data_dir,
            split=split,
            loaded_vocab=tokenizer,
            model_emb=model_emb.cpu(),  # using the same embedding wight with tranining data
            loop=False
        )

    all_test_data = []

    _, cond = next(data_test)
    all_test_data.append(cond)

    model_emb.to(device)
    
    # ---- iterating through the test data to generate sequences ----
    iterator = iter(all_test_data)

    for cond in iterator:

        input_ids_x = cond.pop('input_ids').to(device)
        x_start = model.get_embeds(input_ids_x)
        input_ids_mask = cond.pop('input_mask')
        input_ids_mask_ori = input_ids_mask

        noise = torch.randn_like(x_start)
        input_ids_mask = torch.broadcast_to(input_ids_mask.unsqueeze(dim=-1), x_start.shape).to(device)
        x_noised = torch.where(input_ids_mask == 0, x_start, noise)

        model_kwargs = {}
        sample_fn = diffusion.p_sample_loop_progressive
        sample_shape = (x_start.shape[0], seq_len, hidden_dim)
        curr_sample=None
        count=0

        for sample in sample_fn(
            model,
            sample_shape,
            noise=x_noised,
            clip_denoised=clip_denoised,
            denoised_fn=partial(denoised_fn_round, model_emb),
            model_kwargs=model_kwargs,
            top_p=top_p,
            clamp_step=clamp_step,
            clamp_first=True,
            mask=input_ids_mask,
            x_start=x_start
        ):
            count += 1

            curr_sample = sample['sample']
            logits = model.get_logits(curr_sample)  # bsz, seqlen, vocab
            cands = torch.topk(logits, k=1, dim=-1)
            
            for seq, input_mask in zip(cands.indices, input_ids_mask_ori):
                len_x = seq_len - sum(input_mask).tolist()
                tokens = tokenizer.decode_token(seq[len_x:])
                    
            yield count, tokens


def get_efficient_knn(model_emb, text_emb):
    emb_norm = (model_emb**2).sum(-1).view(-1, 1) # vocab
    text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1) # d, bsz*seqlen
    arr_norm = (text_emb ** 2).sum(-1).view(-1, 1) # bsz*seqlen, 1
    # print(emb_norm.shape, arr_norm.shape)
    dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(model_emb, text_emb_t) # (vocab, d) x (d, bsz*seqlen)
    dist = torch.clamp(dist, 0.0, np.inf)
    # print(dist.shape)
    topk_out = torch.topk(-dist, k=1, dim=0)
    return topk_out.values, topk_out.indices

def denoised_fn_round(model, text_emb, t):
    # print(text_emb.shape) # bsz, seqlen, dim
    model_emb = model.weight  # input_embs
    # print(t)
    old_shape = text_emb.shape
    old_device = text_emb.device

    if len(text_emb.shape) > 2:
        text_emb = text_emb.reshape(-1, text_emb.size(-1))
    else:
        text_emb = text_emb
    # val, indices = get_knn(model_emb, text_emb.to(model_emb.device), dist=dist)
    val, indices = get_efficient_knn(model_emb, text_emb.to(model_emb.device))
    rounded_tokens = indices[0]
    # print(rounded_tokens.shape)
    new_embeds = model(rounded_tokens).view(old_shape).to(old_device)

    return new_embeds