import copy
import functools
import torch
import numpy as np
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import pickle
import os
import glob

from utils import dist_util
from utils.fp16_util import (
    zero_grad
)
from utils.nn import update_ema
from utils.step_sample import LossAwareSampler, UniformSampler
from datetime import datetime

def clear_dir(directory_path):
    try:
        files = glob.glob(os.path.join(directory_path, '*'))
        for file in files:
            if os.path.isfile(file):
                os.remove(file)
        print("Cleared directory to save new best model.")
    except OSError:
        print("Error occurred while deleting files.")

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        schedule_sampler=None,
        weight_decay=0.0,
        epochs=0,
        eval_data=None,
        eval_interval=-1,
        warm_up_steps=100,
        llrd_rate=0.9
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.eval_data = eval_data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.eval_interval = eval_interval
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.learning_steps = epochs

        self.step = 1

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params

#         self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        self.opt = self.AdamW_LLRD()
        self.scheduler = get_cosine_schedule_with_warmup(self.opt, num_warmup_steps = warm_up_steps, num_training_steps=epochs)
        self.ema_params = [copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))]
        self.min_val_loss = float('inf')
        
    def AdamW_LLRD(self): 
        print("\n\n======== Using Layer-wise Learning Rate Decay with AdamW ========\n\n")
        lr = self.lr
        lr_decay = lr
        decay_rate = 0.75 # decay by 0.9 from top to bottom layers
        
        new_model_params = []
        
        # ==== layers arrangement (from most bottom to most top): 
        # ==== word_embedding -> lm_head -> time_embed -> input_up_proj.0 -> input_up_proj.2 -> 0 to input_transformers.layer.11 -> position_embeddings -> LayerNorm -> output_down_proj.0 -> output_down_proj.2.
        hidden_layers = [f'input_transformers.layer.{i}.' for i in range(12)]
        before_hidden = ['word_embedding', 'lm_head', 'time_embed', 'input_up_proj'] 
        after_hidden = ['position_embeddings', 'LayerNorm', 'output_down_proj']
        
        for c in before_hidden:
            for name, param in self.model.named_parameters():
                if name.startswith(c):
                    new_model_params += [{'params': param, 'lr': lr_decay}]
                    print(f'name: {name}, lr: {lr_decay}') # for checking
                    
        lr_decay = lr_decay/decay_rate
        
        for c in hidden_layers:
            for name, param in self.model.named_parameters():
                if name.startswith(c):
                    new_model_params += [{'params': param, 'lr': lr_decay}]
                    print(f'name: {name}, lr: {lr_decay}') # for checking           
            lr_decay = lr_decay/decay_rate # lr increases as we move from bottom to top layers
            
        for c in after_hidden:
            for name, param in self.model.named_parameters():
                if name.startswith(c):
                    new_model_params += [{'params': param, 'lr': lr_decay}]
                    print(f'name: {name}, lr: {lr_decay}') # for checking
        
        assert len(new_model_params) == len(list(self.model.parameters()))
        
        return torch.optim.AdamW(new_model_params, weight_decay=self.weight_decay)
        
    def run_loop(self):
        print("\n\n======== Training starts now ========\n\n")
        
        while (
            not self.learning_steps or self.step < self.learning_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.eval_data is not None and self.step % self.eval_interval == 0:
                batch_eval, cond_eval = next(self.eval_data)
                self.forward_only(batch_eval, cond_eval)
            self.step += 1

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.optimize_normal()

    def forward_only(self, batch, cond):
        val_losses = []
        with torch.no_grad():
            zero_grad(self.model_params)
            for i in range(0, batch.shape[0], self.microbatch):
                micro = batch[i: i + self.microbatch].to(dist_util.dev())
                micro_cond = {
                    k: v[i: i + self.microbatch].to(dist_util.dev())
                    for k, v in cond.items()
                }
                last_batch = (i + self.microbatch) >= batch.shape[0]
                t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
                # print(micro_cond.keys())
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.model,
                    micro,
                    t,
                    model_kwargs=micro_cond,
                )

                losses = compute_losses()
                loss = (losses["loss"] * weights).mean()
                val_losses.append(loss.detach().cpu())
            print(f'Epoch {self.step}/{self.learning_steps} Validation Loss: {np.mean(val_losses)}')
            
        dt = datetime.now().strftime("%m%d")
        if not os.path.isdir(f'models/{dt}'):
            os.mkdir(f'models/{dt}')
            
        if self.min_val_loss > np.mean(val_losses):
            self.min_val_loss = np.mean(val_losses)
            clear_dir(f'models/{dt}')
            print(f'============>Saving current best model with min_val_loss={self.min_val_loss}<=============')
            pickle.dump(self.model, open(f"models/{dt}/model_best_epoch_{self.step}_min_val_loss_{np.round(self.min_val_loss, 4)}.pkl", 'wb'))

    def forward_backward(self, batch, cond):
        train_losses = []
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            # print(micro_cond.keys())
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            losses = compute_losses()
            
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            loss.backward()
            train_losses.append(loss.detach().cpu())
        print(f'Epoch {self.step}/{self.learning_steps} Training Loss: {np.mean(train_losses)}')

    def optimize_normal(self):
#         self._anneal_lr()
        self.opt.step()
        self.scheduler.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.learning_steps:
            return
        frac_done = self.step / self.learning_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr