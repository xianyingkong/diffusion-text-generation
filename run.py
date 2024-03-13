from model_arch.run_train import create_model_and_diffusion
from utils.step_sample import create_named_schedule_sampler
from utils import dist_util
from model_arch.train import TrainLoop
from utils.data import load_data_text
from model_arch.tokenizer import load_tokenizer, load_model_emb
from model_arch.sampling import sampling
from transformers import set_seed
import yaml

config_fp = './config/config.yaml'

if __name__ == '__main__':
    dist_util.clear_cache()
    config = yaml.load(open(config_fp, 'r'), Loader=yaml.SafeLoader)
    set_seed(config['seed'])
    
    tokenizer = load_tokenizer(config['tokenizer'], config['custom_vocab_fp'])
    model_weight, tokenizer = load_model_emb(config['hidden_dim'], tokenizer)
    vocab_size = tokenizer.vocab_size
    print('Vocab size: ', vocab_size)
    
    data = load_data_text(
        batch_size=config['batch_size'],
        seq_len=config['seq_len'],
        data_dir=config['data_dir'],
        loaded_vocab=tokenizer,
        model_emb=model_weight # use model's weights as init
    )
    
    model, diffusion = create_model_and_diffusion(
                        config['hidden_t_dim'],
                        config['hidden_dim'],
                        vocab_size,
                        config['transformer'],
                        config['use_plm_init'],
                        config['dropout'],
                        config['diffusion_steps'],
                        config['noise_schedule'],
                        config['predict_xstart'],
                        config['rescale_timesteps'],
                    )
    
    model.to(dist_util.dev())
    
    schedule_sampler = create_named_schedule_sampler('uniform', diffusion)

    TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=config['batch_size'],
            microbatch=config['microbatch'],
            lr=config['lr'],
            ema_rate=config['ema_rate'],
            schedule_sampler=schedule_sampler,
            weight_decay=config['weight_decay'],
            epochs=config['epochs'],
#             eval_data=data_valid,
            eval_interval=config['eval_interval']
        ).run_loop()
    
    word_lst_source, word_lst_recover, word_lst_ref, inter_lst_recover = sampling(model, 
                                                               diffusion, 
                                                               tokenizer, 
                                                               data_dir=config['data_dir'], 
                                                               batch_size=config['sampling_batch_size'], 
                                                               split='test', 
                                                               seq_len=config['seq_len'],
                                                               clip_denoised=config['clip_denoised'],
                                                               top_p=config['top_p'],
                                                               clamp_step=config['clamp_step'])
    
    
