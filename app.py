from flask import Flask, render_template, request, jsonify
import pickle
from model_arch.tokenizer import load_tokenizer
from model_arch.gaussian_diffusion import get_named_beta_schedule, SpacedDiffusion
from model_arch import transformer
from model_arch.sampling import sampling
from transformers import set_seed
from constants import *
import yaml, sys, os, io, json
import torch


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.modules['transformer'] = transformer

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD']=True

@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/generate", methods=['POST'])
def generate_response():
    diffusion, model, tokenizer = get_diffusion_and_model()

    print(request.form)
    input_text = request.form['input_text']
    write_to_custom_jsonl(input_text, app.data_dir)
    
    _, word_lst_recover, _, inter_lst_recover = sampling(model, 
                                                        diffusion, 
                                                        tokenizer, 
                                                        data_dir=app.data_dir, 
                                                        batch_size=app.sampling_batch_size, 
                                                        split='test_custom', 
                                                        seq_len=app.seq_len, 
                                                        inter_steps=app.inter_steps)


    response = {
            'output_prompt': clean_output(word_lst_recover[0])
        }

    return jsonify(response)

def get_diffusion_and_model():
    # Only initializing the model and environment once!
    if not hasattr(app, 'model'):
        print("### Initializing environment and creating model...")
        app.diffusion, app.model, app.tokenizer = initialize_env_and_model()
    
    return app.diffusion, app.model, app.tokenizer

def initialize_env_and_model():
    config = yaml.load(open(config_fp, 'r'), Loader=yaml.SafeLoader)
    set_seed(config['seed'])
    app.noise_schedule=config['noise_schedule']
    app.predict_xstart=config['predict_xstart']
    app.rescale_timesteps=config['rescale_timesteps']
    app.data_dir=config['data_dir']
    app.transformer_name=config['transformer']
    app.tokenizer_name=config['tokenizer']
    app.sampling_step=config['sampling_step']
    app.inter_steps=list(config['inter_steps'])
    app.sampling_batch_size=config['sampling_batch_size']
    app.seq_len=config['seq_len']
    
    diffusion = SpacedDiffusion(
                    betas=get_named_beta_schedule(app.noise_schedule, app.sampling_step),
                    rescale_timesteps=app.rescale_timesteps,
                    predict_xstart=app.predict_xstart,
                )

    with open(best_model_fp, 'rb') as handle:
        # best_model = pickle.load(handle)
        best_model = CPU_Unpickler(handle).load()

    tokenizer = load_tokenizer(app.tokenizer_name, app.transformer_name)
    
    return diffusion, best_model, tokenizer

def write_to_custom_jsonl(text, folder_dir, filename='test_custom'):
    data = {'src': text, 'trg': ""}
    fn = folder_dir + '/' + filename + '.jsonl'
    if not os.path.exists(folder_dir):
        os.mkdir(folder_dir)
    with open(fn, 'w') as outfile:
        json.dump(data, outfile)
        outfile.write('\n')

def clean_output(text):
    import re

    # Remove [CLS] from the text
    text = re.sub(r'\[CLS\]|\[PAD\]', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# For loading pre-trained model in CPU mode
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


if __name__ == "__main__":
    app.run()