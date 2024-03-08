from flask import Flask, render_template, request, jsonify, Response
import pickle
from model_arch.tokenizer import load_tokenizer
from model_arch.gaussian_diffusion import get_named_beta_schedule, SpacedDiffusion
from model_arch import transformer
from model_arch.sampling import *
from transformers import set_seed
from constants import *
import yaml, sys, os, io, json
import torch
import numpy as np


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.modules['transformer'] = transformer

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD']=True
app.initialized_env=False

@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/generate", methods=['POST', 'GET'])
def generate_response():
    if request.method == 'POST':
        print(request.form)
        input_text = request.form['input_text']
        write_to_custom_jsonl(input_text, app.data_dir)
        response = {'parse_status': True}
        return jsonify(response)

    elif request.method == 'GET':
        diffusion, model, tokenizer = get_model()

        return Response(generate_samples_progressive(model, diffusion, tokenizer), mimetype='text/event-stream')
    
@app.route("/set_seed", methods=['POST'])
def set_new_seed():
    print(request.form)
    input_seed = int(request.form['input_seed'])
    initialize_env(input_seed)
    response = {'seed_status': True}
    return jsonify(response)
    

def generate_samples_progressive(model, diffusion, tokenizer):
    inter_steps_idx = np.array(app.inter_steps)*app.sampling_step
    final_output = None
    
    for count, sample in sampling_progressive(model, 
                                            diffusion, 
                                            tokenizer, 
                                            data_dir=app.data_dir, 
                                            batch_size=app.sampling_batch_size, 
                                            split='test_custom', 
                                            seq_len=app.sampling_seq_len):
        # check for intermediate steps
#         print(count, "\n")
        if count == -1:
            yield "data:" + json.dumps({"progress": count}) + "\n\n"
            
        inter_response = None
        if len(inter_steps_idx) > 0:
            if count == inter_steps_idx[0]:
                inter_response = clean_output(sample)
                inter_steps_idx = np.delete(inter_steps_idx, 0)
                # print(inter_steps_idx)

        if count == app.sampling_step:
            final_output = clean_output(sample)

#         progress_percentage = int(count/app.sampling_step*100)
#         print(inter_response)
        if inter_response:
            yield "data:" + json.dumps({"progress": count, "inter_response": inter_response, "max_progress": app.sampling_step}) + "\n\n"
        else:
            yield "data:" + json.dumps({"progress": count, "max_progress": app.sampling_step}) + "\n\n"

    yield "data:" + json.dumps({"final_response": final_output}) + "\n\n"

def get_model():
    # Only initializing the model and environment once!
    if not hasattr(app, 'model'):
        # print(hasattr(app, 'sampling_step'))
        print("### Creating model...")
        app.diffusion = SpacedDiffusion(
                    betas=get_named_beta_schedule(app.noise_schedule, app.sampling_step),
                    rescale_timesteps=app.rescale_timesteps,
                    predict_xstart=app.predict_xstart,
                )

        with open(app.best_model_fp, 'rb') as handle:
            app.model = pickle.load(handle)
#             app.model = CPU_Unpickler(handle).load()

        app.tokenizer = load_tokenizer(app.tokenizer_name, app.transformer_name)
    
    return app.diffusion, app.model, app.tokenizer

def initialize_env(seed=None):
    if not app.initialized_env:
        print("### Setting up environment...")
        config = yaml.load(open(config_fp, 'r'), Loader=yaml.SafeLoader)
        seed = seed if seed else config['seed']
        set_seed(seed)
        app.noise_schedule=config['noise_schedule']
        app.predict_xstart=config['predict_xstart']
        app.rescale_timesteps=config['rescale_timesteps']
        app.data_dir=config['data_dir']
        app.transformer_name=config['transformer']
        app.tokenizer_name=config['tokenizer']
        app.sampling_step=config['sampling_step']
        app.inter_steps=list(config['inter_steps'])
        app.sampling_batch_size=config['sampling_batch_size']
        app.sampling_seq_len=config['sampling_seq_len']
        app.best_model_fp=config['best_model_fp']
        app.initialized_env=True
    
    # only reinitializing environment
    if app.initialized_env:
        seed = seed if seed else config['seed']
        set_seed(seed)
        app.diffusion = SpacedDiffusion(
                    betas=get_named_beta_schedule(app.noise_schedule, app.sampling_step),
                    rescale_timesteps=app.rescale_timesteps,
                    predict_xstart=app.predict_xstart,
                )
    
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
    initialize_env()
    app.run(host='0.0.0.0', debug=True)