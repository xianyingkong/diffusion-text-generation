# Diffusion Model For Text Generation - DSC180B B05
- The website presentation for this project can be found [here](https://xianyingkong.github.io/diffusion-text-generation/).

- For our web application, go to branch `main-app`.

- The embedding space visualizer can be found [here](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/xianyingkong/diffusion-text-generation/main/projector/template_projector_config.json).

## Project Description
This project aims to explore the ability to use diffusion models to generate text. More specifically we implemented the a diffusion model that is comprised of a BERT Transformer and is trained on Shakespeare text. We then measure the clarity and ability of the model to generate text that mimics Shakespeare's.

If you are interested in training a mini-text diffusion model, the `data/mini-shakespeare` folder contains training and test dataset - both of which are a small subset of the full dataset. The full training process is expected to take around 5 minutes for the mini dataset Shakespeare dataset. 
## Files
We have created files to help facilitate the process from data to trained model:
- `config/config.yaml` - Configuration used to run the `run.py` folder. This is mainly used to define parameters for the model.
- `create_tokenizer.py` - Given a filepath to corpus, creates a bert tokenizer that is able to tokenize each word in the corpus file. 
- `run.py` - Runs the training of the diffusion model. 
## To train the Text Diffusion model
- Clone this repository locally
- [Optional, but recommended] Create a virtual environment: `python -m venv .venv`
- To install the dependencies, run the following command from the root directory of the project: `pip install -r requirements.txt`.
    - **Note that distutils is required, and your Python version must be less than or equal to `3.11`.**
    - If your local desktop is running on CPU only and is having issue with installing all requirements, we recommend commenting out NVIDA requirements as we have noted some issues with downloading CUDA locally.
- Alternative: There is also a [Docker image](https://github.com/users/alexander-jensen/packages/container/package/diffusion-image) that can also replicate the environment.
    - Run `docker pull ghcr.io/alexander-jensen/diffusion-image:latest` to pull the image.
    - `docker run --rm -it diffusion-image` runs the environment in which the next step can be followed.
- To train a model, run `python run.py`. Any hyperparameters should be able to be changed within the `config/config.yaml` file, including number of timesteps, learning rate, epochs, etc.
    - This loads the mini Shakespeare data, trains a mini model with 100 epochs, and saves generated sequence into the `output/` directory.
    - Inside your data directory should be at least the files: train.jsonl test.jsonl. It can also contain the files valid.jsonl and test_custom.jsonl for any custom prompts.
        - These two files should have a format where each line consists of a dictionary of `{"src":"Conditional prompt", "trg":"Target prompt"}`. See `mini_shakespeare/test.jsonl` for examples. 
    - Note that 100 epochs are strictly insufficient for the model to learn and generate meaningful output. However, for convenience of user testing, we set the default as 100 epochs

The specifications of experiment can be found in the config file in the `config/` directory.

## Software & Spec
All dependencies is listed in `requirements.txt` file. For all our experiments, we used 1-4 1080Ti/2080Ti GPUs, where 2-3 GPUs significantly sped up training.

## Reference
This repository is heavily based off of [DiffuSeq's git repository](https://github.com/Shark-NLP/DiffuSeq).
- Gong, Shansan, et al. "Diffuseq: Sequence to sequence text generation with diffusion models." arXiv preprint arXiv:2210.08933 (2022).
- Li, Xiang, et al. "Diffusion-lm improves controllable text generation." Advances in Neural Information Processing Systems 35 (2022): 4328-4343.
