# Diffusion Model For Text Generation - DSC180B B05
The website presentation for this project can be found [here](https://xianyingkong.github.io/diffusion-text-generation/).
## Project Description
This project aims to explore the ability to use diffusion models to generate text. More specifically we implemented the a diffusion model that is comprised of a BERT Transformer and is trained on Shakespeare text. We then measure the clarity and ability of the model to generate text that mimics Shakespeare's.

If you are interested in training a mini-text diffusion model, the `data/mini-shakespeare` folder contains training and test dataset - both of which are a small subset of the full dataset. The full training process is expected to take around 5 minutes for the mini dataset Shakespeare dataset. 

## To train the Text Diffusion model
- Clone this repository locally
- [Optional, but recommended] Create a virtual environment: `python -m venv .venv`
- To install the dependencies, run the following command from the root directory of the project: `pip install -r requirements.txt`
- Alternative: There is also a [Docker image](https://github.com/users/alexander-jensen/packages/container/package/diffusion-image) that can also replicate the environment.
    - Run `docker pull ghcr.io/alexander-jensen/diffusion-image:latest` to pull the image.
    - `docker run --rm -it diffusion-image` runs the environment in which the next step can be followed.
- To train a model, run `python run.py`
    - This loads the mini Shakespeare data, trains a mini model with 100 epochs, and saves generated sequence into the `output/` directory
    - Note that 100 epochs are strictly insufficient for the model to learn and generate meaningful output. However, for convenience of user testing, we set the default as 100 epochs

The specifications of experiment can be found in the config file in the `config/` directory

## Software & Spec
All dependencies is listed in `requirements.txt` file. For all our experiments, we used 1 GPU

## Reference
- Gong, Shansan, et al. "Diffuseq: Sequence to sequence text generation with diffusion models." arXiv preprint arXiv:2210.08933 (2022).
- Li, Xiang, et al. "Diffusion-lm improves controllable text generation." Advances in Neural Information Processing Systems 35 (2022): 4328-4343.
