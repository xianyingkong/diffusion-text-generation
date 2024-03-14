# Diffusion Model For Text Generation - DSC180B B05

## Application Demo
![](https://github.com/xianyingkong/diffusion-text-generation/blob/main-app/diffu-final-app.gif)

The website presentation for this project can be found [here](https://xianyingkong.github.io/diffusion-text-generation/).

## Requirements for Launching Application
Due to the size of the pretrained model (over 300MB), we are unable to host the model on GitHub. For the application to launch, **model path is required** which can be found in `config.yaml`. If you have trained one of our models, we recommend saving the pretrained model under a `models` folder.

## Project Description
This project aims to explore the ability to use diffusion models to generate text. More specifically we implemented the a diffusion model that is comprised of a BERT Transformer and is trained on Shakespeare text. We then measure the clarity and ability of the model to generate text that mimics Shakespeare's.

If you are interested in training a mini-text diffusion model, visit the `main` branch

## Files
The following files are important for launching the application:

- `app.py` - The Flask application file. To launch, run `python app.py`
- `templates` - HTML and JavaScript can be found here
- `static` - CSS can be found here
- `config/config.yaml` - Configuration used to run the `app.py` file. This is mainly used to define parameters for the model.
- `data/resources.jsonl` - A list of prompts in Shakespeare's style. This is for the random prompt generator.
