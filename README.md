# Diffusion Model For Text Generation
23-24 DSC Capstone Project Description
This project aims to explore the ability to use diffusion models to generate text. More specifically we implemented the a Diffusion model that is comprised of the BERT Transformer and is trained on Skaepeare text utilzing the diffusion process. We then measure teh clarity and ability of the mdoel to generate text that mimics shakespeare.


effect of non-Gaussian distributions, combined with other loss functions on the outcome of Denoising Diffusion Probabilistic Model (DDPM). More specifically, we implement noise sampled from Laplace distribution and Salt & Pepper (S&P) noise, together with L1 loss, Laplace negative log likelihood, and Sigmoid loss. The datasets used in this project are MNIST and CIFAR10 from Pytorch.

If you are interested in training a mini Text diffusion model, the data folder contains Json dataset (subset of the full dataset) - 20 data points of lines used by Shakesperae and their target outputs. The full training process for the mini dataset is expected to take around 5-10 minutes on either the full MNIST training data or CIFAR10.

To train the mini DDPM model
Clone this repository locally
[Optional, but recommended] Create a virtual environment: python -m venv [venv_name]
To install the dependencies, run the following command from the root directory of the project: pip install -r requirements.txt
To train a model, run python run.py build
This loads the MNIST test data, trains a mini model with 5 epochs, and saves the model and network into the output/ directory
Software & Spec
All dependencies is listed in requirements.txt file. For all our experiments, we used 1 GPU, 4 CPUs and 80GB RAM.

Reference
Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. arXiv preprint arXiv:2006.11239, 2020
Eliya Nachmani, Robin San Roman, and Lior Wolf. Non Gaussian Denoising Diffusion Models. arXiv preprint arXiv:2106.07582, 2021
Tien Chen. On the Importance of Noise Scheduling for Diffusion Models. arXiv:2301.10972 [cs.CV]
