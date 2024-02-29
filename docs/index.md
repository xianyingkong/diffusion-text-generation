---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
title: Diffusion-Driven Shakespeare
---
# Introduction
Diffusion-Driven Shakespeare is a project that aims to further explore the capabilities of diffusion networks in a text-based domain. Diffusion networks such as DALLE, Midjourney, and most recently SORA, have been used to create high quality samples in image and video domains, where there is an inherent locality to the data structure, such as related pixels being next to each other. However, text does not have an inherent locality similar to images, and must consider more factors, such as context (our input sequence). In this project, we use the DiffuSeq model architecture trained on Shakespeare plays to assess whether diffusion models are capable of sampling sentences that have similar style to Shakespeare analogous to high quality images sampled in other models. For our results, we found that the Shakespeare model does have comprehension of the context, although the length of the generated texts are fairly short.

Our working model can be found here (currently in development). Our github repository can be found [here](https://github.com/xianyingkong/diffusion-text-generation), and our report can be found here (not published yet).
# Data
For our data, we used a Shakespeare dataset based off of all of Shakespeare's plays. For every line, we create the source sequence containing the line, followed by the target sequence being the next line that follows this line. Upon further development, we cleaned the data to remove errors in encoding (incorrectly formatted approstrophes, colons, etc), as well as removing lines that were in a different language, such as certain sections in Henry V. 

# Basic Methodology
The basic methodology involves preprocessing a dataset comprising Shakespeare's plays to prepare it for training the diffusion model. The dataset undergoes cleaning procedures to remove inconsistencies and is segmented to ensure that each data point represents a coherent sequence of text. This formatted dataset serves as the foundation for training the diffusion model to generate text that adheres to the style and context of Shakespearean language.

The training process involves implementing a diffusion model architecture based on Gong et al.'s (2023) framework, utilizing a BERT-based transformer alongside a Byte Pair Encoding (BPE) tokenizer. This architecture enables efficient computation and transfer learning, reducing computation costs and training time. Additionally, a customized loss function is employed, incorporating components for denoising, regularization, and embedding learning to optimize the training process.

# Experiments
The experiments involved in training a text generation model based on Shakespearean plays and sonnets revealed several key findings. Initially attempting to train on a combination of plays and sonnets proved ineffective due to the distinct differences between conversational dialogue and poetic structures. However, the best-performing model emerged when layer-wise learning rate decay and warm-up steps were introduced, strategies not originally present in the DiffuSeq model. Despite attempts at transfer learning, the model struggled to grasp modern text structures before fine-tuning on Shakespearean data.

Cleaning the data was crucial for model performance, with the method of sentence segmentation significantly impacting results. Particularly, limiting conversations to a maximum of two participants and cutting sentences based on the last line break within a sequence length of 128 characters proved beneficial. Moreover, randomizing the data was deemed essential, as training on consecutive plays led to poor performance due to the strong correlation between consecutive data points within the same play.

After numerous experiments, specific hyperparameters were settled upon, including a dropout rate of 0.1, no weight decay, layer-wise learning rate decay set at 0.99, 500 warm-up steps, and 30,000 epochs, with 2000 diffusion steps per training session, taking approximately three hours to complete. Notably, incorporating character names before dialogue improved the model's ability to capture social networks within the text, although it resulted in all generated responses adhering to a "name: response" format, such as "Juliet: life's good."


# Results
