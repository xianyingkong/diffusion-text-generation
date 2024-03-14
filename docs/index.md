---
layout: default
title: Diffusion-Driven Shakespeare
page_image: shakespeare-picture.jpg
---

## **Introduction to Diffusion Models** {#background}
{:refdef: style="text-align: center;"}
![diffusion_img](/assets/Fixed_Forward_Diffusion_Process.png){: style="text-align: center; display: block; margin: 0 auto;" #large-img}
<p style="font-size: smaller; text-align: center; color: rgb(151, 151, 151);"> <span style="font-size: smaller; color: rgb(151, 151, 151);">Source: NVIDIA. <a href= "https://developer.nvidia.com/blog/improving-diffusion-models-as-an-alternative-to-gans-part-1"> Link to Page</a></span></p>
{: refdef}
<br>
Diffusion models are a type of generative model that simulates the natural data generation process. They are characterized by a forward process that adds random noise to the input sample and a reverse process that iteratively denoises noisy samples to generate high-quality samples that closely approximate its training data distribution.

{:refdef: style="text-align: center;"}
![diffusion_gif](/assets/Diffusion_cropped.gif){: style="text-align: center; display: block; margin: 0 auto;" #small-img}
<p style="font-size: smaller; text-align: center;"><span style="font-size: smaller; color: rgb(151, 151, 151);">Source:  NVIDIA. <a href= "https://developer.nvidia.com/blog/improving-diffusion-models-as-an-alternative-to-gans-part-2/">Link to Page</a></span></p>
{: refdef}
<br>
Diffusion networks such as DALL-E, Midjourney, and most recently SORA by OpenAI, have found success in the domain of images, videos and audios where data is inherently continuous. However, the application of diffusion models in natural language has been limited because **text is inherently discrete in nature.** Despite that, recent research has successfully applied continuous diffusion on text and shown breakthrough in conditional text generation. The success of diffusion models in conditional text generation can be attributed to the **end-to-end learning of feature and embedding space of text.**

<br>

## **Diffusion Models for Conditional Text Generation** {#methodology}
{:refdef: style="text-align: center;"}
![Methodology Image](/assets/diffuseq-process.png){: style="text-align: center; display: block; margin: 0 auto;" #medium-img}
<p style="font-size: smaller; text-align: center;"><span style="font-size: smaller; color: rgb(151, 151, 151);">Source: DiffuSeq. <a href= "https://github.com/Shark-NLP/DiffuSeq/">Link to Page</a></span></p>
{: refdef}
<br>
We follow the architecture of DiffuSeq (Gong et al. 2023) that **performs partial noising on a text sequence**. Given a source sequence (prompt) and a target sequence (response), DiffuSeq learns the feature and embedding spaces by partially corrupting the embedding space of a target sequence in the forward process. In the reverse process, the source sequence serves as guidance for denoising the noisy word embeddings to reconstruct the target sequence.

<br>
More details of the diffusion process are as follow:

<br>
- The input sequence is a combination of the source (x) and target (y) sequence separated by a separator token.
- In the forward process, random Gaussian noise is added to the target sequence’s embeddings iteratively at every timestep while keeping the source sequence unchanged.
- In the reverse process, a BERT-based transformer is used to predict the noise added to the combined sequence at any given timestep as the noisy sequence is being denoised gradually to recover the original sequence z.

<br>

## **Data** {#data}

We use Shakespeare's plays in our task to demonstrate that diffusion models are able to capture complex semantics and structures of text. We arrange conversations in the format: `{src: <text>, trg:<text>}`

<br>
Each sequence is capped at a length of 200 characters. Our train data consists of 32,531 data samples, each is a pair of source-target sequence while our validation data contains 8,123 data samples. Furthermore, we train a **tokenizer on the original Shakespeare's plays consisting of 30,268 vocabularies**.

<br>
Extensive effort is taken to clean the dataset, such as: 

<br>
- Removing non-conversational lines and non-English lines (mostly in Henry V) 
- Segmenting conversations by the last punctuation within the specified length of sequence
- Grouping lines of conversations of each character based on their turns in each act scene. 


<br>

## **Experiments** {#experiments}

Some notable experiments that we have conducted include:

<br>

- **Transfer Learning** 
    - Modern dialogue & Shakespeare plays &rarr; Shakespeare plays
    - Shakespeare plays &rarr; Comedies in Shakespeare plays only

- **Data Engineering**
    - Conversation length
    - With and without shuffling conversations
    - More Shakespeare by adding Shakespeare's sonnets

- **Fine-tuning transformer**
    - Warm up steps
    - Layer-wise learning rate decay (LLRD)
    - Reinitializing the last layer of encoder

- **Hyperparameters**
    - Diffusion steps, dropout rate, batch size, hidden dimension, and weight decay

<br>

**The best-trained model emerged when LLRD and warm-up steps are introduced, strategies not originally present in the DiffuSeq model.**

<br>

Our final best-trained model has 1000 diffusion steps, 128 hidden dimension, 0.1 dropout rate, 0.0 weight decay, batch size of 30 and layer-wise learning rate decay of 0.9 with a starting learning rate of 0.0001 at the bottom-most layer (closest to input). The model is trained on 1 GPU with 30,000 epochs and 500 warm up steps, taking roughly 7-8 hours. The effect of reinitializing the last layer of the encoder transformer is negligible.

<br>

## **Results** {#results}
Our models are able to sample new continuations of Shakespeare lines that match a similar sentiment to the Shakeaspearean line input into the model. Our models **perform better when trained on long input sequences**. Our model samples are also of high diversity, with many different possible lines that follow the given line. These responses are not entirely coherent all the time however, as some generations may not follow the given line well, and thus some human verification may be required.

<br>

Table 1 shows some example responses that the diffusion model generated at each of the
following diffusion steps and the prompt that was used.


<br>

{:refdef: style="text-align: center;"}
![diffusion_img](/assets/model_comparison.JPG){: #very-small-img}
{: refdef}

Figure 1 below visually depicts the embedding space of positive and negative words learned by our model and shows how the model can **clearly cluster similar words belonging to comedies or tragedies**.

<br>

{:refdef: style="text-align: center;"}
![diffusion_img](/assets/tragedies_comedies.png){: #very-small-img}
{: refdef}
<p style="font-size: large; text-align: center;"><strong style="font-size: smaller;">Figure 1: UMAP projection of word embeddings by positive and negative sentiments</strong></p>

<br>

When the model is trained with character names at the start of each conversation in the source-target pairs, we find the our model is **able to capture some level of social network amongst the characters**. Figure 2 below visually depicts the embedding space of character names in selected plays and shows how our model can associate characters of the same play.

<br>

{:refdef: style="text-align: center;"}
![diffusion_img](/assets/social_net.png){: #very-small-img}
{: refdef}
<p style="font-size: large; text-align: center;"><strong style="font-size: smaller;">Figure 2: UMAP projection of character names in selected famous plays</strong></p>


<br>

## **Meet the Team Behind** {#team}

![Team Photo](/assets/team.jpg){: style="text-align: center; display: block; margin: 0 auto;" #small-img }
<p style="font-size: smaller; text-align: center;">Left to Right: Alexander Jensen, (Mentor) Arya Mazumdar, (Winfrey) Xian Ying Kong, Joseph Samuel</p>
<br>

## **Discussion & Conclusion** {#findings}

The experiments involved in training a text generation model based on Shakespearean plays and sonnets revealed several key findings. Initially attempting to train on a combination of plays and sonnets proved ineffective due to the distinct differences between conversational dialogue and poetic structures. Furthermore, despite multiple attempts at transfer learning, the model struggled to grasp modern text structures given our training setup before fine-tuning on Shakespearean data.

<br>

Cleaning the data was crucial for model performance, with the **method of sentence segmentation and conversation length significantly impacting results**. Particularly, limiting conversations to a maximum of two participants and cutting sentences based on the last line break within a sequence length of 200 characters proved beneficial. 

<br>

Moreover, **randomizing the data** was deemed essential, as training on consecutive lines in any specific play led to poor performance due to the strong correlation between data samples within the same play.

<br>

We find that our modified DiffuSeq lacks at upholding sentence coherence. We hypothesize that this could be attributed to the end-to-end training of embedding space (i.e. using a pre-trained embedding model should result in better sentence coherence) and quality of training data used to train these models.

<br> 

Last but not least, it is important to take into account the many different hyperparameters that can be adjusted, and to also consider the tradeoffs that these hyperparameters pose. Diffusion steps is an important hyperparameter, as it gives the model an easier prediction task with a larger amount of timesteps at the cost of slower sampling and the possibility of overfitting due to the smaller noise required to be predicted between each timestep.

<br>
<br>
<br>
