# Auxiliary Classifier Generative Adversarial Network (ACGAN) using ASL Alphabet Datasets

Individual Project for DATA 4380 course at The University of Texas at Arlington.
<br>
This repository holds an attempt to build a ACGAN model to generate conditioned images to a given label using two ASL Alphabet Datasets from Kaggle. 

## Overview  
* The goal was to break down the task of text-to-image translation into dedicated sub-processes (generator and dicriminator model).
* The task, as the model name implies, is to use a classifier network to condition 29 ASL class labels to sign images. We compared the performance of generated images as the model trained per epoch to see if can meet the quality of truth images.

## Summary of Workdone

### Data  
* Data:
  * Datasets from Kaggle: [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) and [ASL Alphabet Test](https://www.kaggle.com/datasets/danrasband/asl-alphabet-test)
  * Type: ASL Alphabet images (200x200 pixel jpegs)
  * Size: 87,029 total images (1.1 GB) from 1st dataset, 870 total images (12.64MB) from second dataset

* Preprocessing / Clean up:
  * Given two datasets, I combined them into one dataframe of images' paths and labels. Index of images where shuffle and normalized to the format needed for model (28,28,1).

### Problem Formulation
* ACGAN Model compose of:
  * Generator Model:
    * Input: Random point from the latent space, and the class label.
    * Output: Generated image.
  * Discriminator Model:
    * Input: Image (real images and generated images)
    * Output: Probability that the provided image is real, probability of the image belonging to each known class.
* Adam Optimizer used 

### Training
* For first attempt, model trained for 15 epochs. This took the model about 40 minutes
* For the second attempt, model trained for 100 epochs. This took model about 6 hours.
* Actual training shown under "Train Model" of ASL_ACGAN notebook.
* Training curves (loss vs epoch steps for generator/discriminator) under LossHistory notebook.
* Generator model outputs a extra value for accuracy metric. This value is ignored when showing training summary perfornace due to only using losses to evaluate training performance.


### Performance Comparison
* Comparison of a images given conditioned label (ex. Letter C)

  Real Image C, Generated C at 15 epochs during first attempt, Generated C at 100 epochs during second attempt (image oder left to right)
   <p float="left">
    <img src="https://github.com/citgua/ACGAN_ASL/blob/main/exampleC_images/exarealc.png" width="200" />
    <img src="https://github.com/citgua/ACGAN_ASL/blob/main/exampleC_images/exgenc.png" width="200" /> 
    <img src="https://github.com/citgua/ACGAN_ASL/blob/main/exampleC_images/exgenc100.png" width="200" />
   </p>
 
* 100 Random Generated Images at 15 epochs -- from first attempt 

![100 Radom Generated Images at 15 epochs](https://github.com/citgua/ACGAN_ASL/blob/main/acgan_logs/generated_plot_9783.png)  
* 100 Radom Generated Images at 100 epochs -- from second attempt

![100 Radom Generated Images at 100 epochs](https://github.com/citgua/ACGAN_ASL/blob/main/asl_logs/generated_plot_98917.png)  

### Conclusions
* Based on losses, the model demonstrates the adivsory game intended in which the generator learns to produce more and more realistic samples as the discriminator learns to get better and better at distinguishing generated data from real data. However, generated images still not meet the quailty of truth images. Longer training still needs to be tested.

### Future Work
* Continue to modify training setup of model for discriminator losses of real and fake images to show under one
* Modify training setup to save model history losses at each epoch instead of each epoch step
* Modify the input dimmesions of ACGAN model so that it can take colored images.
* Train for longer periods to see if improvement of generated images


## How to reproduce results

### Overview of files in repository
* [modules](https://github.com/citgua/ACGAN_ASL/tree/main/modules)
  * acganmodel.py: ACGAN model built of a combination of the generator and discriminator models
  * prepdata.py: Helpers functions used for cleaning and loading data needed in the AC-GAN implementation
  * trainhelpers.py: Training helper functions used to train model

* [acgan_logs](https://github.com/citgua/ACGAN_ASL/tree/main/acgan_logs)
  * Saved model and plot of 100 random generated images at each epoch for first attempt of 15 epochs total.

* [asl_logs](https://github.com/citgua/ACGAN_ASL/tree/main/asl_logs)
  * Saved model and plot of 100 random generated images at each epoch for second attempt of 100 epochs total.

* [asl_acgan_history.pkl](https://github.com/citgua/ACGAN_ASL/blob/main/asl_acgan_history.pkl)
  * Saved history of losses at each epoch step for first attempt of 15 epochs total.

* [asll_acgan_history.pkl](https://github.com/citgua/ACGAN_ASL/blob/main/asll_acgan_history.pkl)
  * Saved history of losses at each epoch step for second attempt of 100 epochs total.

* [ASL_ACGAN.ipynb](https://github.com/citgua/ACGAN_ASL/blob/main/ASL_ACGAN.ipynb)
  * Breakdown of:
    * Background
    * Examine and understand data
    * Loading and normalizing data
    * Building model
    * Training model
    * Evaluating losses
    * Example of generated images conditioned on a label
    * Refernces
  * If notebook takes too long to render, try this link ([nviewer](https://nbviewer.org/github/citgua/ACGAN_ASL/blob/main/ASL_ACGAN.ipynb)).

* [LossHistory.ipynb](https://github.com/citgua/ACGAN_ASL/blob/main/LossHistory.ipynb)
  * This notebook loads and displays the history losses plots for both attempts. Same plots can be found under the "Evaluating Losses" of ASL_ACGAN.ipynb notebook.

### Software Setup
* Packages used in notebook: numpy, pandas, matplotlib, tenserflow, sklearn, scipy cv2, os, PIL, pickle
* Tensorflow-metal PluggableDevice was installed to accelerate training with Metal on Mac GPUs using this [link](https://developer.apple.com/metal/tensorflow-plugin/).

