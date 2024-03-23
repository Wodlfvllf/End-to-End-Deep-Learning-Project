# End To End Deep Learning Project

This repository contains solutions for various particle physics tasks using deep learning models. Below are the details of each task along with their respective implementations.

## Common Task 1: Electron/Photon Classification

### Dataset Description
- Two types of particles: electrons and photons
- Images are represented as 32x32 matrices with two channels: hit energy and time

<p align="center">
  <img src="https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Common%20Task%201/ResNetCBAM.png" width="350" title="hover text">
</p>

### Solution
- Implemented a ResNet-15 model for classification
- I used ensembling of two Resnet-15 models, one for learning pixel distribution of hit energy channel and other of time channel.
- Trained the model using K-Fold Cross Validation. I trained the model for 5 folds.
- Ensured no overfitting on the test dataset

#### Notebook: [Common Task 1.ipynb](https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Common%20Task%201/Common%20Task1%20Approach%201/common-task-1.ipynb)
- Includes data loading, model definition, training, evaluation, and model weights

## Common Task 2: Deep Learning based Quark-Gluon Classification

### Dataset Description
- Two classes of particles: quarks and gluons
- Images are represented as 125x125 matrices in three-channel images

<p align="center">
  <img src="https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Common%20task2/1_B_ZaaaBg2njhp8SThjCufA.png" width="350" title="hover text">
</p>


<p align="center">
  <img src="https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Common%20task2/VisionTransfomer.png" width="350" title="hover text">
</p>

### Solutions
1. VGG with 12 layers
2. Vision Transformer

### Implementation
- Implemented both models for classification
- Trained the model using K-Fold Cross Validation(5 Folds is performed).
- Ensured no overfitting on the test dataset

#### Notebooks:
- [Quark_Gluon_Classification_VGG.ipynb](https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Common%20task2/vgg-task2.ipynb)
- [Quark_Gluon_Classification_Custom.ipynb](https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Common%20task2/VisionTransformer%20-%20Task%202.ipynb)
- Include data loading, model definition, training, evaluation, and model weights

## Specific Task 3d: Masked Auto-Encoder for Efficient End-to-End Particle Reconstruction and Compression

### Tasks
1. Train a lightweight ViT using the Masked Auto-Encoder (MAE) training scheme on the unlabelled dataset.
2. Compare reconstruction results using MAE on both training and testing datasets.
3. Fine-tune the model on a lower learning rate on the provided labelled dataset and compare results with a model trained from scratch.

### Implementation
- Trained a lightweight ViT using MAE on unlabelled dataset
- Compared reconstruction results on training and testing datasets
- Fine-tuned the model on a lower learning rate using the labelled dataset
- Compared results with a model trained from scratch
- Ensured no overfitting on the test dataset

###Image Reconstruction

### Comparison of With and Without Pretrained Vision Transformer Model
                          | Model               | Accuracy |
                          |---------------------|----------|
                          | With Pretrained     | 0.8548   |
                          | Without Pretrained  | 0.0951   |

            
#### Notebooks: 
- [MAE_Particle_Reconstruction.ipynb](https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Specific%20Task%203d%20-%20Masked_Autoencoder/Masked_Autoencoder/Masked%20Autoencoder.ipynb)
- [linear-probing-Pretraining.ipynb](https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Specific%20Task%203d%20-%20Masked_Autoencoder/Linear%20Probing%20MAE/linear-probing-Pretraining.ipynb)
- [linear-probing-without Pretraining.ipynb](https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Specific%20Task%203d%20-%20Masked_Autoencoder/Linear%20Probing%20MAE/linear-probing-without%20Pretraining.ipynb)
- Includes data loading, model training (pre-training and fine-tuning), evaluation, and model weights

## Dependencies
- Python 3.x
- Jupyter Notebook
- PyTorch
- NumPy
- Pandas
- Matplotlib

Install these dependencies using pip or conda.
