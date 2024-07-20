# Specific Task 3d: Masked Auto-Encoder for Efficient End-to-End Particle Reconstruction and Compression

### Tasks
1. Train a lightweight ViT using the Masked Auto-Encoder (MAE) training scheme on the unlabelled dataset.
2. Compare reconstruction results using MAE on both training and testing datasets.
3. Fine-tune the model on a lower learning rate on the provided labelled dataset and compare results with a model trained from scratch.

<p align="center">
  <img src="https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Specific%20Task%203d%20-%20Masked_Autoencoder/MAE.png" width="700" title="hover text">
</p>

### Implementation
- Trained a lightweight ViT using MAE on unlabelled dataset
- Compared reconstruction results on training and testing datasets
- Fine-tuned the model on a lower learning rate using the labelled dataset
- Compared results with a model trained from scratch
- Ensured no overfitting on the test dataset

### Image Reconstruction
####                                           Original
<p align="center">
  <img src="https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Specific%20Task%203d%20-%20Masked_Autoencoder/Original.jpg" width="700" title="hover text">
</p>

####                                           Reconstructed
<p align="center">
  <img src="https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Specific%20Task%203d%20-%20Masked_Autoencoder/Reconstructed.jpg" width="700" title="hover text">
</p>

### Comparison of With and Without Pretrained Vision Transformer Model
<p align="center">
  <img src="
https://github.com/Wodlfvllf/E2E/blob/Masked_autoencoders_Shashank/Masked_AutoEncoders_E2E_GSCO24_Shashank_Shukla/Performance_table.png" width="700" title="hover text">
</p>                         
Both models are fine-tuned on learning rate of 1.e-5 using AdamW optimizer.

            
#### Notebooks: 
Here are the notebooks showing complete training process.

- [MAE_Particle_Reconstruction.ipynb](https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Specific%20Task%203d%20-%20Masked_Autoencoder/Masked_Autoencoder/Masked%20Autoencoder.ipynb)
- [linear-probing-Pretraining.ipynb](https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Specific%20Task%203d%20-%20Masked_Autoencoder/Linear%20Probing%20MAE/linear-probing-Pretraining.ipynb)
- [linear-probing-without Pretraining.ipynb](https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Specific%20Task%203d%20-%20Masked_Autoencoder/Linear%20Probing%20MAE/linear-probing-without%20Pretraining.ipynb)
- Includes data loading, model training (pre-training and fine-tuning), evaluation, and model weights

#### Example Notebooks:
These are Example Notebooks to inference or reproduce the results

-  [Example Train MAE](https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Specific%20Task%203d%20-%20Masked_Autoencoder/Masked_Autoencoder/Example_Train.ipynb)
-  [Example Test MAE](https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Specific%20Task%203d%20-%20Masked_Autoencoder/Masked_Autoencoder/Example_Test.ipynb)
-  [Example Test Linear Probing](https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Specific%20Task%203d%20-%20Masked_Autoencoder/Linear%20Probing%20MAE/Example_test_linear%20probe.ipynb)
-  [Example Train Linear Probing](https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Specific%20Task%203d%20-%20Masked_Autoencoder/Linear%20Probing%20MAE/Example%20Train%20Linear%20Probing.ipynb)

## Dependencies
- Python 3.x
- Jupyter Notebook
- PyTorch
- NumPy
- Pandas
- Matplotlib

Install these dependencies using pip or conda.
