# **CNN-Based Autoencoder for SVHN Dataset**

**📌 Project Overview**

This project implements a **Convolutional Neural Network (CNN)-based Autoencoder** using **PyTorch**, trained on the **Street View House Numbers (SVHN) dataset** for image reconstruction. The key focus is on weight-clipping regularization and sparsity constraints to analyze their effects on reconstruction performance.

**🎯 Objectives**

- Train an **autoencoder** for image reconstruction using **Mean Squared Error (MSE) loss**.
- Implement a **weight-clipping constraint** to restrict all weights to the range **[-0.5, 0.5]**.
- Train another autoencoder with an **L1-based sparsity constraint** alongside the weight-clipping.
- Compare both models using **Peak Signal-to-Noise Ratio (PSNR)** to evaluate reconstruction quality.
- Perform **hyperparameter tuning** by experimenting with different **activation functions** and **optimizers**.
- Train the models for **at least 100 epochs** or until the loss curve stabilizes.

**📂 Dataset**

We use the **SVHN dataset**, a real-world digit dataset obtained from house numbers in Google Street View images.

- Download the dataset: SVHN Dataset (torchvision)

**🛠️ Implementation**

**Model Architecture**

Each autoencoder consists of:

- **Encoder**: Convolutional layers to compress image features.
- **Latent Representation**: A bottleneck representation.
- **Decoder**: Deconvolutional layers to reconstruct images.

**Constraints Applied**

- **Model 1**: CNN autoencoder with **weight clipping** (restricts weights to [-0.5, 0.5]).
- **Model 2**: CNN autoencoder with **weight clipping** + **L1-based sparsity** constraint.

**Performance Metric**

- **PSNR (Peak Signal-to-Noise Ratio)** is used to measure the quality of reconstructed images.
- The higher the PSNR, the better the reconstruction.

**🚀 Training Strategy**

1. Load the **SVHN dataset** and preprocess images.
1. Train two CNN-based autoencoders:
   1. **Autoencoder 1**: MSE loss + weight clipping.
   1. **Autoencoder 2**: MSE loss + weight clipping + L1-based sparsity.
1. Train each model for **at least 100 epochs**, monitoring loss and PSNR.
1. Experiment with different **activation functions** and **optimizers** for hyperparameter tuning.
1. Compare the models based on reconstruction quality.

**🔧 Hyperparameter Tuning**

We experiment with:

- **Activation Functions**: ReLU, LeakyReLU, Sigmoid, Tanh.
- **Optimizers**: Adam, RMSprop, SGD.
- Learning rate adjustments for optimal convergence.

**📈 Results & Comparison**

- The models are evaluated based on **loss curves and PSNR scores**.
- Plots are generated to visualize the reconstruction quality.
- A comparison is made to determine the **most effective regularization strategy**.
- 
![3](https://github.com/user-attachments/assets/08d380d4-0738-4112-b904-9f7de5eb5bd2)
![2](https://github.com/user-attachments/assets/a0b3be40-4d8d-4a99-89f3-81eaa825b366)
![1](https://github.com/user-attachments/assets/b9431572-9995-4e48-b1b8-d20aab74e7cc)

