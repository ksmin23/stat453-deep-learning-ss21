# stat453-deep-learning-ss21
STAT 453: Intro to Deep Learning @ UW-Madison (Spring 2021)

## Table of Contents

  - [Part 1: Introduction](#part-1-introduction)
    - [L01: Introduction to deep learning](#l01-introduction-to-deep-learning)
    - [L02: The brief history of deep learning](#l02-the-brief-history-of-deep-learning)
    - [L03: Single-layer neural networks: The perceptron algorithm](#l03-single-layer-neural-networks-the-perceptron-algorithm)
  - [Part 2: Mathematical and computational foundations](#part-2-mathematical-and-computational-foundations)
    - [L04: Linear algebra and calculus for deep learning](#l04-linear-algebra-and-calculus-for-deep-learning)
    - [L05: Parameter optimization with gradient descent](#l05-parameter-optimization-with-gradient-descent)
    - [L06: Automatic differentiation with PyTorch](#l06-automatic-differentiation-with-pytorch)
    - [L07: Cluster and cloud computing resources](#l07-cluster-and-cloud-computing-resources)
  - [Part 3: Introduction to neural networks](#part-3-introduction-to-neural-networks)
    - [L08: Multinomial logistic regression / Softmax regression](#l08-multinomial-logistic-regression--softmax-regression)
    - [L09: Multilayer perceptrons and backpropration](#l09-multilayer-perceptrons-and-backpropration)
    - [L10: Regularization to avoid overfitting](#l10-regularization-to-avoid-overfitting)
    - [L11: Input normalization and weight initialization](#l11-input-normalization-and-weight-initialization)
    - [L12: Learning rates and advanced optimization algorithms](#l12-learning-rates-and-advanced-optimization-algorithms)
  - [Part 4: Deep learning for computer vision and language modeling](#part-4-deep-learning-for-computer-vision-and-language-modeling)
    - [L13: Introduction to convolutional neural networks](#l13-introduction-to-convolutional-neural-networks)
    - [L14: Convolutional neural networks architectures](#l14-convolutional-neural-networks-architectures)
    - [L15: Introduction to recurrent neural networks](#l15-introduction-to-recurrent-neural-networks)
  - [Part 5: Deep generative models](#part-5-deep-generative-models)
    - [L16: Autoencoders](#l16-autoencoders)
    - [L17: Variational autoencoders](#l17-variational-autoencoders)
    - [L18: Introduction to generative adversarial networks](#l18-introduction-to-generative-adversarial-networks)
    - [L19: Self-attention and transformer networks](#l19-self-attention-and-transformer-networks)
  - [Supplementary Resources](#supplementary-resources)


## Part 1: Introduction
### L01: Introduction to deep learning

| Videos | Material |
|--------|----------|
| [L1.0 Intro to Deep Learning, Course Introduction](https://www.youtube.com/watch?v=1nqCZqDYPp0) | [L01-intro_slides.pdf](https://sebastianraschka.com/pdf/lecture-notes/stat453ss21/L01-intro_slides.pdf) |
| [L1.1.1 Course Overview Part 1: Motivation and Topics](https://www.youtube.com/watch?v=6VbtJ9nn5ng) ||
| [L1.1.2 Course Overview Part 2: Organization (Optional)](https://www.youtube.com/watch?v=s7ZCbKI5Exw) ||
| [L1.2 What is Machine Learning?](https://www.youtube.com/watch?v=d6oQzE4kst0) ||
| [L1.3.1 Broad Categories of ML Part 1: Supervised Learning](https://www.youtube.com/watch?v=UadzJLHJB50) ||
| [L1.3.2 Broad Categories of ML Part 2: Unsupervised Learning](https://www.youtube.com/watch?v=nHhuuUwd05g) ||
| [L1.3.3 Broad Categories of ML Part 3: Reinforcement Learning](https://www.youtube.com/watch?v=EQCZUOxGrOo) ||
| [L1.3.4 Broad Categories of ML Part 4: Special Cases of Supervised Learning](https://www.youtube.com/watch?v=B59lK5yo57M) ||
| [L1.4 The Supervised Learning Workflow](https://www.youtube.com/watch?v=nd9dhrvtIA0) ||
| [L1.5 Necessary Machine Learning Notation and Jargon](https://www.youtube.com/watch?v=o-yHLOvuh2o) ||
| [L1.6 About the Practical Aspects and Tools Used in This Course](https://www.youtube.com/watch?v=R16VmI2ZhR0) | [code](./L01) |

### L02: The brief history of deep learning

| Videos | Material |
|--------|----------|
| [L2.0 A Brief History of Deep Learning -- Lecture Overview](https://www.youtube.com/watch?v=Ezig00nypvU) | [L02_dl-history_slides.pdf](https://sebastianraschka.com/pdf/lecture-notes/stat453ss21/L02_dl-history_slides.pdf) |
| [L2.1 Artificial Neurons](https://www.youtube.com/watch?v=gbLasjwAGik) ||
| [L2.2 Multilayer Networks](https://www.youtube.com/watch?v=G7oqVqU5qsQ) ||
| [L2.3 The Origins of Deep Learning](https://www.youtube.com/watch?v=tkUCMtJd43Y) ||
| [L2.4 The Deep Learning Hardware &amp; Software Landscape](https://www.youtube.com/watch?v=TMCNkeJGIfg) ||
| [L2.5 Current Trends in Deep Learning](https://www.youtube.com/watch?v=FpOpb-BMIH8) ||

### L03: Single-layer neural networks: The perceptron algorithm

| Videos | Material |
|--------|----------|
| [L3.0 Perceptron Lecture Overview](https://www.youtube.com/watch?v=cm_wv2QpTgc) | [L03_perceptron_slides.pdf](https://sebastianraschka.com/pdf/lecture-notes/stat453ss21/L03_perceptron_slides.pdf) |
| [L3.1 About Brains and Neurons](https://www.youtube.com/watch?v=AnSDPcvtRLo) ||
| [L3.2 The Perceptron Learning Rule](https://www.youtube.com/watch?v=C8Uns9HEVXI) | [code](./L03) |
| [L3.3 Vectorization in Python](https://www.youtube.com/watch?v=OnG2NfuC5aY) | [code](./L03) |
| [L3.4 Perceptron in Python using NumPy and PyTorch](https://www.youtube.com/watch?v=TlGpIKMVoOg) | [code](./L03) |
| [L3.5 The Geometric Intuition Behind the Perceptron](https://www.youtube.com/watch?v=Fj7BgxI73TA) ||

## Part 2: Mathematical and computational foundations
### L04: Linear algebra and calculus for deep learning

| Videos | Material |
|--------|----------|
| [L4.0 Linear Algebra for Deep Learning -- Lecture Overview](https://www.youtube.com/watch?v=3mjJxu3B0zA) | [L04_linalg-dl_slides.pdf](https://sebastianraschka.com/pdf/lecture-notes/stat453ss21/L04_linalg-dl_slides.pdf) |
| [L4.1 Tensors in Deep Learning](https://www.youtube.com/watch?v=JXfDlgrfOBY) ||
| [L4.2 Tensors in PyTorch](https://www.youtube.com/watch?v=zk_asBov8QI) ||
| [L4.3 Vectors, Matrices, and Broadcasting](https://www.youtube.com/watch?v=4Ehb_is-MFU) ||
| [L4.4 Notational Conventions for Neural Networks](https://www.youtube.com/watch?v=4pnoymfFiYM) ||
| [L4.5 A Fully Connected (Linear) Layer in PyTorch](https://www.youtube.com/watch?v=XswEBzNgIYc) ||

### L05: Parameter optimization with gradient descent

| Videos | Material |
|--------|----------|
| [L5.0 Gradient Descent -- Lecture Overview](https://www.youtube.com/watch?v=VBOxg62CwCg) | [L05_gradient-descent_slides.pdf](https://sebastianraschka.com/pdf/lecture-notes/stat453ss21/L05_gradient-descent_slides.pdf) |
| [L5.1 Online, Batch, and Minibatch Mode](https://www.youtube.com/watch?v=b4DXHd3RwqA) ||
| [L5.2 Relation Between Perceptron and Linear Regression](https://www.youtube.com/watch?v=4JB1j8eIGzI) ||
| [L5.3 An Iterative Training Algorithm for Linear Regression](https://www.youtube.com/watch?v=1QH2bVuV98A) ||
| [L5.4 (Optional) Calculus Refresher I: Derivatives](https://www.youtube.com/watch?v=tL1THESrXgI) ||
| [L5.5 (Optional) Calculus Refresher II: Gradients](https://www.youtube.com/watch?v=YPZVGSRmjLk) ||
| [L5.6 Understanding Gradient Descent](https://www.youtube.com/watch?v=L4xzybIa-bo) ||
| [L5.7 Training an Adaptive Linear Neuron (Adaline)](https://www.youtube.com/watch?v=iLCT0i-lCsw) ||
| [L5.8 Adaline Code Example](https://www.youtube.com/watch?v=GGcaqzhKzLc) | [code](./L05) |

### L06: Automatic differentiation with PyTorch

| Videos | Material |
|--------|----------|
| [L6.0 Automatic Differentiation in PyTorch -- Lecture Overview](https://www.youtube.com/watch?v=j1-r1vO2a_o) | [L06_pytorch_slides.pdf](https://sebastianraschka.com/pdf/lecture-notes/stat453ss21/L06_pytorch_slides.pdf) |
| [L6.1 Learning More About PyTorch](https://www.youtube.com/watch?v=LjdiVPQ45GE) ||
| [L6.2 Understanding Automatic Differentiation via Computation Graphs](https://www.youtube.com/watch?v=oY6-i2Ybin4) ||
| [L6.3 Automatic Differentiation in PyTorch -- Code Example](https://www.youtube.com/watch?v=VvUz0Q9e09g) | [code](./L06) |
| [L6.4 Training ADALINE with PyTorch -- Code Example](https://www.youtube.com/watch?v=00KgeJwNaZA) | [code](./L06) |
| [L6.5 A Closer Look at the PyTorch API](https://www.youtube.com/watch?v=klc79sZ1yVc) ||

### L07: Cluster and cloud computing resources

| Videos | Material |
|--------|----------|
| [L7.0 GPU resources &amp; Google Colab](https://www.youtube.com/watch?v=5pew4YEa1ww) | [L07_cloud-computing_slides.pdf](https://sebastianraschka.com/pdf/lecture-notes/stat453ss21/L07_cloud-computing_slides.pdf) |

## Part 3: Introduction to neural networks
### L08: Multinomial logistic regression / Softmax regression

| Videos | Material |
|--------|----------|
| [L8.0 Logistic Regression -- Lecture Overview](https://www.youtube.com/watch?v=10PTpRRpRk0) | [L08_logistic__slides.pdf](https://sebastianraschka.com/pdf/lecture-notes/stat453ss21/L08_logistic__slides.pdf) |
| [L8.1 Logistic Regression as a Single-Layer Neural Network](https://www.youtube.com/watch?v=ncZ5iSZekVQ) ||
| [L8.2 Logistic Regression Loss Function](https://www.youtube.com/watch?v=GxJe0DZvydM) ||
| [L8.3 Logistic Regression Loss Derivative and Training](https://www.youtube.com/watch?v=7rR1L7t2EnA) ||
| [L8.4 Logits and Cross Entropy](https://www.youtube.com/watch?v=icQaFxKa_J0) ||
| [L8.5 Logistic Regression in PyTorch -- Code Example](https://www.youtube.com/watch?v=6igMArA6k3A) | [code](./L08) |
| [L8.6 Multinomial Logistic Regression / Softmax Regression](https://www.youtube.com/watch?v=L0FU8NFpx4E) ||
| [L8.7.1 OneHot Encoding and Multi-category Cross Entropy](https://www.youtube.com/watch?v=4n71-tZ94yk) | [code](./L08) |
| [L8.7.2 OneHot Encoding and Multi-category Cross Entropy -- Code Example](https://www.youtube.com/watch?v=5bW0vn4ISqs) | [code](./L08) |
| [L8.8 Softmax Regression Derivatives for Gradient Descent](https://www.youtube.com/watch?v=aeM-fmcdkXU) ||
| [L8.9 Softmax Regression -- Code Example Using PyTorch](https://www.youtube.com/watch?v=mM6apVBXGEA) | [code](./L08) |

### L09: Multilayer perceptrons and backpropration

| Videos | Material |
|--------|----------|
| [L9.0 Multilayer Perceptrons -- Lecture Overview](https://www.youtube.com/watch?v=jD6IKpqSJM4) | [L09_mlp__slides.pdf](https://sebastianraschka.com/pdf/lecture-notes/stat453ss21/L09_mlp__slides.pdf) |
| [L9.1 Multilayer Perceptron Architecture](https://www.youtube.com/watch?v=IUylp47hNA0) ||
| [L9.2 Nonlinear Activation Functions](https://www.youtube.com/watch?v=-_7W0KE8Ykg) | [code](./L09) |
| [L9.3.1 Multilayer Perceptron -- Code Example Part 1/3 (Slide Overview)](https://www.youtube.com/watch?v=zNyEzACInRg) ||
| [L9.3.2 Multilayer Perceptron in PyTorch -- Code Example Part 2/3 (Jupyter Notebook)](https://www.youtube.com/watch?v=Ycp4Si89s5Q) | [code](./L09) |
| [L9.3.3 Multilayer Perceptron in PyTorch -- Code Example Part 3/3 (Script Setup)](https://www.youtube.com/watch?v=cDbQgQv_Yz0) | [code](./L09) |
| [L9.4 Overfitting and Underfitting](https://www.youtube.com/watch?v=hFGZyDVNgS4) ||
| [L9.5.1 Cats &amp; Dogs and Custom Data Loaders](https://www.youtube.com/watch?v=RQIAmvElu1g) ||
| [L9.5.2 Custom DataLoaders in PyTorch --Code Example](https://www.youtube.com/watch?v=hPzJ8H0Jtew) | [code](./L09) |

### L10: Regularization to avoid overfitting

| Videos | Material |
|--------|----------|
| [L10.0 Regularization Methods for Neural Networks -- Lecture Overview](https://www.youtube.com/watch?v=Va4K-wYh_p8) | [L10_regularization__slides.pdf](https://sebastianraschka.com/pdf/lecture-notes/stat453ss21/L10_regularization__slides.pdf) |
| [L10.1 Techniques for Reducing Overfitting](https://www.youtube.com/watch?v=KOBmBjlMVAE) ||
| [L10.2 Data Augmentation in PyTorch](https://www.youtube.com/watch?v=qLIosWyrh9Q) | [code](./L10) |
| [L10.3 Early Stopping](https://www.youtube.com/watch?v=YA1OdkiHJBY) ||
| [L10.4 L2 Regularization for Neural Nets](https://www.youtube.com/watch?v=uu2X47cSLmM) | [code](./L10) |
| [L10.5.1 The Main Concept Behind Dropout](https://www.youtube.com/watch?v=IHrZNBsgtwU) ||
| [L10.5.2 Dropout Co-Adaptation Interpretation](https://www.youtube.com/watch?v=GAE8dpDWo6E) ||
| [L10.5.3 (Optional) Dropout Ensemble Interpretation](https://www.youtube.com/watch?v=4We9G5jgKvI) ||
| [L10.5.4 Dropout in PyTorch](https://www.youtube.com/watch?v=kma-4wqp_-k) | [code](./L10) |

### L11: Input normalization and weight initialization

| Videos | Material |
|--------|----------|
| [L11.0 Input Normalization and Weight Initialization -- Lecture Overview](https://www.youtube.com/watch?v=xk6qb2IePaE) | [L11_norm-and-init__slides.pdf](https://sebastianraschka.com/pdf/lecture-notes/stat453ss21/L11_norm-and-init__slides.pdf) |
| [L11.1  Input Normalization](https://www.youtube.com/watch?v=jzJactQXFDk) ||
| [L11.2 How BatchNorm Works](https://www.youtube.com/watch?v=34PDIFvvESc) ||
| [L11.3 BatchNorm in PyTorch -- Code Example](https://www.youtube.com/watch?v=8AUDn7iF2DY) | [code](./L11) |
| [L11.4 Why BatchNorm Works](https://www.youtube.com/watch?v=uI19wIdzh9M) ||
| [L11.5 Weight Initialization -- Why Do We Care?](https://www.youtube.com/watch?v=RsX01aYbQdI) ||
| [L11.6 Xavier Glorot and Kaiming He Initialization](https://www.youtube.com/watch?v=ScWTYHQra5E) ||
| [L11.7 Weight Initialization in PyTorch -- Code Example](https://www.youtube.com/watch?v=nA6oEAE9IVc) | [code](./L11) |

### L12: Learning rates and advanced optimization algorithms

| Videos | Material |
|--------|----------|
| [L12.0: Improving Gradient Descent-based Optimization -- Lecture Overview](https://www.youtube.com/watch?v=7RhNXYqDBfU) | [L12_optim__slides.pdf](https://sebastianraschka.com/pdf/lecture-notes/stat453ss21/L12_optim__slides.pdf) |
| [L12.1 Learning Rate Decay](https://www.youtube.com/watch?v=Owm1H0ukjS4) ||
| [L12.2 Learning Rate Schedulers in PyTorch](https://www.youtube.com/watch?v=tB1rz4L93JA) | [code](./L12) |
| [L12.3 SGD with Momentum](https://www.youtube.com/watch?v=gMxvefj0YAM) ||
| [L12.4 Adam: Combining Adaptive Learning Rates and Momentum](https://www.youtube.com/watch?v=eUOvUIRPSX8) | [code](./L12) |
| [L12.5 Choosing Different Optimizers in PyTorch](https://www.youtube.com/watch?v=c-SRPvK_zzs) ||
| [L12.6 Additional Topics and Research on Optimization Algorithms](https://www.youtube.com/watch?v=7yoAocFiUh8) ||

## Part 4: Deep learning for computer vision and language modeling
### L13: Introduction to convolutional neural networks

| Videos | Material |
|--------|----------|
| [L13.0 Introduction to Convolutional Networks -- Lecture Overview](https://www.youtube.com/watch?v=i-Ngb6tn_KM) | [L13_intro-cnn__slides.pdf](https://sebastianraschka.com/pdf/lecture-notes/stat453ss21/L13_intro-cnn__slides.pdf) |
| [L13.1 Common Applications of CNNs](https://www.youtube.com/watch?v=I5B7pgSEMhE) ||
| [L13.2 Challenges of Image Classification](https://www.youtube.com/watch?v=0FtJbmuUdFo) ||
| [L13.3 Convolutional Neural Network Basics](https://www.youtube.com/watch?v=7fWOE-z8YgY) ||
| [L13.4 Convolutional Filters and Weight-Sharing](https://www.youtube.com/watch?v=ryJ6Bna-ZNU) ||
| [L13.5 Cross-correlation vs. Convolution (Old)](https://www.youtube.com/watch?v=ICWHhxox1ho) | [code](./L13) |
| [L13.5 What's The Difference Between Cross-Correlation And Convolution?](https://www.youtube.com/watch?v=xbO-iIzkBy0) | [code](./L13) |
| [L13.6 CNNs &amp; Backpropagation](https://www.youtube.com/watch?v=-SwKNK9MIUU) ||
| [L13.7 CNN Architectures &amp; AlexNet](https://www.youtube.com/watch?v=-IHxe4-09e4) ||
| [L13.8 What a CNN Can See](https://www.youtube.com/watch?v=PRFP5YC3u7g) ||
| [L13.9.1 LeNet-5 in PyTorch](https://www.youtube.com/watch?v=ye5k82FQC7I) | [code](./L13) |
| [L13.9.2 Saving and Loading Models in PyTorch](https://www.youtube.com/watch?v=vB_Y04gsyBI) | [code](./L13) |
| [L13.9.3 AlexNet in PyTorch](https://www.youtube.com/watch?v=mlXRVuD_HEg) | [code](./L13) |

### L14: Convolutional neural networks architectures

| Videos | Material |
|--------|----------|
| [L14.0: Convolutional Neural Networks Architectures -- Lecture Overview](https://www.youtube.com/watch?v=1A6HViSXaqQ) | [L14_cnn-architectures_slides.pdf](https://sebastianraschka.com/pdf/lecture-notes/stat453ss21/L14_cnn-architectures_slides.pdf) |
| [L14.1: Convolutions and Padding](https://www.youtube.com/watch?v=6v05kAtV1M0) ||
| [L14.2: Spatial Dropout and BatchNorm](https://www.youtube.com/watch?v=TGqqTgn4cAg) ||
| [L14.3: Architecture Overview](https://www.youtube.com/watch?v=WyXO762G2_A) ||
| [L14.3.1.1 VGG16 Overview](https://www.youtube.com/watch?v=YcmNIOyfdZQ) ||
| [L14.3.1.2 VGG16 in PyTorch -- Code Example](https://www.youtube.com/watch?v=PlFiRPdBEAo) | [code](./L14) |
| [L14.3.2.1 ResNet Overview](https://www.youtube.com/watch?v=q_IlqYlYhlo) ||
| [L14.3.2.2 ResNet-34 in PyTorch -- Code Example](https://www.youtube.com/watch?v=JG_ODvnlgjY) | [code](./L14) |
| [L14.4.1 Replacing Max-Pooling with Convolutional Layers](https://www.youtube.com/watch?v=Lq83NFkkJCk) ||
| [L14.4.2 All-Convolutional Network in PyTorch -- Code Example](https://www.youtube.com/watch?v=A5dC5yuPXwo) | [code](./L14) |
| [L14.5 Convolutional Instead of Fully Connected Layers](https://www.youtube.com/watch?v=rqLjZ8k4va8) ||
| [L14.6.1 Transfer Learning](https://www.youtube.com/watch?v=OkQRtm9JY1k) ||
| [L14.6.2 Transfer Learning in PyTorch -- Code Example](https://www.youtube.com/watch?v=FaW9JCSJn2s) | [code](./L14) |

### L15: Introduction to recurrent neural networks

| Videos | Material |
|--------|----------|
| [L15.0: Introduction to Recurrent Neural Networks -- Lecture Overview](https://www.youtube.com/watch?v=q5YxK17tRm0) | [L15_intro-rnn__slides.pdf](https://sebastianraschka.com/pdf/lecture-notes/stat453ss21/L15_intro-rnn__slides.pdf) |
| [L15.1: Different Methods for Working With Text Data](https://www.youtube.com/watch?v=kwmZtkzB4e0) ||
| [L15.2 Sequence Modeling with RNNs](https://www.youtube.com/watch?v=5fdy-hBeWCI) ||
| [L15.3 Different Types of Sequence Modeling Tasks](https://www.youtube.com/watch?v=Ed8GTvkzkZE) ||
| [L15.4 Backpropagation Through Time Overview](https://www.youtube.com/watch?v=0XdPIqi0qpg) ||
| [L15.5 Long Short-Term Memory](https://www.youtube.com/watch?v=k6fSgUaWUF8) ||
| [L15.6 RNNs for Classification: A Many-to-One Word RNN](https://www.youtube.com/watch?v=TI4HRR3Hd9A) ||
| [L15.7 An RNN Sentiment Classifier in PyTorch](https://www.youtube.com/watch?v=KgrdifrlDxg) | [code](./L15) |

## Part 5: Deep generative models
### L16: Autoencoders

| Videos | Material |
|--------|----------|
| [L16.0 Introduction to Autoencoders -- Lecture Overview](https://www.youtube.com/watch?v=9Ujv_IoBtF4) | [L16_autoencoder__slides.pdf](https://sebastianraschka.com/pdf/lecture-notes/stat453ss21/L16_autoencoder__slides.pdf) |
| [L16.1 Dimensionality Reduction](https://www.youtube.com/watch?v=UgOHupaIfcA) ||
| [L16.2 A Fully-Connected Autoencoder](https://www.youtube.com/watch?v=8O_FDPIlj1s) ||
| [L16.3 Convolutional Autoencoders &amp; Transposed Convolutions](https://www.youtube.com/watch?v=ilkSwsggSNM) ||
| [L16.4 A Convolutional Autoencoder in PyTorch -- Code Example](https://www.youtube.com/watch?v=345wRyqKkQ0) | [code](./L16) |
| [L16.5 Other Types of Autoencoders](https://www.youtube.com/watch?v=FPZeRM1p1ao) ||

### L17: Variational autoencoders

| Videos | Material |
|--------|----------|
| [L17.0 Intro to Variational Autoencoders -- Lecture Overview](https://www.youtube.com/watch?v=UnImUYOdWgk) | [L17_vae__slides.pdf](https://sebastianraschka.com/pdf/lecture-notes/stat453ss21/L17_vae__slides.pdf) |
| [L17.1 Variational Autoencoder Overview](https://www.youtube.com/watch?v=H2XgdND0DV4) ||
| [L17.2 Sampling from a Variational Autoencoder](https://www.youtube.com/watch?v=YgSWrafXI8U) ||
| [L17.3 The Log-Var Trick](https://www.youtube.com/watch?v=pmvo0S3-G-I) ||
| [L17.4 Variational Autoencoder Loss Function](https://www.youtube.com/watch?v=ywYuZrLENH0) ||
| [L17.5 A Variational Autoencoder for Handwritten Digits in PyTorch -- Code Example](https://www.youtube.com/watch?v=afNuE5z2CQ8) | [code](./L17) |
| [L17.6 A Variational Autoencoder for Face Images in PyTorch -- Code Example](https://www.youtube.com/watch?v=sul2ExoUrnw) | [code](./L17) |
| [L17.7 VAE Latent Space Arithmetic in PyTorch -- Making People Smile (Code Example)](https://www.youtube.com/watch?v=EfFr87ARDF0) | [code](./L17) |

### L18: Introduction to generative adversarial networks

| Videos | Material |
|--------|----------|
| [L18.0: Introduction to Generative Adversarial Networks -- Lecture Overview](https://www.youtube.com/watch?v=OnoPaZaKoS8) | [L18_gan__slides.pdf](https://sebastianraschka.com/pdf/lecture-notes/stat453ss21/L18_gan__slides.pdf) |
| [L18.1: The Main Idea Behind GANs](https://www.youtube.com/watch?v=-Zi5SReze6U) ||
| [L18.2: The GAN Objective](https://www.youtube.com/watch?v=m_H6viKCTEE) ||
| [L18.3: Modifying the GAN Loss Function for Practical Use](https://www.youtube.com/watch?v=ILpC3b-819Q) ||
| [L18.4: A GAN for Generating Handwritten Digits in PyTorch -- Code Example](https://www.youtube.com/watch?v=cTlxZ1FO1mY) | [code](./L18) |
| [L18.5: Tips and Tricks to Make GANs Work](https://www.youtube.com/watch?v=_cUdjPdbldQ) | https://github.com/soumith/ganhacks |
| [L18.6: A DCGAN for Generating Face Images in PyTorch -- Code Example](https://www.youtube.com/watch?v=5fs9PMzrVig) | [code](./L18) |

### L19: Self-attention and transformer networks

| Videos | Material |
|--------|----------|
| [L19.0 RNNs &amp; Transformers for Sequence-to-Sequence Modeling -- Lecture Overview](https://www.youtube.com/watch?v=DlWTTrHa8bI) | [L19_seq2seq_rnn-transformers__slides.pdf](https://sebastianraschka.com/pdf/lecture-notes/stat453ss21/L19_seq2seq_rnn-transformers__slides.pdf) |
| [L19.1 Sequence Generation with Word and Character RNNs](https://www.youtube.com/watch?v=fSBw6TrePPg) ||
| [L19.2.1 Implementing a Character RNN in PyTorch (Concepts)](https://www.youtube.com/watch?v=PFcWQkGP4lU) ||
| [L19.2.2 Implementing a Character RNN in PyTorch --Code Example](https://www.youtube.com/watch?v=tL5puCeDr-o) | [code](./L19) |
| [L19.3 RNNs with an Attention Mechanism](https://www.youtube.com/watch?v=mDZil99CtSU) ||
| [L19.4.1 Using Attention Without the RNN -- A Basic Form of Self-Attention](https://www.youtube.com/watch?v=i_pfHD4P_wg) ||
| [L19.4.2 Self-Attention and Scaled Dot-Product Attention](https://www.youtube.com/watch?v=0PjHri8tc1c) ||
| [L19.4.3 Multi-Head Attention](https://www.youtube.com/watch?v=A1eUVxscNq8) ||
| [L19.5.1 The Transformer Architecture](https://www.youtube.com/watch?v=tstbZXNCfLY) ||
| [L19.5.2.1 Some Popular Transformer Models: BERT, GPT, and BART -- Overview](https://www.youtube.com/watch?v=iFhYwEi03Ew) ||
| [L19.5.2.2 GPT-v1: Generative Pre-Trained Transformer](https://www.youtube.com/watch?v=LOCzBgSV4tQ) ||
| [L19.5.2.3 BERT: Bidirectional Encoder Representations from Transformers](https://www.youtube.com/watch?v=_BFp4kjSB-I) ||
| [L19.5.2.4 GPT-v2: Language Models are Unsupervised Multitask Learners](https://www.youtube.com/watch?v=BXv1m9Asl7I) ||
| [L19.5.2.5 GPT-v3: Language Models are Few-Shot Learners](https://www.youtube.com/watch?v=wYdKn-X4MhY) ||
| [L19.5.2.6 BART:  Combining Bidirectional and Auto-Regressive Transformers](https://www.youtube.com/watch?v=1JBMCG8rW18) ||
| [L19.5.2.7: Closing Words -- The Recent Growth of Language Transformers](https://www.youtube.com/watch?v=OyqIuxMmLRg) ||
| [L19.6 DistilBert Movie Review Classifier in PyTorch -- Code Example](https://www.youtube.com/watch?v=emDmznRlsWw) | [code](./L19) |

## Supplementary Resources

 * [(Blog) Introduction to Deep Learning (2021-07-09)](https://sebastianraschka.com/blog/2021/dl-course.html)
 * [(Blog) Scientific Computing in Python: Introduction to NumPy and Matplotlib](https://sebastianraschka.com/blog/2020/numpy-intro.html)
 * [(GitHub) Udacity's Deep Learning (PyTorch) - ND101 v7](https://github.com/ksmin23/deep-learning-v2-pytorch)
