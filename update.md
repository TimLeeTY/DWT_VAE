---
title: |
    | **Hierarchical Block Matrices for**
    | **Multi-scale Neural Networks**
author: |
    | Candidate: Tim T. Y. Lee, 
    | CRSid: `tyl35`
    | Supervisor: Pietro Lio'
geometry: margin=3cm
numbersections: true
header-includes:
    - \newcommand{\diff}[2][]{\mathop{}\!\mathrm{d}^{#1} {#2} \,}
    - \newcommand{\given}[1][]{\, {#1} \vert \,}
    - \usepackage{amsmath}
    - \usepackage{mathtools}
    - \usepackage{mathrsfs}
    - \usepackage{float}
    - \usepackage{subfig}
    - \usepackage{xfrac}
    - \usepackage{caption}
---

# Progress Update

As per our previous discussions, I have shifted the focus of the project to look at the combination of 1D Discrete Wavelet Transformations with Variational Auto-encoders applied to audio samples as a baseline for my project.

Audio is an ideal 1-dimensional data that contains data on multiple scales. While the pitch of a note is determined by its fundamental frequency (usually around $10^2$ Hz), its timbre depends on the overtones/harmonics and its envelope (time variations in the intensity of the sound). As such, wavelet transformations which capture both the time and frequency variations of a signal are ideal for this task.

- Code written in Python, using `PyWavelet` for efficient 1D DWT, TensorFlow for training VAE, and `Numpy`/`Matplotlib`
- Audio samples taken from the NSynth Dataset (from `tensorflow-datasets`), an annotated dataset of audio clips, each containing a different musical note played by one of 1006 different instruments. 

Currently, I have implemented a working pipeline that applies a Discrete Wavelet Transformation to each audio sample, which is used to train the Variational Auto-Encoder. The VAE is trained to minimise the reconstruction error of the data in 'wavelet space'.

- Structure of VAE: only tested 3 fully connected layers each for encoder (inference network) and decoder (generative network), subject to some downsampling
- Latent dimension of VAE set to 10
- Latent dimensions are then projected onto 2 dimensions for data visualisation to check the quality of the separation of different instruments/pitches etc.


In the meantime, I am working to also include the following into my project:

- Introduce noise at different spectral densities (varying $1/f^\beta$) and intensities to the training data to investigate de-noising capabilities.
- Train generator network to reconstruct wavelet transformation as well, (also done by [Recoskie and Mann, 2018](https://arxiv.org/pdf/1802.02961.pdf))
