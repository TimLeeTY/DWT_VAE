---
documentclass: report
title: |
    | \huge Variational Autoencoder for Audio Signals
    | \huge Using the Discrete Wavelet Transform
    |
    |
    |
    |
author: |
    | \Large Supervisor: Pietro Lio'
    | \Large Canidate Number: 8222V
    |
    |
    |
    |
    |
    |
    |
    |
    | \includegraphics[width=6cm]{CambridgeUniversityCMYK.eps} 
    |
    |
    | Department of Physics
    | University of Cambridge
geometry: margin=3cm
numbersections: true
output: pdf_document
bibliography: biblio.bib
toc: true
date: May 13, 2019
header-includes: |
    ```{=latex}
    \usepackage{amsmath}
    \usepackage{mathtools}
    \usepackage{mathrsfs}
    \usepackage{bm}
    \usepackage{float}
    \usepackage{subfig}
    \usepackage{xfrac}
    \usepackage[binary-units=true]{siunitx}
    \usepackage{caption}
    \newcommand{\diff}[2][]{\mathop{}\!\mathrm{d}^{#1} {#2} \,}
    \newcommand{\given}[1][]{\, {#1} \vert \,}
    ```
abstract: | 
    | Since its inception in 2015, variational autoencoders have emerged as a versatile machine learning tool to treat complex distributions for producing generative models, dimensionality reduction, and denoising. This project aims to enhance its structure with the aid of the discrete wavelet transform coefficients as training features to accelerate learning and improve latent space structure. The cascading structure of the multi-level wavelet decomposition allows us to further restrict the information bottle-neck of the autoencoder structure by minimising mixing across levels.  Using the NSynth dataset as an example, we compare results obtained from the modified variational autoencoder structure trained on the discrete wavelet transform, to that of an unmodified one trained on the discrete Fourier transform; taking into account the quality of the reconstruction as well as the latent space structure as heuristics for overall performance. The framework we have developed to combine the discrete wavelet transform with a variational autoencoder also provides additional flexibility to tune a variational autoencoder to the specific dataset by choosing the most suitable wavelet; thus it can be recalibrated to accommodate a wide range of tasks where wavelet analysis may be suitable.
---


# Introduction

Over the past decade, deep learning has seen an unprecedented boom in popularity, providing us with an array of powerful tools for use in areas such as image recognition [@krizhevsky], text generation, and playing go [@silver2017] all tasks that have stumped traditional tree and filter based methods. Powered by artificial neural networks, these algorithms provide novel methods for analyses in data abundant fields.

Physics research has benefited tremendously from these techniques, from star-galaxy classification and gravitational lens identification [@kim2016star; @pourrahmani2018lensflow] in astrophysics to data reduction in high-energy physics at the LHC [@guest2018deep]. In quantum theory, neural networks are used to tackle the quantum many-body problem in condensed matter physics [@carleo2017], quantum state tomography [@carleo2018] and more [@dunjko2018]. Clearly demonstrating the versatility of deep learning and its many incarnations in physics alone. 

Unsupervised learning is a subset of machine learning that attempts to infer useful information without requiring labelled data. Neural networks have shown to be effective at tackling such tasks as they demonstrate an unparalleled agility in mapping complex relationships. To paraphrase, neural networks apply non-linear transformations to data in a series of layers---coined deep learning---which when combined have shown to accurately perform a wide variety of tasks.

Of these techniques, variational autoencoders (VAEs) have shown to be reliable at dimensionality reduction tasks, with the added benefit of producing well structured latent spaces [@kingma]. They rely on a Bayesian probabilistic model of the data and attempts to learn its mapping into a latent space by approximating the posterior distribution. The reverse mapping, the decoder, is then a robust generator capable of producing unseen data with features specified by sampling the latent space. Such reduction and generative properties have been leveraged to produce models for data in a wide range of tasks. Moreover, the robustness of these networks also proves useful for auxiliary tasks such as noise reduction and pre-training for other machine learning tasks.

The training of neural networks can be aided by various preprocessing methods. There exist many ways of abstracting better features from raw data, ranging from Fourier transforms to statistical methods. This project will focus on the use of the discrete wavelet transform (DWT) which has already seen applications in data compression [@jpeg2002] and denoising [@taswell2000]. As such the DWT possesses similar properties to the variational autoencoder. We investigate the use of a signal's discrete wavelet decompositions as features to train a variational encoder, which also conveniently allows us to simplify the neural networks by treating different levels of wavelet coefficients separately. This project attempts to build an unsupervised model for feature extraction using the variational autoencoder when applied to audio from the NSynth dataset. Three approaches are contrasted:

- The audio signal is fed directly into the variational autoencoder without preprocessing, 
- the audio signal is first preprocessed using the discrete Fourier transform,
- the audio signal is first preprocessed using the discrete wavelet transform with a modified VAE structure.

Through these toy examples, I hope to demonstrate that when combined with a variational autoencoder, the discrete wavelet transform produces a convincing model for both feature extraction and generative purposes. The methodology developed in this project is very general and can in principle be applied to any data where a wavelet approach is desired; we could also consider modifying the structure of the neural networks to incorporate more hidden layers, or use convolutional layers in place of fully connected ones.

An outline of the rest of the dissertation is as follows:

- Chapter 2 outlines the related work in the use of autoencoders and variational autoencoders, and the use of the wavelet transform in handling audio data.
- Chapter 3 gives a brief overview of the theory behind neural networks, variational autoencoders, and wavelet transforms.
- Chapter 4 discusses the methodology used when implementing the variational autoencoder and suggests suitable metrics to compare the different approaches.
- Chapter 5 presents and evaluates the results from the tests performed.
- Chapter 6 and 7 give an overall conclusion of the whole project and suggests possible areas for future work.

# Related Work

As a comprehensive review of the literature behind neural networks will require nuanced discussions about a range of issues that are far beyond the scope of this project, I will only outline major milestones in the development of autoencoders and variational autoencoders in this section and point the reader to literature that cover the fundamentals of neural networks. This is then followed by some relevant literature on discrete wavelet transforms. 

For a concise review of deep learning in its modern form, see [@lecun2015]; a more in-depth 30 page overview is also given by [@schmidhuber] in which he also outlines the history and development of autoencoders.

## Autoencoders and variational autoencoders

While it is difficult to pinpoint when autoencoders were first conceptualized, their application to neural networks can be dated back to [@ballard]. Their use in unsupervised learning was not mainstream until deep learning became feasible in the past decade [@bengio2007], spawning multiple varieties of autoencoder architectures that attempt to learn better representations, including the variational flavour that will be a focus of this project. Leveraging their generative properties, autoencoders have been applied to image modelling with convolutional neural networks [@masci2011; @krizhevsky; @pu2016], image compression [@cheng2018], word generation with recurrent neural networks [@mikolov; @socher2011] and much more. 

To further improve an autoencoder's ability to learn meaningful representations, additional constraints can be enforced. The de-noising autoencoder, first introduced in [@vincent], is a simple extension, where the network is trained to reconstruct partially corrupted ('noisy') data. A more nuanced approach comes in the form of the sparse autoencoder (SAE) that forces activations in hidden layers to be close to zero during training [@poultney2007efficient]. Most commonly, these robust autoencoders have seen use to pre-train other deep neural networks to find better initial weights.

Moving forward to 2013, the variational auto-encoder was introduced in [@kingma]; based on a Bayesian learning model; it pioneered the idea of approximating posterior distributions between some data and its latent representation by variational parameters (a neural networks). Since then, variational autoencoders have been used in a wide array of fields from learning music [@roberts], to predicting movement from static images [@walker2016].

In high energy physics, we find the use of unsupervised learning on deep autoencoders used in identifying particles within benchmark datasets, which improved on conventional methods by 8% [@baldi]. Due to the shear volume of data that is collected at the LHC, neural networks have become crucial in data reduction algorithms [@guest2018deep].

## Wavelet transform

A comprehensive guide to the 'Theory of Wavelets' can be found in [@meyer2011]. Unlike their more well-known counterpart, Fourier transforms, wavelet transforms offer both spatial and frequency resolution. They have seen many applications in signal coding and data compression, most notably the JPEG 2000 image compression standard [@jpeg2002]. Machine learning methods are also incorporate wavelet transforms to preprocess data with complex spatial and frequency dependencies like electroencephalograms [@amin2015]. 

There have been numerous studies on the classification of sounds both with [@han2017] and without [@herrera2002] neural networks. A more meta analysis on directly learning wavelet filters using autoencoders was proposed recently [@recoskie2018]. 

The use of wavelet analysis in deep learning is also a very active area of research. Most notably, the wavelet network (WN) [@zhang1992wavelet; @alexandridis2013wavelet] has been presented as an alternative to traditional neural networks and convolutional neural networks. The added degree of freedom introduced by the wavelet choice allows the initial network to be closer to the true global minimum, accelerating the training process.

More recently, wavelet analysis has also been used in conjunction with autoencoders to pre-train a neural network for music recognition [@klec2015pre]. However, due to the novelty of variational autoencoders, to the best of my knowledge the use of wavelet features in training VAEs has not been widely researched.

# Theory

## Artificial neural networks

Inspired by biological neural networks, an artificial neural network (or simply neural networks) is a generalized term for a series of nodes (or neurons) interconnected via edges (synapses). These nodes can be arranged in feedforward layers which transmit information from one layer of neurons to the next without looping back on itself.

A multilayer perceptron (MLP) is the archetypal feedforward networks where node $i$ in layer $l$ computes its activation, $a_i^{(l)}$, as a weighted sum of all the nodes in preceding layer. In practice each node is also given a bias, $b^{(l)}_i$, and a transfer function, $f$, which help regularize the training process. The activations can be iterative form:

\begin{equation}
    a_i^{(l)} = f\left(\sum_j w^{(l)}_{ij} a_j^{(l-1)} + b^{(l)}_i\right)
\end{equation}

where $w^{(l)}_{ij}$ represents the weight between the $i$th and $j$th node in layers $l$ and $l-1$ respectively, layers are said to be fully connected if $w_{ij}$ is a dense matrix (i.e.\ each node receives contributions from every node in the previous layer). These activations are then fed on to the next layer, repeating until the output layer is reached. The most common choices for the activation function are the rectified linear unit (ReLU), $f(x) = \mathrm{max}(0, x)$ and the sigmoid, $f(x) = (1+\exp{(-x)})^{-1}$.

\begin{figure}[h]
\captionsetup{width=0.9\textwidth}
\centering
\vspace{-10pt}
{\includegraphics[width=4.2in]{NN.pdf}}
\vspace{-10pt}
\caption{A typical example of an MLP with 1 hidden layer used to classify images of handwritten digits. The colour of each node indicates its type: blue for input, orange for hidden, and green for output. The dotted lines between each layer represents the weight matrices $w_{ij}$.}
\label{fig:NN}
\end{figure}

The weights and biases are parameters that can be varied to allow the network to 'learn' through training on examples. Given a cost function (e.g. the accuracy of a classifier), we can apply the backpropagation algorithm and perform stochastic gradient descent to minimize said cost and gradually improve our network. Refer to the 'Deep Learning' book for a more detailed explanation [@goodfellow2016]. 

Machine learning tasks can be broadly categorised into two regimes: supervised learning, where the network is trained on labelled data; and Unsupervised learning, where a network is required to learn features about the data without being given explicit labels, usually in the form of a mapping from the input data into a lower-dimensional feature (latent) space. Depending on the implementation, a reverse mapping can also be learned to create a generative model which is capable of producing unseen data by sampling the feature space.

## Autoencoders and variational autoencoders

An autoencoder is a type of neural network that learns to reduce and reconstruct data in an unsupervised manner. The autoencoder is separated into 2 components, an encoder that maps data vectors in input space to feature vectors in the latent space, and a decoder that does the opposite. The encoder/decoder pair is trained simultaneously such that the reconstruction loss between the input into the encoder and the output of the decoder is minimized. Alternatively, they can be framed as an atypical MLP where the size of one of the intermediate hidden layers has fewer dimensions than the input or output layers, forming a bottleneck in the information pipeline through the network.

The following paragraphs paraphrase the theory behind variational autoencoders and follows their original formulation in [@kingma]. The concept of a variational autoencoder is framed in the context of Bayesian statistics. We define a parametrized distribution, $p_\theta (\mathbf{z})$, to a set of latent variables, $\mathbf{z}$, and a generative model for some data $\mathbf{x}$ by sampling the parameterized conditional distribution $p_\theta (\mathbf{x} \given \mathbf{z})$. One could imagine optimizing these true parameters $\theta$ by maximizing the likelihood $p_\theta(\mathbf{x})$. However, computing this marginal likelihood involves computing the generally intractable $N$ dimensional integral: 

\begin{equation}
  p_\theta(\mathbf{x})=\int \diff{\mathbf{z}}p_\theta (\mathbf{x} \given \mathbf{z})p_\theta(\mathbf{x})  
\end{equation}

over the whole space of $\mathbf{z}$, scaling exponentially with its dimension. Furthermore, we are also interested the inference of data, for which we require the posterior, $p_\theta (\mathbf{z} \given \mathbf{x})$. We might attempt to calculate this using Bayes theorem

\begin{equation}
    p_\theta (\mathbf{z} \given \mathbf{x}) = \frac{p_\theta (\mathbf{x}\given\mathbf{z})p_\theta(\mathbf{z})}{p_\theta(\mathbf{x})}.
\end{equation}

However, as the marginal likelihood appears again in the denominator, this is again an intractable problem. While the marginalization integral can theoretically be approximated with Monte Carlo methods, it is far from optimal for learning due to the high computational overheads. 

Thus we turn to variational inference to approximate the posterior with a family of distributions $q_\phi(\mathbf{z} \given \mathbf{x})$ and reframes inference into an optimization problem by varying the parameters $\phi$. Defining the prior distribution of the latent variables to be a unit Gaussian,  $p_\theta(\mathbf{z})=\mathcal{N}(\mathbf{z} \given \mathbf{0}, \mathbf{1})$, and our generative distribution $q$ to be a class of diagonal Gaussian distributions; the local variational parameters for each data point $\mathbf{x_n}$, is just the mean and variance, namely $\phi_n = \{\mu_n, \sigma_n^2\}$. However, it is impractical and unnecessary to learn separate distributions for each data point, these can instead be approximated by way of a global variational parameter $\phi$ which allows us to infer the mean, $\mu_\phi(\mathbf{x_n})$ and variance $\sigma_\phi(\mathbf{x_n})$ directly from data. 

\begin{equation}
    q_\phi (\mathbf{z}_n\given \mathbf{x}_n) = \mathcal {N}(\mathbf{z}_n \given \mu_\phi(\mathbf{x_n}), \mathbf{1}\cdot\sigma_\phi^2(\mathbf{x_n}))
\end{equation}

We now have a clear path to use the variational inference model $q_\phi(\mathbf{z}_n\given \mathbf{x}_n)$ to approximate the true posterior $p_\phi(\mathbf{z}_n\given \mathbf{x}_n)$. The Kullbeck-Leibler divergence, a measure of the relative entropy in the two distributions, is a convenient method of measuring the fitness of such a reconstruction defined as:

\begin{equation}
    D_\mathrm{KL}\left[(q_\phi(x) \, \vert\vert\,  p_\theta(x))\right] = - \sum_x q_\phi(x)\log p_\theta(x) + \sum_x q_\phi(x)\log q_\phi(x)
\end{equation}

where we have denoted $p_\theta(\mathbf{z} \given \mathbf{x})$ and $q_\phi(\mathbf{z} \given \mathbf{x})$ as $p(x)$ and $q(x)$ for brevity. It turns out that minimising the KL divergence is equivalent to maximising the evidence lower bound (ELBO) function:
\begin{equation}
    \mathrm{ELBO}(\phi, \theta) = \mathrm{E}_q\left[\log p_\theta(\mathbf{x} \given \mathbf{z})\right] - D_\mathrm{KL}\left[q_\phi(\mathbf{z} \given \mathbf{x})\, \vert\vert \, p_\theta(\mathrm{z})\right]
\end{equation}

which is convenient as the ELBO is a computationally much cheaper. The first term on the RHS is the expectation value of the reconstruction distribution w.r.t.\ $q$, and is a measure of the reconstruction loss of the generator. The second KL divergence term, also know as the latent loss, is a measure of structure of the latent space compared to the defined $\mathrm{z}$ distribution.

We can now frame the whole structure as two neural network structures.

\begin{figure}[H]
\vspace{-10pt}
  \centering
	\captionsetup{width=0.95\textwidth}
    \includegraphics[width=4.2in]{VAE_6.pdf} 
  \caption{Diagrammatic representations of a variational autoencoder (right) each with one hidden layer in the encoder and decoder. Here pink squares represent the components of the latent space. The encoder models \(q_{\phi} (\mathbf{z}\given\mathbf{x})\) which approximates the true posterior \(p_{\theta} (\mathbf{z}\given\mathbf{x})\). The decoder models the generator \(p_{\theta} (\mathbf{x}\given\mathbf{z})\). Reparameterization is represented by the arrow between the 2 latent payers.}
  \label{fig:VAEfig}
\vspace{-10pt}
\end{figure}

- An encoder that infers a distribution $q_{\phi} (\mathbf{z}\given\mathbf{x})$ when given some data $\mathbf{x}$ by specifying the mean and distribution of a normal distribution: $\mu_\phi(\mathbf{x})$ and $\sigma_\phi(\mathbf{x})$. The variational parameters $\phi$ are thus the weights and biases of the encoder network.
- A decoder that generates data $\mathbf{x}'$ by sampling the distribution $p_\theta(\mathbf{x}'\given\mathbf{z})$. The model parameters $\theta$ are thus the weights and biases of the decoder network.

To find the optimal parameters $\phi$ and $\theta$, the variational autoencoder is trained to minimize the ELBO, through the typical backpropagation/stochastic gradient descent methods with a reparameterization trick applied to sample the latent space.

## Wavelet transform

The wavelet transform, like the Fourier transform, is a decomposition of a function into a different basis; however wavelet transforms are able to retain information about both the location as well as frequency. Here we will focus on the discrete wavelet transform that (akin to the fast Fourier transform) scales linearly in time. For more detailed guide to wavelet theory, consult [@meyer2011]. 

### Discrete wavelet transform {-}

For an orthonormal mother wavelet $\psi(t)$, we shift and rescale it by powers of 2 to obtain a set of child wavelets $\psi_{m,n}(t)$

\begin{equation}
    \psi_{m,n}(t) = \frac{1}{\sqrt{2^m}}\psi\left(\frac{t-n2^m}{2^m}\right)
\end{equation}

A set of discrete high and low pass filters, $h[n]$ and $g[n]$, can then be constructed from these child wavelets and convoluted with the signal, $x[n]$, to produce a series of wavelet coefficients.

\begin{equation}
    y_{\mathrm{high}}[n] = \sum_{k} x[k]h[2n-k] \qquad \text{and} \qquad y_{\mathrm{low}}[n] = \sum_{k} x[k]g[2n-k] 
    \label{eq:DWT}
\end{equation}

This doubles the frequency resolution as each part contains only half the frequencies of the original sample, but halves the spatial resolution (by Nyquist's rule). Thus each of the decomposed signals is further decimated giving the factors of 2 in equation \ref{eq:DWT}. $y_\mathrm{low}$ is known as the approximation coefficients and $y_\mathrm{high}$ the detail coefficients. This process can then be repeated iteratively on $y_\mathrm{low}$ to produce a full discrete wavelet decomposition. 

\begin{equation}
    y^{(i+1)}_{\mathrm{high}}[n] = \sum_{k} y^{(i)}_{\mathrm{low}}[k]h[2n-k] \qquad \text{and} \qquad 
    y^{(i+1)}_{\mathrm{low}}[n] = \sum_{k} y^{(i)}_{\mathrm{low}}[k]g[2n-k] 
    \label{eq:DWT}
\end{equation}

where $y^{(i)}_{\mathrm{high}}$ will be the discrete wavelet decomposition coefficients at the $i$th level. These coefficients can then be square and rescaled to the time axis to form a 'scalogram' to show their magnitudes in time and frequency space. A proper choice of $h$ and $g$, will allow the DWT to be inverted.

The DWT is also utilised for the fast compression of real time signals in preparation for analyses due to its low computational requirements [@quotb2011wavelet].


## Fourier transform

For a comprehensive guide to Fourier transforms and the discrete Fourier transform, see [@bracewell1999].

# Methods

In this section I present the methods used to implement the variational autoencoder architecture and how it is incorporated with the discrete wavelet transform. A brief overview of the input data is included at the end.

## Variational autoencoder implementation

The VAE was built and trained using a *Tensorflow* 1.13.1 backend [@tensorflow2015] with its Python API. All data visualization and analysis were done with the *NumPy*, *Scikit-learn*, and *Matplotlib* libraries. The training of the models along with all subsequent testing were performed on my personal computer, which runs on a $\SI{2.6}{\giga\hertz}$ Intel Core i7 Processor with $\SI{16}{\giga\byte}$ of memory. The training of each VAE took ~6 hours each with the following set of hyperparameters.

### Hyperparameter selection {-}

The design and implementation of the variational autoencoder remains largely unchanged from that described in the Theory section; the neural networks used in the encoder and decoder have mirrored layer structures. The final values chosen for the variational autoencoder is summarized below:

\begin{table}[H] \captionsetup{width=0.9\textwidth}
\centering
\begin{tabular}{c|c}
Hyperparameter    & Value               \\ \hline \hline
Hidden Layers     & FC(2000), FC(1000)  \\ 
Latent dimension  & 64                  \\
Batch size        & 50                  \\ 
Batches per epoch & 400                 \\ 
Epochs            & 40                  \\ 
Optimizer         & Adam                \\ 
Learning rate     & $10^{-4}$           \\ 
$\beta$ sacling   & $0.1$               \\ 
\end{tabular}
\caption{Hyperparameters selected for the variational autoencoder. The fully connected layers are identified by the number of nodes they contain, i.e.\ FC(2000) indicates a fully connected layer with 2000 nodes.}
\vspace{-10pt}
\label{tab:hyper}
\end{table}

Adhering to most common practices, the learning rate of the Adam optimizer [@kingma2014] was set to $10^{-4}$.

The architecture was further modified to incorporate the structure of the discrete wavelet transform such that the first hidden layer is trained on each level of coefficients separately (effectively turning the weight matrix into block diagonal form with each block corresponding to the different levels of coefficients). This drastically reduced the number of unused weights, improving information bottle-neck and accelerating the training process.

The latent dimensionality, $D_z$ was chosen such that the intrinsic dimensions of the data would be accounted for without costing too much in the way of computational complexity. For many cases where $D_z>D_\mathrm{data}$, further increases in $D_z$ rarely yield better models and instead often lead to vanishing gradients and/or overfitting.

### Regularization {-}

A few measures were taken to regularize and aid the training process:

**Batch normalization** 

Occasionally, certain weights in the graph would diverge leading to large discrepancies in the activations of hidden layers, effectively 'drowning out' all other nodes, known as internal covariant shift. A solution to this was introduced in [@ioffe2015batch] which involves normalising the input to each hidden layer, known as Batch Normalization and has been shown to both increase the rate of training, and increased the chance of reaching the global minimum. Below is a brief summary of its implementation.

If layer $l$, with $m$ nodes, is to be batch normalized for mini-batch $\mathcal{B}$, its outputs, $\{a_1^{(l)},  \dots , a_m^{(l)}\}$, are first normalized to $\{\tilde{a}_1^{(l)},  \dots , \tilde{a}_m^{(l)}\}$ with 0 mean and unit variance by:

\begin{equation}
   \tilde{a}_i^{(l)} = \frac{x_i^{(l)} - \mu_\mathcal{B}}{\sqrt{\sigma^2_{\mathcal{B}} + \epsilon}},
\end{equation}

where $\epsilon$ is some small constant added for numerical stability. $\mu_\mathcal{B}$ and $\sigma^2_{\mathcal{B}}$ are the mean and variance of the mini-batch respectively, computed as:

\begin{equation}
    \mu_\mathcal{B} = \frac{1}{m}\sum_{i=1}^{m}a_i^{(l)} \qquad \text{and} \qquad \sigma^2_{\mathcal{B}} = \frac{1}{m}\sum_{i=1}^{m}(a_i^{(l)})^2 - \mu_\mathcal{B} ^2. 
\end{equation}

The inputs are then scaled and shifted to remove the restrictions on the normalized batch's mean and variance by introducing parameters $\gamma_l$ and $\beta_l$ such that the outputs, $\{x_1^{(l)},  \dots , x_m^{(l)}\}$, are

\begin{equation}
    y_i^{(l)} = \gamma_l \tilde{x}_i^{(l)} + \beta_l
\end{equation}

These 2 additional parameters are treated in the same way as the weights and biases, and are be optimized through backpropagation and SGD.

**$\beta$-Variational autoencoder and Xavier initialization**  

During training, the model would often reach a stable point where the KL divergence term in the cost function would collapse within a few epochs, rendering most of the latent structure unused. A solution to this problem proposed in [@higgins2017] involves weighing the KL divergence with respect to the reconstruction loss by some factor $\beta$. Intuitively, for some $\beta < 1$, better reconstruction is encouraged at the cost of latent structure.

Furthermore, the initialization of weights in the networks also proved to be crucial, and adhering to common practices, the Xavier initialization method [@glorot2010] was chosen. Weights, $w^{(l)}_{ij}$, are initialized with a random value in the uniform range

\begin{equation}
    U \left[-\frac{\sqrt{6}}{\sqrt{m_l + m_{l+1}}}, \frac{\sqrt{6}}{\sqrt{m_l + m_{l+1}}}\right]
\end{equation}

where $m_l$ represents the number of nodes in layer $l$

## Discrete wavelet transform

\begin{figure}[H]
    \vspace{-10pt}
    \centering
	\captionsetup{width=0.95\textwidth}
    \includegraphics[width=4.2in]{db6.pdf} 
    \caption{A construction of the 2 db6 wavelets, the scaling function is the low pass filter, $h[n]$, and the wavelet function is the high pass filter, $g[n]$, as per the theory section.}
    \vspace{-10pt}
\label{fig:db6}
\end{figure}

The discrete wavelet transform was handled by the Python package *PyWavelets*. For the purposes of this project, we chose a wavelet from the Daubechies family, specifically the db6 wavelet (figure \ref{fig:db6}. This choice was motivated by a previous study on the effectiveness of wavelet transforms for the classification of percussive sounds [@daniels2010classification].

\begin{figure}[h]
    \vspace{-10pt}
    \centering
	\captionsetup{width=0.95\textwidth}
    \includegraphics[width=4.7in]{scalogram.pdf} 
    \caption{3 plots showing the same signal represented in 3 different ways showing representing the signal: (1) in temporal space (top); (2) as a scalogram showing the magnitude of coefficients in wavelet wavelet space (bottom left); (3) a spectrogram showing the magnitude of the windowed Fourier transforms (bottom right).}
\label{fig:scalo}
\end{figure}

Figure \ref{fig:scalo} shows both the wavelet and Fourier decompositions (also known as a scalogram and spectrogram respectively) of one of the signals from the *NSynth* database. Whereas the spectrogram might be a more intuitive representation of the signal, values in wavelet space are often more sparsely distributed, making for better features to train a neural network on.

## Training data \label{training-data}

For the purposes of this project, the variational autoencoders were trained on audio data from the *NSynth* dataset, introduced in [@nsynth2017] which contains 305,979 unique musical notes, recorded from 1006 instruments. Out of them, I chose to train my models on the subset of notes with the 'distortion' quality, which makes up 17\% of the whole dataset. The raw `tfrecord` file was around $\SI{70}{\giga\byte}$ in total, and when converted to the 16-bit PCM WAV audio format, the audio data was about $\SI{23}{\giga\byte}$. 

The `audio` feature of each example is a list of floating points values in the range $[-1, 1]$. Each 4 second clip is truncated to include only the first 0.5 seconds, as the majority of samples were short and decayed substantially within that time-frame. The samples are finally decimated by a factor of $2$ (i.e.\ only every 2nd sample was kept). Overall, this reduced the size of the input space to $4000$ samples. 

An overall normalization was required to bring the samples into the $[0, 1]$ range. The obvious choice of rescaling and shifting was tested; however this centered all samples at $0.5$ which proved to be problematic during training. This is thought to be brought about by the sigmoid function applied to the output (for the Bernoulli distribution), which for values of $0.5$ would encourage a uniform output of zeros corresponding to vanishing weighs and biases. Hence a choice was made to simply take the square values of the audio samples and its transforms to bring them into the desired range.

# Results and Discussion

For each of the following sections, the results for the 3 different variational autoencoders will be presented and then compared:

1) The naive approach where the VAE is trained on the raw intensity signal,
2) the benchmark approach where the VAE is trained on the discrete Fourier transform of the full signal,
3) the modified VAE structure trained on the wavelet decomposition of the signal.

We will treat the discrete Fourier approach as the benchmark to compare the performance of our modified VAE. Note that all signal plots will correspond to the square signal as per the discussion in the Methods section about squaring waveforms for the sake of normalization.

## Reconstruction quality

We first compare the quality of the reconstruction for variational autoencoders. The reconstruction and latent losses were logged during training to keep track of progress, after all 40 epochs, the approximate losses for the three VAEs were as follows:

\begin{table}[H] \captionsetup{width=0.9\textwidth}
\centering
\begin{tabular}{c|c c c}
           & Reconstruction Loss & Latent Loss & Total Loss (weighted) \\ \hline \hline
Raw Signal & 1370                & 50          & 1370                  \\
DFT signal & 130                 & 10          & 130                   \\
DWT signal & 400                 & 10          & 400                  
\end{tabular}
\caption{Losses for the 3 different VAE types after 40 epochs of training. These values are all accurate to within $\pm 10$.}
\label{tab:hyper}
\end{table}

Since they were all trained using the same hyperparameters, the direct comparison of the losses between these networks is somewhat justified. It is clear that the loss during training can be significantly reduced by the use of the DWT and DFT as inputs. Furthermore, the bulk of the difference in total losses is caused by the disparities in reconstruction losses. For a more detailed examination of how the reconstructions differ, see plots below. N.B.\ the figures are based on samples taken at random from the testing dataset, and are reconstructions of different inputs.

\begin{figure}[H]
    \vspace{-10pt}
    \centering
	\captionsetup{width=0.95\textwidth}
    \includegraphics[width=4.7in]{recon.pdf} 
    \vspace{-10pt}
    \caption{A test input intensity signal (left), contrasted with its reconstruction on the right by the VAE.}
    \vspace{-10pt}
\label{fig:recon}
\end{figure}

Starting with the raw signal, it is apparent that while the reconstruction manages to capture the intensity envelope, it fails at reconstructing any of the finer details. The reconstruction is also somewhat reminiscent of a moving average or a low pass filter of the original signal, suggesting that this network is more sensitive to the low frequency components.

\begin{figure}[H]
    \vspace{-10pt}
    \centering
	\captionsetup{width=0.95\textwidth}
    \includegraphics[width=4.7in]{reconDFT.pdf} 
    \vspace{-10pt}
    \caption{A test input frequency spectrum (left), contrasted with its reconstruction on the right by the VAE. The frequency is plotted in arbitrary units.}
    \vspace{-10pt}
\label{fig:reconDFT}
\end{figure}

\begin{figure}[H]
    \vspace{-10pt}
    \centering
	\captionsetup{width=0.95\textwidth}
    \includegraphics[width=4.7in]{reconDWT_scalo.pdf} 
    \vspace{-10pt}
    \caption{A test input wavelet spectrum signal (top left) and its scalogram (bottom left) is plotted on the left, contrasted with its reconstruction on the right.}
    \vspace{-10pt}
\label{fig:reconDWT_scalo}
\end{figure}

In stark contrast to figure \ref{fig:recon}, figure \ref{fig:reconDFT} clearly shows not only has this network learned to recreate the broad strokes of the signal, but also the finer details in the form of overtones.

Visual inspection of the decomposition levels in figure \ref{fig:reconDWT_scalo} suggests that while the reconstruction retains the overall envelope of intensity distributions, it fails to accurately reproduce the finer details within. This echos observations from the raw input signal.

We can attribute the superior reconstruction quality of the DFT signal its sparsity in input space, especially in the context of musical notes where each audio sample has well defined frequency components. 

## Latent structure

\begin{figure}[h]
\vspace{-10pt}
	\captionsetup{width=0.95\textwidth}
    \subfloat[Raw signal]{ 
    \includegraphics[width=6in]{latent.pdf} \label{subfig:latent} } \\
    \subfloat[DFT signal]{
    \includegraphics[width=6in]{latentDFT.pdf} \label{subfig:latentDFT}}\\
    \subfloat[DWT with altered VAE]{
    \includegraphics[width=6in]{latentDWT.pdf} \label{subfig:lantentDWT}}
    \caption{Plots showing the structure of data points from the validation dataset in the their respective latent spaces learned by the VAEs. These correspond to the raw intensity signal (top), DFT intensity signal (middle), DWT intensity signal (bottom). Data points are classified by 3 different classes features, their instrument family (left column), note qualities (center column) and their pitch (right column). Points possessing only the 'distortion' quality is omitted in the central column for clarity.}
\label{fig:latent}
\end{figure}

However, the quality of reconstructions only gives half of the story. The other contribution to loss comes in the form of latent structure, measure by the KL-divergence term in the overall loss function. Working again from the testing dataset, we first embedded the 64 dimensional latent spaces into a more manageable 2 dimensions using the t-SNE algorithm, designed for dimensionality reduction [@maaten2008visualizing], as shown in figure \ref{fig:latent}. Each point in these plots correspond to an audio sample in the test dataset, coloured according to their features. For a good latent representation, we expect the points to be well distributed about the origin, whilst also tightly clustering data points that are similar, and segregating dissimilar ones. While convenient for visualisation purposes, it is still important to keep in mind that the 2D embedding will still be far from an ideal representation of the 64 dimensional latent spaces.

Perhaps unexpectedly, the structure of the DFT latent space is clearly least structured; little to no cluster is observed aside from the overall grouping around the origin. Moreover similar points do no appear to be grouped particularly well, most evident in the central plot that shows a large regions of space occupied by points of different note qualities. This could be attributed to the $\beta=0.1$ weight applied to the latent cost. As mentioned in the Methods section, this effectively incentivize better reconstruction at the cost of latent structure. Ideally, we would also investigate the effectiveness of gradually increasing $\beta$ during training to weaken its penalty on latent structure. 

The most apparent difference between the 3 latent structures is the degree of local clustering. Whereas points in the DWT latent space are very tightly packed, the same cannot be said for the DFT. The raw input signal strikes an interesting balance between the two with a few tightly clustered regions located around the central unit despite the aforementioned poor reconstruction quality. The latent space representation of the data-points in the DWT latent space also gives the most reasonable segregation between dissimilar features (the 'note qualities' feature in particular). Working from this latent representation, one could easily apply a simple machine learning technique (something like k-nearest neighbours perhaps) to classify points, demonstrating that the network has clearly fulfilled its purpose of unsupervised learning. The same case can be made for the representation of pitches in its latent space where---bar a few anomalies---the pitches follow a smooth gradient from low pitches in the top right, to high pitches in the bottom left. 

In order for the VAE to be used as a useful generative model, both reconstruction quality and latent space structure are vital. Without reconstruction quality, the generator could never reconstruct clean, meaningful signals, and without latent structure, we would have poor control over the samples that we do generate.

\begin{figure}[h]
    \vspace{-10pt}
    \centering
	\captionsetup{width=0.95\textwidth}
    \includegraphics[width=6in]{latenttSNE.pdf} 
    \vspace{-10pt}
    \caption{Data mapped onto a 2D latent space using primary component analysis (PCA) for the raw data signal (top) and the wavelet decomposed signal (bottom) for 3 different classes of features, their instrument family (left column), note qualities (center column) and their pitch (right column). Points possessing only the 'distortion' quality is omitted in the central column for clarity.}
    \vspace{-10pt}
\label{fig:latentPCA}
\end{figure}

Figure \ref{fig:latentPCA} compares the structure of the 2D latent spaces learnt by primary component analysis (PCA, a linear dimensionality reduction technique) when applied to the raw signal and the wavelet decomposed signal. A marked improvement in the wavelet version strongly suggests that the DWT is indeed a helpful tool for priming data before learning.

# Conclusions

As a toy example, we have shown that the use of a variational autoencoder aided by discrete wavelet transforms are able to learn meaning representations of the audio samples within the *NSynth* dataset. With the modified 'block diagonal' VAE structure, we were able to drastically reduce the computational cost of training without sacrificing too much in terms of reconstruction quality or latent structure. 

We hope that these results will encourage further research into the use of alternative VAE structures for unsupervised learning in a broader context, be it for compression/dimensionality reduction, or generative purposes. Though most of the comparisons made are qualitative, they have verified that the VAE structure is indeed capable of capturing complex non-linear relationships between the data.

While further work is needed to improve the reconstruction quality of the model, its comparison with the discrete Fourier transform shows that the wavelet transforms is a viable candidate for treating signals, and in particular those with both frequency and temporal complexity. With the added degree of freedom granted by the choice of wavelets, this model could also be highly adaptable for a wider range of inputs.

# Further Work

Further testing of the current model could vary the hyperparameters and initialization methods to potentially overcome problems with stagnating costs and improve training speeds. Alternatively, the objective function can be modified to maximize the mutual information between the input and its latent representation (InfoVAE) to improve the quality of the variational posterior [@zhao2017infovae]. Sparsity could also be gradually introduced by deactivating weights over the training process to improve the information bottleneck. 

While the code developed for this project is somewhat specific to the task of learning on audio data, it can easily be extended to incorporate a wide variety of data, timeseries or otherwise. The applications of wavelet transforms to preprocessing variational autoencoder data is clearly much wider than the context of our toy example used in this project; there is no better way to discover these applications other than to directly apply it to real-world problems.

We hope that this work encourages more interest in the use of wavelet transforms in tandem with VAEs to perform generative and dimensionality reduction tasks.

# References

