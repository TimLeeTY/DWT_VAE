# Abstract

Since its inception in 2015, variational autoencoders have emerged as a
versatile machine learning tool to treat complex distributions for
producing generative models, dimensionality reduction, and denoising.
This project aims to enhance its structure with the aid of the discrete
wavelet transform coefficients as training features to accelerate
learning and improve latent space structure. The cascading structure of
the multi-level wavelet decomposition allows us to further restrict the
information bottle-neck of the autoencoder structure by minimising
mixing across levels. Using the NSynth dataset as an example, we compare
results obtained from the modified variational autoencoder structure
trained on the discrete wavelet transform, to that of an unmodified one
trained on the discrete Fourier transform; taking into account the
quality of the reconstruction as well as the latent space structure as
heuristics for overall performance. The framework we have developed to
combine the discrete wavelet transform with a variational autoencoder
also provides additional flexibility to tune a variational autoencoder
to the specific dataset by choosing the most suitable wavelet; thus it
can be recalibrated to accommodate a wide range of tasks where wavelet
analysis may be suitable.

# Introduction

Over the past decade, deep learning has seen an unprecedented boom in pop- ularity, providing us with an array of powerful tools for use in areas such as image recognition (Krizhevsky, Sutskever, and Hinton 2012), text generation, and playing go (Silver et al. 2017) all tasks that have stumped traditional tree and filter based methods. Powered by artificial neural networks, these algorithms provide novel methods for analyses in data abundant fields.

Physics research has benefited tremendously from these techniques, from star- galaxy classification and gravitational lens identification (Kim and Brunner 2016; Pourrahmani, Nayyeri, and Cooray 2018) in astrophysics to data reduction in high-energy physics at the LHC (Guest, Cranmer, and Whiteson 2018). In quantum theory, neural networks are used to tackle the quantum many-body problem in condensed matter physics (Carleo and Troyer 2017), quantum state tomography (Torlai et al. 2018) and more (Dunjko and Briegel 2018). Clearly demonstrating the versatility of deep learning and its many incarnations in physics alone.

Unsupervised learning is a subset of machine learning that attempts to infer useful information without requiring labelled data. Neural networks have shown to be effective at tackling such tasks as they demonstrate an unparalleled agility in mapping complex relationships. To paraphrase, neural networks apply non- linear transformations to data in a series of layers—coined deep learning—which when combined have shown to accurately perform a wide variety of tasks.

Of these techniques, variational autoencoders (VAEs) have shown to be reliable at dimensionality reduction tasks, with the added benefit of producing well structured latent spaces (Kingma and Welling 2013). They rely on a Bayesian probabilistic model of the data and attempts to learn its mapping into a latent space by approximating the posterior distribution. The reverse mapping, the decoder, is then a robust generator capable of producing unseen data with features specified by sampling the latent space. Such reduction and generative properties have been leveraged to produce models for data in a wide range of tasks. Moreover, the robustness of these networks also proves useful for auxiliary tasks such as noise reduction and pre-training for other machine learning tasks.

The training of neural networks can be aided by various preprocessing methods. There exist many ways of abstracting better features from raw data, ranging from Fourier transforms to statistical methods. This project will focus on the use of the discrete wavelet transform (DWT) which has already seen applications in data compression (Taubman and Marcellin 2002) and denoising (Taswell 2000). As such the DWT possesses similar properties to the variational autoencoder. We investigate the use of a signal’s discrete wavelet decompositions as features to train a variational encoder, which also conveniently allows us to simplify the neural networks by treating different levels of wavelet coefficients separately. This project attempts to build an unsupervised model for feature extraction using the variational autoencoder when applied to audio from the NSynth dataset. Three approaches are contrasted:

• The audio signal is fed directly into the variational autoencoder without preprocessing,
• the audio signal is first preprocessed using the discrete Fourier transform,
• the audio signal is first preprocessed using the discrete wavelet transform

with a modified VAE structure.

Through these toy examples, I hope to demonstrate that when combined with a variational autoencoder, the discrete wavelet transform produces a convincing model for both feature extraction and generative purposes. The methodology developed in this project is very general and can in principle be applied to any data where a wavelet approach is desired; we could also consider modifying the structure of the neural networks to incorporate more hidden layers, or use convolutional layers in place of fully connected ones.

# References

Carleo, Giuseppe, and Matthias Troyer. 2017. “Solving the Quantum Many-Body Problem with Artificial Neural Networks.” Science 355 (6325): 602–6.

Dunjko, Vedran, and Hans J Briegel. 2018. “Machine Learning & Artificial Intelligence in the Quantum Domain: A Review of Recent Progress.” Reports on Progress in Physics 81 (7): 074001.

Guest, Dan, Kyle Cranmer, and Daniel Whiteson. 2018. “Deep Learning and Its Application to Lhc Physics.” Annual Review of Nuclear and Particle Science 68: 161–81.

Kim, Edward J, and Robert J Brunner. 2016. “Star-Galaxy Classification Using Deep Convolutional Neural Networks.” Monthly Notices of the Royal Astronomical Society, stw2672.

Kingma, Diederik P, and Max Welling. 2013. “Auto-Encoding Variational Bayes.” arXiv Preprint arXiv:1312.6114.

Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E Hinton. 2012. “Imagenet Classification with Deep Convolutional Neural Networks.” In Advances in Neural Information Processing Systems, 1097–1105.

Pourrahmani, Milad, Hooshang Nayyeri, and Asantha Cooray. 2018. “LensFlow: A Convolutional Neural Network in Search of Strong Gravitational Lenses.” The Astrophysical Journal 856 (1): 68.

Silver, David, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, et al. 2017. “Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm.” arXiv Preprint arXiv:1712.01815.

Taswell, Carl. 2000. “The What, How, and Why of Wavelet Shrinkage Denoising.” Computing in Science & Engineering 2 (3): 12–19.

Taubman, David S, and Michael W Marcellin. 2002. “JPEG2000: Standard for Interactive Imaging.” Proceedings of the IEEE 90 (8): 1336–57.

Torlai, Giacomo, Guglielmo Mazzola, Juan Carrasquilla, Matthias Troyer, Roger Melko, and Giuseppe Carleo. 2018. “Neural-Network Quantum State Tomogra- phy.” Nature Physics 14 (5): 447–50.
