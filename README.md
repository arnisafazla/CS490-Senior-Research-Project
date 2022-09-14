# CS490

An LSTM conditional WGAN-GP implementation and data processing pipeline for generating Human Motion Capture data sequences
belonging to specific classes.

Credits:
* We adapted the PyMO library (https://github.com/omimo/PyMO) for processing and visualization of 3D Human Motion Capture data.
* We used the Emotional Body Motion database provided by the Max-Planck Institute for Biological Cybernetics in
Tuebingen, Germany (Volkova, EP, Mohler, BJ, Dodds, TJ, Tesch, J, B ̈ulthoff, HH. Emotion categorization of body expressions in narrative
scenarios. Frontiers in Psychology 2014;5:623.)

About the techniques used:
* Conditional GANs: https://arxiv.org/abs/1411.1784
* Wasserstein Loss Function: https://arxiv.org/abs/1701.07875
* Using Gradient Penalty instead of clipping the Critic weights: https://arxiv.org/pdf/1704.00028v3.pdf
