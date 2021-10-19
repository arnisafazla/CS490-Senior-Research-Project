# CS490

Project Idea:
1.	Train a model to estimate emotions from body poses. Report: https://arxiv.org/pdf/1904.09435.pdf. Dataset: http://ebmdb.tuebingen.mpg.de/
2.	Train GAN model to generate body poses according to emotions. Style transfer for animation: https://medium.com/deepgamingai/motion-style-transfer-for-3d-character-animation-7d66c423e743. Motion synthesis: https://www.arxiv-vanity.com/papers/2104.05670/. Dataset: https://deepai.org/publication/animgan-a-spatiotemporally-conditioned-generative-adversarial-network-for-character-animation
3.	If dataset is not large enough: Use openCV emotion recognition from facial expressions to detect the images with different types of emotions. Download those images and label them by the emotions detected: https://pypi.org/project/simple-image-download/. Process the images. Train GAN again. Pose is more effective in emotion estimation than facial expression: https://www.encephale.com/content/download/91193/1652030/version/1/file/aviezer_how_discriminate_intense_positive_negative_emotions_Science_2012.pdf

Implementation Details:
1.	Use Keras CNN/LSTM for emotion recognition model.

Visualizing your model:
1.	https://machinelearningmastery.com/visualize-deep-learning-neural-network-model-keras/ 
2.	https://keras.io/api/utils/model_plotting_utils/ 

Similar work for generating images:
1.	J. Xu, J. Broekens, K. Hindriks, and M. A. Neerincx, “Robot mood is contagious: effects of robot body language in the imitation game,” in Proceedings of the 2014 international conference on Autonomous agents and multi-agent systems. International Foundation for Autonomous Agents and Multiagent Systems, 2014, pp. 973–980.
