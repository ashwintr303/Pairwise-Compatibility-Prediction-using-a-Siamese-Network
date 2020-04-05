# Pairwise-Compatibility-Prediction-using-a-Siamese-Network

## Dataset description
Polyvore Outﬁts is a real-world dataset created based on users’ preferences of outﬁt conﬁgurations on an online website named polyvore.com: items within the outﬁts that receive high-ratings are considered compatible and vice versa. It contains a total of 365,054 items and 68,306 outﬁts. The maximum number of items per outﬁt is 19. A visualization of an outﬁt is shown in Figure 1.

## File description  
1. dataloader.py: Data preprocessing and loading.  
2. model.py: Siamese model   
3. utils.py: hyperparameters and file paths  

## Dependencies
1. Python 3.7  
2. tensorflow 2.1  
3. tensorflow-gpu 2.1  
4. tqdm 4.43  
5. torchvision 0.5  
6. pillow 7.0.0  
To install the complete list of dependencies, run:  
```
pip install -r requirements.txt
```

## Running the code:  
Set the parameters in utils.py. The code uses CUDA which needs an Nvidia GPU. If not using a GPU, set use_cuda flag to False in utils.py.

## References  
[1] https://github.com/davidsonic/EE599-CV-Project  
[2] https://www.tensorflow.org/api_docs/python/tf/keras/Model
[3] https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d
