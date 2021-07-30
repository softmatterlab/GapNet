# GapNet
© Yu-Wei Chang, Laura Natali, Oveis Jamialahmadi, Stefano Romeo, Joana B. Pereira & Giovanni Volpe
http://www.softmatterlab.org

GapNet is an alternative deep-learning training approach that can use highly incomplete datasets. This is the code for the arXiv preprint 2107.00429 [Neural Network Training with Highly Incomplete Datasets](https://arxiv.org/abs/2107.00429). 

## Dependencies 
* Python 3.8.5
* Tensorflow 2.5.0
* pydot 1.2.3
* Pandas 1.3.1
* scikit-learn 0.24.2

## Usage
To see GapNet working principle, we provide two well-documented tutorial notebooks that train the GapNet model on a simulated dataset:

1. [GapNet tutorial.ipynb](https://github.com/softmatterlab/GapNet/blob/main/src/GapNet%20tutorial.ipynb) demonstrates how to train a GapNet model on a simulated dataset with highly incomplete features.
2. [Comparison with Vanilla neural network.ipynb](https://github.com/softmatterlab/GapNet/blob/main/src/Comparison%20with%20Vanilla%20neural%20network.ipynb) compares the performance between GapNet and Vanilla models.

Each code example is a Jupyter Notebook that also includes detailed comments to guide the user. All neccesary files to run the code examples are provided. 
