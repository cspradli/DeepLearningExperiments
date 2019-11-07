import math
import numpy as np
import os
import matplotlib as pyplot
from scipy import optimize
from scipy.io import loadmat
import utils

a = np.array([[1, 3, 4, 6, 7, 10, 24],
              [12,23, 3, 5,10, 23, 34],
              [13,73, 9, 5,10, 23, 34],
              [12,21, 32, 55,1, 21, 32]])

print(a)

a.transpose()
print(a.transpose())

def sigmoid(input):
    """ sigmoid activation function """
    return (1 / (1 + math.exp(-input)))

print(sigmoid(12))