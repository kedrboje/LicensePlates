import numpy as np 
import matplotlib.pyplot as plt
from keras.datasets import mnist 

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

digit = train_images[2]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()