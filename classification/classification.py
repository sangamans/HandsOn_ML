# example from chapter 2 of the book
# practice with classification systems
# dataset based MNIST which is a set of 70,000 small digits handwritten by high school students and emplyees of US Census Buraeu
from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt

# load the MNIST dataset
mnist = fetch_openml("mnist_784", version=1)
# print(mnist.keys()) -> dict_keys(['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url'])
# DESCR descirbes the data set, target has the array with labels

X, y = mnist["data"], mnist["target"]
# print(X.shape) -> (70000, 784)
# print(y.shape) -> (70000,)
# 70,000 images and 784 features; each image is 28x28 pixels and each feature represents one pixel's intenisty from white to black

# **** to view an image: ****
#some_digit = X[0]
#some_digit_image = some_digit.reshape(28,28)

#plt.imshow(some_digit_image, cmap=mpl.cm.binary, interpolation="nearest")
#plt.axis("off")
#plt.show() >>> shows image of 5

# the labels are strings change to ints
y = y.astype(np.uint8)

# creating test set and train set - MNIST data set was already split this way
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]