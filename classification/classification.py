# example from chapter 2 of the book
# practice with classification systems to see the end to end process of a data science project
# dataset based MNIST which is a set of 70,000 small digits handwritten by high school students and emplyees of US Census Buraeu
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# load the MNIST dataset
mnist = fetch_openml("mnist_784", version=1)
# print(mnist.keys()) -> dict_keys(['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url'])
# DESCR descirbes the data set, target has the array with labels

X, y = mnist["data"], mnist["target"]
# print(X.shape) -> (70000, 784)
# print(y.shape) -> (70000,)
# 70,000 images and 784 features; each image is 28x28 pixels and each feature represents one pixel's intenisty from white to black

# **** to view an image: ****
some_digit = X[0]
#some_digit_image = some_digit.reshape(28,28)

#plt.imshow(some_digit_image, cmap=mpl.cm.binary, interpolation="nearest")
#plt.axis("off")
#plt.show() >>> shows image of 5

# the labels are strings change to ints
y = y.astype(np.uint8)

# creating test set and train set - MNIST data set was already split this way
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# ****TRAINING BINARY CLASSIFIER****
# for example a detector capable of detecting either a 5 or not

# create target vectors for the classification task
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
# SGD classifier and train - SGD relies on randomness during training 
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
# predict the image called earlier which is actually a 5
# print(sgd_clf.predict([some_digit])) >>> [ True]

# ****PERFORMANCE MEASURES****
# now find accuracy of the model with K-fold cross-validation
# cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy") >>> array([0.96355, 0.93795, 0.95615])
# high accuracy! but only 10% of the dataset is 5 high chance of labeling things not 5
# not preffered method for classifiers especially with skewed datasets (some classes are more frequent than others)

# Confusion Matrix
# example: the number of times the classifier confused images of 5s with 3s, you would look in the 5th row and 3rd column of the confusion matrix
# use cross val predict to perform K-fold cross-validation, but instead of returning scores returns predictions made at each test fold
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
# dont want to use the test set yet because that is only at the very end
# now get the confusion matrix
# print(confusion_matrix(y_train_5, y_train_pred)) >>> array([[53057, 1522],[ 1325, 4096]]) from left to right top to bottom - true negative, false positive, false negative, true positive

# ****PRECISION AND RECALL****
# precision of classifier: using true/false positive and negative values (TP/TP+FP)
# another common metric is recall or sensitivity: TP/TP+FN
# precision_score(y_train_5, y_train_pred) # == 4096 / (4096 + 1522) >>> 0.7290850836596654
# recall_score(y_train_5, y_train_pred) # == 4096 / (4096 + 1325) >>> 0.7555801512636044

# F-score: harmonic mean of the precision and recall - 2*(p*r/p+r)
# f1_score(y_train_5, y_train_pred) >>> 0.7420962043663375
# increasing precision, decreases recall and vice versa

# Thresholding Technique
# see the decision scores with decision_function()
y_scores = sgd_clf.decision_function([some_digit]) # array([2412.53175101])
threshold = 0
y_some_digit_pred = (y_scores > threshold) # array([ True])
# return score for each instance and make predictions based on those scores with any threshold
# now increase the threshold; classifier detects 
threshold = 8000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred #array([False])
# NOTE: raising threshold decreases recall

# how to select threshold?
# get scores of all instances using the cross val predict function
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
# compute precision and recall for all possible thresholds
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
# plot the precision, recall vs threshold :
#def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
#plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
#plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
#[...] # highlight the threshold, add the legend, axis label and grid
#plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
#plt.show()

# to find a threshold for a certain precision percent (90% in this case):
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)] # ~7816
# now check precision and recall for 90% threshold
y_train_pred_90 = (y_scores >= threshold_90_precision)
precision_score(y_train_5, y_train_pred_90) # 0.9000380083618396
recall_score(y_train_5, y_train_pred_90) # 0.4368197749492714

# **** THE ROC CURVE ****
# Recevier Operating Characteristic - insted of precision vs recall, ROC plots true positive rate (recall or TPR) vs false positive rate
# FPR is ratio of negative instances that are incorrectly classified as positive (1 - true negative rate)
# true negative rate AKA TNR AKA specificity
# so ROC plots sensitivity (recall) vs 1 - specificity
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
# To Plot:
#def plot_roc_curve(fpr, tpr, label=None):
#plt.plot(fpr, tpr, linewidth=2, label=label)
#plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
#[...] # Add axis labels and grid
#plot_roc_curve(fpr, tpr)
#plt.show()
# Note: good classifier stays as far away from the dashed diagnonal as possible

# another way is AUC (area under the curve) - perfect classifier will have AUC of 1, random classifier has 0.5
# use scik-kit learn function to compute ROC AUC
roc_auc_score(y_train_5, y_scores) # 0.9611778893101814

# BIG NOTE: 
# use precision and recall when positive class is rare or care more for false positive than flase negatives
# use ROC otherwise

# **** MULTICLASS CLASSIFICATION**** 
# aka multinodal classifier
# algorithms capable of handling multiple classes like Random Fores and Naive Bayes Classification

# or you can use multiple binary classifiers (linear and SVM):
# One Versus All strategy(OvA): train binary classifier for each class and you get the decision score from each binary classifier and choose the highest
# One Versus One strategy(OvO): train binar classifier for every pair of class

# Scikit-Learn detects when you try to use a binary classification algorithm for a multiclass classification task, and it automatically runs OvA
sgd_clf.fit(X_train, y_train)
# sgd_clf.predict([some_digit]) >>> array([5], dtype=uint8)
some_digit_scores = sgd_clf.decision_function([some_digit]) # you can get the deicion scores for each class

# if you want to use OvO classifier:
from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([some_digit]) # array([5], dtype=uint8)

# training the Random Forest Classifier
forest_clf.fit(X_train, y_train)
forest_clf.predict([some_digit]) # array([5], dtype=uint8)
# to view the list of probabilities for each class
forest_clf.predict_proba([some_digit]) # array([[0. , 0. , 0.01, 0.08, 0. , 0.9 , 0. , 0. , 0. , 0.01]])

# evaluate the accuracy usuing cross val score
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy") # array([0.8489802 , 0.87129356, 0.86988048])
# can increase accuracy by scaling the inputs
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy") # array([0.89707059, 0.8960948 , 0.90693604])

# ****ERROR ANALYSIS****
# look at the confusion matrix with the cors val predict function
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred) 
#array([[5578, 0, 22, 7, 8, 45, 35, 5, 222, 1],
#[ 0, 6410, 35, 26, 4, 44, 4, 8, 198, 13],
#[ 28, 27, 5232, 100, 74, 27, 68, 37, 354, 11],
#[ 23, 18, 115, 5254, 2, 209, 26, 38, 373, 73],
#[ 11, 14, 45, 12, 5219, 11, 33, 26, 299, 172],
#[ 26, 16, 31, 173, 54, 4484, 76, 14, 482, 65],
#[ 31, 17, 45, 2, 42, 98, 5556, 3, 123, 1],
#[ 20, 10, 53, 27, 50, 13, 3, 5696, 173, 220],
#[ 17, 64, 47, 91, 3, 125, 24, 11, 5421, 48],
#[ 24, 18, 29, 67, 116, 39, 1, 174, 329, 5152]])
# fairly good confusion matrix because the diagonal has most images
# use the following to visualize the confusion matrix
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
# now to plot the errors:
# divide each value in the confusion matrix by the number of images in the corresponding class to compare error rates
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
# plot the error rates
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()
# NOTE: rows represent classes and columns represent predicted classes; more bright means higher error rate

# ****MULTILABEL CLASSIFICATION****
# when classifier needs to output multiple classes for each instance
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf.fit(X_train, y_multilabel)
# the y_multilabel created two target labels for each digit image
# and now a prediction is possible
knn_clf.predict([some_digit]) # array([[False, True]])
# (says that the The digit 5 is indeed not large (False) and odd (True).)

