#!pip install pandas
#!pip install numpy
#!pip install sklearn
#!pip install matplotlib

from scipy import stats
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from google.colab import files

# Import cleaned.csv dataset (may need to upload again if runtime expires)
# FROM: https://archive-beta.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
# Original Source with information about how this dataset was procured: https://www.mdpi.com/2306-5729/7/11/146
uploaded = files.upload()
print(uploaded)


# Reads csv and creates dataframe
filename = list(uploaded.keys())[0]
print(filename)
print("-----------------")
cleaned = pd.read_csv(filename)
cleaned.head()

# Drops Enrolled in Target because we want binary classification for dropout or not.
cleaned = cleaned[cleaned['Target'] != 'Enrolled']

# Prepares dependent and independent variables
x = pd.DataFrame(cleaned.drop(columns=["Unnamed: 0", "Target"]))
y = pd.DataFrame(cleaned["Target"])
features = x
target = y

# Split data for train and test
x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=0)  # returns a tupule
# Added random_state=0 to ensure consistent splitting
# x_train is features, y_train is the labels for training (.7)
# x_test is the features, y_test is the lables for testing (.3)
# training data .7 rows of dataset, and testing with .3 rows of data set
print("Dataset Splitted")
print(x.shape)
print(x_train.shape)
print(x_test.shape)


# Trains the classifier model
model = RandomForestClassifier(random_state=0)
model.fit(x_train, np.array(y_train).ravel())
# features, labels - .ravel helped with an error- converted a type

# Testing our model
# Accuracy score and report for test
y_test_predict = model.predict(x_test)
# Get the models prediction y_test_predict for x_test
print(classification_report(y_test, y_test_predict))
# Compare model y_test predict with actual y_test
print("Accuracy: ", accuracy_score(y_test, y_test_predict))

# Recall = Correct Predictions / Total of particular class

# Generate plots, confusion matrix? using matplotlib?
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html

cm = confusion_matrix(y_test, y_test_predict, labels=model.classes_)
display = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=model.classes_)
display.plot()
plt.show()

# Topleft = Number of correct predictions of Dropouts from model
# Topright = Number of incorrect predictions of Dropouts from model
# Bottomleft = Number of incorrect predictions of Graduate from model
# Bottomright = Number of correct predictions of Graduate from model
# bottom right is true positive
# bottom left is false negative
# top right is false positive
# top left  is true negative
# precision: TP/ TP + FP
# recall : TP/ TP + FN

# 640/(640 + 28) Recall for positive
# 349/(349 + 72) Recall for negative
# 640/(640 + 72) Precision for positive
# 349/(349 + 28) Precision for negative

# Recall for Positive = TP / TP + FN
# Recall for Negative = TN / TN + FP
# Precision for Positive = TP / TP + FP
# Precision for Negative = TN / TN + FN

# roc curve

# Encodes labels to 1s and 0s
y_test_reshaped = np.reshape(np.array(y_test.T).ravel(), y_test_predict.shape)
y_test_encoded = np.where(y_test_reshaped == 'Dropout', 0, 1)
y_test_predict_encoded = np.where(y_test_predict == 'Dropout', 0, 1)


false_pos_rate, true_pos_rate, thresholds = roc_curve(
    y_test_encoded, y_test_predict_encoded)
print(false_pos_rate)
print(true_pos_rate)
print(thresholds)

# Compute AUC (Area Under the ROC Curve)
auc = roc_auc_score(y_test_encoded, y_test_predict_encoded)

# Plot the ROC curve
plt.plot(false_pos_rate, true_pos_rate,
         label='ROC curve (AUC = {:.2f})'.format(auc))
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# roc curve

# Gets prediction probabilities instead of just predicted binary labels
y_test_predictMix = model.predict_proba(x_test)

# Encodes labels to 1s and 0s
y_test_reshaped = np.reshape(np.array(y_test.T).ravel(), y_test_predict.shape)
y_test_encoded = np.where(y_test_reshaped == 'Dropout', 0, 1)


false_pos_rate, true_pos_rate, thresholds = roc_curve(
    y_test_encoded, y_test_predictMix[:, 1])
print(thresholds)

# Compute AUC (Area Under the ROC Curve)
auc = roc_auc_score(y_test_encoded, y_test_predictMix[:, 1])

# Plot the ROC curve
plt.plot(false_pos_rate, true_pos_rate,
         label='ROC curve (AUC = {:.2f})'.format(auc))
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# print(x_test.columns)

# Given the exact same features, except that the gender column is set to female
x_testFemale = x_test.copy()
x_testFemale["Gender"].replace(1, 0, inplace=True)

# Given the exact same features, except that the gender column is set to male
x_testMale = x_test.copy()
x_testMale["Gender"].replace(0, 1, inplace=True)


# Predict the probabilities using the classifier model
y_test_predictFemale = model.predict_proba(x_testFemale)
y_test_predictMale = model.predict_proba(x_testMale)

# Print averages
print("Prediction    ", model.classes_)
print("Female        ", np.mean(y_test_predictFemale, axis=0))
print("Male          ", np.mean(y_test_predictMale, axis=0))
print("Base         ", np.mean(y_test_predictMix, axis=0))


# T-Test
t_stat, p_value = stats.ttest_ind(
    y_test_predictMale[:, 0], y_test_predictFemale[:, 0], equal_var=True)

print("t-statistic:", t_stat)
print("p-value:", p_value)

# print(x_test.columns)

# Given the exact same features, except that the Scholarship holder column is set to 0
x_testNoScholarship = x_test.copy()
x_testNoScholarship["Scholarship holder"].replace(
    1, 0, inplace=True)  # Scholarship holder

# Given the exact same features, except that the Scholarship holder column is set to 1
x_testScholarship = x_test.copy()
x_testScholarship["Scholarship holder"].replace(0, 1, inplace=True)


# Predict the probabilities using the classifier model
y_test_predictNoScholarship = model.predict_proba(x_testNoScholarship)
y_test_predictScholarship = model.predict_proba(x_testScholarship)

# Print averages
print("Prediction    ", model.classes_)
print("No Scholarship", np.mean(y_test_predictNoScholarship, axis=0))
print("Scholarship   ", np.mean(y_test_predictScholarship, axis=0))
print("Base          ", np.mean(y_test_predictMix, axis=0))


# T-Test
t_stat, p_value = stats.ttest_ind(
    y_test_predictScholarship[:, 0], y_test_predictNoScholarship[:, 0], equal_var=True)

print("t-statistic:", t_stat)
print("p-value:", p_value)
