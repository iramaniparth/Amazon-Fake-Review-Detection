# 3.4 Generation of pesudo-labels for the partitions with embeddings

#installing required packages
pip install numpy # https://numpy.org/
pip install pandas # https://pandas.pydata.org/
pip install matplotlib # https://matplotlib.org/
pip install -U scikit-learn # https://scikit-learn.org/
pip install joblib # https://joblib.readthedocs.io/en/latest/

#import required libraries
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib
import numpy as np
import pandas as pd


#load the saved logistic regression model
filename = "LogReg.joblib"
LogRegloaded = joblib.load(filename)
print("Model loaded")
#read the input csv file of the cleaned embeddings
df_predict = pd.read_csv("clean_part_4_sittun.csv", header=None)
print("csv read")

#create pseudo labels using predict()
pred = LogRegloaded.predict(df_predict)
df_predict['pseudo_label'] = pred
df_pred_withindex = pd.DataFrame(df_predict['pseudo_label'])
print("pseudolabels created")

#create probability of prediction
proba = pd.DataFrame(LogRegloaded.predict_proba(df_predict.drop("pseudo_label", axis=1)))
proba['max'] = proba.apply(lambda x: max(x[0],x[1]), axis=1)
df_pred_withindex['prob'] = proba['max']
df_predict['prob'] = proba['max']
print("prediction confidence created")

#create .csv of clean_embedding with pseudo_labels
df_predict.to_csv('with_pseudo_label_clean_part_4_sittun.csv')
