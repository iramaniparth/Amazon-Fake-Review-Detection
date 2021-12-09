#2.1 Experimenting different models with the sentence tranformed text data 

#install the  numpy and pandas packages using preferred installer program (pip)
pip install numpy  # https://numpy.org/
pip install pandas # https://pandas.pydata.org/
pip install tranformers # https://pypi.org/project/transformers/

#import the installed packages into your program
import numpy as np
import pandas as pd

#Reading the obtained sentence embeddings for Logistic Regression

df = pd.read_csv("df_kaggle_mod_embed.csv") # set correct path for reading the csv file with embeddings 

#initial cleaning of the dataframe
df.drop("Unnamed: 0", axis = 1, inplace = True)
df.drop("TEXT", axis = 1, inplace = True)
df = df.rename({'EMBEDDINGS': 'embeddings', 'LABEL': 'label'}, axis=1)
df = df[["embeddings","label"]]

#Cleaning the dataset to obtain the embeddings as columns of dataframe
def clean_embeddings(row):
  data = row["embeddings"]
  data = data.replace('[',"")
  data = data.replace(']',"")
  list_of_nums = data.split(" ")

  clean_list = []
  final_clean_list = []

  for entry in list_of_nums:
    if (entry != ''):
      clean_list.append(entry)

  final_clean_list = [float(entry) for entry in clean_list]

  return final_clean_list

#apply() function to implement cleaning of embeddings
df["clean_embeddings"] = df.apply(lambda x: clean_embeddings(x), axis = 1)
df.drop("embeddings", axis=1, inplace = True)
df = df.rename({"clean_embeddings" : "embeddings"}, axis=1)
df = df [["embeddings","label"]]
df_embeddings_explode = pd.DataFrame(df["embeddings"].to_list())
df.drop("embeddings", axis=1, inplace = True)
df_embeddings_explode.fillna(0, inplace = True)
df_final = pd.concat([df_embeddings_explode, df], axis = 1)

#assign "fake" as 1 and "real" as 0
def num_label(row):
  label = row['label']
  if (label == "fake"):
    return 1
  else:
    return 0

#using apply() to implement the assigning     
df_final['label'] = df_final.apply(lambda x: num_label(x), axis = 1)

#writing the cleaned dataframe to a csv file for further use
df_final.to_csv('df_clean_embeddings.csv')

#Trying different models to obtain the best one for labelled dataset

#installing required packages
pip install matplotlib # https://matplotlib.org/
pip install -U scikit-learn # https://scikit-learn.org/
pip install joblib # https://joblib.readthedocs.io/en/latest/

#importing the installed packages
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
import joblib

#assigning independent variables into X and dependent variable to y
X = df_final.drop("label", axis = 1) #embeddings as columns are independent variable
y = df_final["label"] # label (0 or 1) is the dependent variable

#Splitting dataset for Train, validation and Test purpose
data_size = 21000
train_split = 0.7
sup_val_split = 0.1
unsup_val_split = 0.1
test_val_split = 0.1

X_train, X_test_1, y_train, y_test_1 = train_test_split(df_final.drop('label', axis=1), df_final['label'], test_size=(1-train_split), random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test_1, y_test_1, test_size=0.333, random_state=42)
X_val_sup, X_val_unsup, y_val_sup, y_val_unsup = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

#X_train, y_train - Training dataset (70% of the labelled dataset)
#X_test_1, y_test_1 - First Validation dataset (10% of the labelled dataset)
#X_val_sup, y_val_sup - Second Validation dataset for supervised learning (10% of the labelled dataset)
#X_val_unsup, y_val_unsup - Third Validation dataset for unsupervised learning (10% of the labelled dataset)

# a. Logistic Regression (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
param_grid = { 
    'C' : [2, 6, 10]
}
#Grid Search for Logistic Regression
LogReg = GridSearchCV(estimator=LogisticRegression(solver='liblinear',random_state=0), param_grid=param_grid)
LogReg.fit(X_train, y_train)   
print("Best Params: {}".format(LogReg.best_params_))
print("Best Scores: {}".format(LogReg.best_score_))   

pred = LogReg.predict(X_val_sup)
accuracy = LogReg.score(X_val_sup, y_val_sup)
print("Accuracy = {}".format(accuracy))
print("Macro F1 Score = {}".format(f1_score(y_val_sup, pred, average='macro')))
print(classification_report(y_val_sup, pred))

LogReg = LogisticRegression(solver='liblinear', C=2, random_state=0)
LogReg.fit(X_train, y_train)   

pred = LogReg.predict(X_val_unsup)
accuracy = LogReg.score(X_val_unsup, y_val_unsup)
print("Accuracy = {}".format(accuracy))
print("Macro F1 Score = {}".format(f1_score(y_val_unsup, pred, average='macro')))
print(classification_report(y_val_unsup, pred))

#saving the obtained logistic regression model as file for further reusage
filename = "/content/gdrive/My Drive/models/LogReg.pkl"
joblib.dump(LogReg, filename)

# b. Random Forest Classifier (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
param_grid = { 
    'n_estimators': [16, 256],
    'max_depth' : [8, 16]
}
#Grid Search for Random Forest Classifier
RFModel = GridSearchCV(estimator=RandomForestClassifier(random_state=0), param_grid=param_grid)
RFModel.fit(X_train, y_train)   
print("Best Params: {}".format(RFModel.best_params_))
print("Best Scores: {}".format(RFModel.best_score_))  

pred = RFModel.predict(X_val_sup)
accuracy = RFModel.score(X_val_sup, y_val_sup)
print("Accuracy = {}".format(accuracy))
print("Macro F1 Score = {}".format(f1_score(y_val_sup, pred, average='macro')))
print(classification_report(y_val_sup, pred))

# c. Naive Bayes Classifier (https://scikit-learn.org/stable/modules/naive_bayes.html)
param_grid = { 
    'var_smoothing': np.logspace(0,-9, num=100)
}

#Grid search for naive bayes classifier
NBModel = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid)
NBModel.fit(X_train, y_train)   
print("Best Params: {}".format(NBModel.best_params_))
print("Best Scores: {}".format(NBModel.best_score_))  

pred = NBModel.predict(X_val_sup)
accuracy = NBModel.score(X_val_sup, y_val_sup)
print("Accuracy = {}".format(accuracy))
print("Macro F1 Score = {}".format(f1_score(y_val_sup, pred, average='macro')))
print(classification_report(y_val_sup, pred))

# d. Multi-layer Perceptron classifier (https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

MLPModel = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter = 5000)
MLPModel.fit(X_train, y_train)
pred = MLPModel.predict(X_val_sup)
accuracy = MLPModel.score(X_val_sup, y_val_sup)
print("Accuracy = {}".format(accuracy))
print(classification_report(y_val_sup, pred))


 