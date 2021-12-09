# 4.1 Unsupervised learning using Stochastic Gradient Descent (SGD)
# A logistic regressions model using SGD is trained with the labelled dataset initially 
# This model is then paritally fitted with the dataset with pseudo-labels

#installing required packages
pip install numpy # https://numpy.org/
pip install pandas # https://pandas.pydata.org/
pip install matplotlib # https://matplotlib.org/
pip install -U scikit-learn # https://scikit-learn.org/
pip install joblib # https://joblib.readthedocs.io/en/latest/

#import required libraries
import os
import numpy as np
import pandas as pd
from sklearn import (model_selection, metrics, preprocessing, linear_model)
import joblib
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# reading the labelled dataset
df_final = pd.read_csv("df_clean_embeddings.csv")
df_final = df_final.drop("Unnamed: 0", axis = 1)

# Assigning dependent and independent variables
X = df_final.drop("label", axis = 1)
y = df_final["label"]

# Splitting the dataset into train and test data
data_size = 21000
train_split = 0.7
sup_val_split = 0.1
unsup_val_split = 0.1
test_val_split = 0.1

X_train, X_test_1, y_train, y_test_1 = model_selection.train_test_split(df_final.drop('label', axis=1), df_final['label'], test_size=(1-train_split), random_state=42)
X_val, X_test, y_val, y_test = model_selection.train_test_split(X_test_1, y_test_1, test_size=0.333, random_state=42)
X_val_sup, X_val_unsup, y_val_sup, y_val_unsup = model_selection.train_test_split(X_val, y_val, test_size=0.5, random_state=42)

# Initializing parameters for SGD grid search 
random_seed = 42
init_est = linear_model.SGDClassifier(loss='log',
            random_state=random_seed, n_jobs=-1, warm_start=True)
small_alphas = [10.0e-08, 10.0e-09, 10.0e-10]
alphas = [10.0e-04, 10.0e-05, 10.0e-06, 10.0e-07]
l1_ratios = [0.075, 0.15, 0.30]
param_grid = [
            {'alpha': alphas, 'penalty': ['l1', 'l2'], 'average':[False]},
            {'alpha': alphas, 'penalty': ['elasticnet'], 'average':[False],
            'l1_ratio': l1_ratios},
            {'alpha': small_alphas, 'penalty': ['l1', 'l2'], 'average':[True]},
            {'alpha': small_alphas, 'penalty': ['elasticnet'], 'average':[True],
            'l1_ratio': l1_ratios}
]

#Using stochastic gradient descent (SGD) for logistic regression to patially fit for unsupervised learning
#Grid Search for SGD 

LogReg = model_selection.GridSearchCV(estimator=init_est, param_grid=param_grid, verbose=10)

LogReg.fit(X_train, y_train)
pred = LogReg.predict(X_val_sup)
accuracy = LogReg.score(X_val_sup, y_val_sup)
print("Accuracy = {}".format(accuracy))
print("Best Params: {}".format(LogReg.best_params_))
print("Best Scores: {}".format(LogReg.best_score_))   

#initializing SGD with the best parameters from Gradient Descent
init_est = linear_model.SGDClassifier(loss='log',
            random_state=random_seed, n_jobs=-1,alpha= 1e-09, average=True, penalty='l1', warm_start=True)

pred = init_est.predict(X_val_sup)
accuracy = init_est.score(X_val_sup, y_val_sup)
print("Accuracy = {}".format(accuracy))

#fitting the model with train split of labelled dataset
init_est.fit(X_train,y_train)

#Saving the SGD model 
filename = './sgdmodels/inital_sgd.pkl'
joblib.dump(init_est, filename)

#Partial fit using dataset with pseudo-labels
f_list = os.listdir("./pseudo_labels")
i=1
#Loading the saved model file
filename = './sgdmodels/inital_sgd.pkl'
LogRegloaded = joblib.load(filename)
dfn= pd.DataFrame()
dfn['file_name'] = ['']
dfn['model_name'] = ''
dfn['accuracy'] = 0
dfn['macro_f1_score'] = 0
dfn['Classification_report'] = ''
for file in f_list:    
    # Reading partitions of dataset for partial fit
    fname = "./pseudo_labels/"+file
    df = pd.read_csv(fname)
    print("file read")
    
    #enter probability threshold here ( Selecting pseudo-labels with greater than 0.90 probability threshold )    
    df = df[df['prob']>=0.90]
    df = df.drop(["Unnamed: 0",'prob'], axis = 1)
    X = df.drop("pseudo_label", axis = 1)
    y = df["pseudo_label"]
    if i>1:
        filename2 = './sgdmodels/LogReg_sgd'+str(i-1)+'.pkl'
        LogRegloaded = joblib.load(filename2)
    LogRegloaded.partial_fit(X,y)
    print("partial fit of model")
    pred = LogRegloaded.predict(X_val_unsup)
    accuracy = LogRegloaded.score(X_val_unsup, y_val_unsup)
    modelfname = './sgdmodels/LogReg_sgd'+str(i)+'.pkl'
    i+=1
    joblib.dump(LogRegloaded, modelfname)
    print("model saved")
    temp = pd.DataFrame()
    temp['file_name'] = [file]
    temp['model_name'] = modelfname
    temp['accuracy'] = accuracy
    temp['macro_f1_score'] = f1_score(y_val_unsup, pred, average='macro')
    temp['Classification_report'] = classification_report(y_val_unsup, pred)
    dfn = dfn.append(temp)
    print(file,"completed")

# writing the model accuracy, F1 score into a CSV file
dfn = dfn.reset_index().drop("index",axis=1)
dfn = dfn.drop([0,0]).reset_index().drop("index",axis=1)
dfn.to_csv("sgd_model_partial_train_val_unsup_greater_than_90.csv")    



