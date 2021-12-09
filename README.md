# Amazon-Fake-Review-Detection

Authors: Parth Iramani, Sittun Swayam Prakash, Atrima Ghosh, Zoe Masood, Jenna Gottschalk, Mugundhan Murugesan

__DESCRIPTION:

The aim of our project is to detect fake reviews on Amazon using the review text. Our approach combines semi-supervised learning and transformer models to identify fake Amazon reviews.
The end product of our project is a web application which will allow the user to predict whether a review is fake or real.
The Datasets used in our projects are:
Labelled Dataset from Kaggle - https://www.kaggle.com/lievgarcia/amazon-reviews
Unlabelled dataset from Amazon - https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_multilingual_US_v1_00.tsv.gz

Following are the steps involved in creating and evaluation of the Model to predict fake reviews using review text:
(1) We split the original labeled dataset into four parts: 70% training set, 10% first validation set to compare initial supervised classification models, 10% second validation set to compare the updated classification models, and 10% test set to evaluate the final classifier’s performance. 
(2) We trained the initial model on the training set of labeled data. 
(3) We generated pseudo-labels by using the initial model to classify the unlabeled data set. 
(4) The most confidently predicted pseudo-labels above a specific threshold became training data for the next step. 
(5) We updated the initial model using the pseudo-labels as training data. 
(6) Finally, we tested the final classifier on the test set of the original labeled data. 

Modules used:
Pandas, Numpy, scikit-learn - Commonly used Machine Learning python libraries
transformers - Transformers provides thousands of pretrained models to perform tasks on texts such as classification, information extraction, question answering, summarization, translation, text generation and more in over 100 languages.
datasets - Huggingface Datasets is a library for easily accessing and sharing datasets, and evaluation metrics for Natural Language Processing (NLP), computer vision, and audio tasks.

The following are the steps and respective code (.py files) invloved in obtaining the final model file that is used for prediction on the website.
1. Modeling - Sentence Transformer and getting embeddings for review text
	|___ 11initial_cleaning.py
2. Logistic Regression with Labelled Dataset
	|___ 21initial_modelling.py
3. Generation of Pseudo Labels
	|___ 31clean_final_dataset.py
	|___ 32get_partitions.py
	|___ 33get_embeddings_for_partition.py
	|___ 34pseudo_labels.py
4. Unsupervised Learning using Stochastic Gradient Descent (SGD)
	|___ 41SGD_log.py

Analysis using BERT (Bidirectional Encoder Representations from Transformers) model:
1. Supervised Learning
	|___ 1_6242_GROUP_PROJECT_BERT_training_for_a.py
2. Generation of Pseudo_labels
	|___ 2_6242_GROUP_PROJECT_Generate_BERT_Pseudo_Labels.py
3. Unsupervised Learning
	|___ 3_6242_GROUP_PROJECT_BERT_training_for_b.py
4. Compilation of final dataset for visualization
	|___ 4_6242_GROUP_PROJECT_compiling_final_dataset


__Description of the website:

The web-application hosted on (http://machinelearner.eastus2.cloudapp.azure.com/) allows the user to enter the URL of the review or the text of the review and get a prediction. 

The public Tableau dashboard (https://public.tableau.com/app/profile/zoe.masood/viz/Book1_16383996536860/Dashboard1#1) is also accessible from the website. 

Directory Structure:
static
  |___ style.css (Style Sheet for html files)
templates
  |___ home.html (Renders the default home page)
  |___ manual.html (Renders the manual entry page for review data)
  |___ result.html (Renders the result page after model prediction)
LogReg.pkl (The saved Logistic Regression model that generates the predictions under the hood
download_sentence_transformer.py (Commands used to download and save st_model folder) 
run.py (The main python file running the Flask application which carries out all the computation)
gibberish_classifier.py (The Python classifier which checks if review text entered is gibberish - if yes, it asks user to re-enter review data)
st_model (The Sentence Transformer model that is used to generate paragraph embeddings from text data in run.py)

Modules used in run.py:
flask - to build the Python web application
pandas - to process data as dataframes
joblib - to load the LogReg.pkl saved model
sklearn - to generate the prediction output and prediction report
requests - to get html data from amazon URL
bs4 - BeautifulSoup package to scrape relevant data off html page returned by requests
sentence_transformers - to get the paragraph embeddings on text data from the saved st_model 

__INSTALLATION:

The python code files include commands that will install all the required libraries.

__EXECUTION:

First execute download_sentence_transformer.py to create the st_model folder needed for the website.
Execute run.py to run the website on your local server. 


