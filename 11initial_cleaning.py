#1.1 Cleaning the labelled Dataset (https://www.kaggle.com/lievgarcia/amazon-reviews)
#Creating sentence embedding for labelled dataset

#install the  numpy, pandas, and sentence tranformer packages using preferred installer program (pip)
pip install -U sentence-transformers # https://www.sbert.net/
pip install numpy # https://numpy.org/
pip install pandas # https://pandas.pydata.org/

#import the installed packages into your program
import numpy as np 
import pandas as pd 
from sentence_transformers import SentenceTransformer

#reading the labelled Dataset into a dataframe

filename = "/content/gdrive/My Drive/amazon_reviews_kaggle.txt" #path of the csv file 

df_kaggle = pd.read_csv(filename,sep="\t")

#Data cleaning process - obtaining the required columns from the dataframe of  the csv file
df_kaggle_text = df_kaggle[['RATING','VERIFIED_PURCHASE','PRODUCT_CATEGORY','REVIEW_TITLE', 'REVIEW_TEXT','LABEL']].copy()

df_kaggle_text['TEXT'] = df_kaggle_text.apply(lambda x: x['REVIEW_TITLE'] + ". " + x['REVIEW_TEXT'], axis=1)

df_kaggle_text.drop(['REVIEW_TITLE','REVIEW_TEXT'], axis=1, inplace=True)

df_kaggle_text = df_kaggle_text[['RATING','VERIFIED_PURCHASE','PRODUCT_CATEGORY','TEXT','LABEL']]

#assigning '__label1_' as 'fake' and '__label2__' as 'real' for more clarity
def parse_label(label):
    if label == '__label2__':
        return 'real'
    else:
        return 'fake' 

#using apply() command of pandas for assigning the labels
df_kaggle_text['LABEL'] = df_kaggle_text['LABEL'].apply(lambda x: parse_label(x))

#Concatenating the categorical variables such as 'RATING', 'VERIFIED_PURCHASE', 'PRODUCT_CATEGORY' into the review text
#for example: RATING will be concatenated as "The Rating is 5."
#VERIFIED_PURCHASE and PRODUCT_CATEGORY will be concatenated as "It is a verified purchase with product category {}"
def concatenate_categorical_values_in_text(row):
  rating = row['RATING']
  verified = row['VERIFIED_PURCHASE']
  category = row['PRODUCT_CATEGORY']
  verified_string = ""
  text = row['TEXT']

  new_string = "The rating is {}. ".format(rating)
  if (verified == "Y"):
    new_string += "It is a verified purchase with product category {}. ".format(category)
  else:
    new_string += "It is not a verified purchase with product category {}. ".format(category)
  new_string += text

  return new_string


df_kaggle_mod = df_kaggle_text.copy()

#implementing concatenation using apply() function
df_kaggle_mod["TEXT"] = df_kaggle_mod.apply(lambda x: concatenate_categorical_values_in_text(x), axis=1)

#Dropping the categorical variable rows after the concatenation process
df_kaggle_mod.drop(["RATING","VERIFIED_PURCHASE","PRODUCT_CATEGORY"], axis=1, inplace = True)

#Creating sentence embeddings using the sentence transformer package
sentence_transformer_model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')

#function to transform text data
def get_paragraph_embeddings(row):
  embedding = sentence_transformer_model.encode(row["TEXT"])
  return embedding

#implementing sentence embedding using apply() function 
df_kaggle_mod["EMBEDDINGS"] = df_kaggle_mod.apply(lambda x: get_paragraph_embeddings(x), axis=1)

#writing the obtained embeddings into a csv file for further use
df_kaggle.to_csv("df_kaggle_mod_embed.csv")