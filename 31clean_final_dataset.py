# 3.1 Generation of pesudo-labels for the unlabelled dataset [reviews.tsv] (https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_multilingual_US_v1_00.tsv.gz)
# Program to get only the review text data for creating embeddings 

#install the  numpy and pandas packages using preferred installer program (pip)
pip install numpy # https://numpy.org/
pip install pandas # https://pandas.pydata.org/
pip install tqdm # https://pypi.org/project/tqdm/

#import the installed packages into your program
import numpy as np 
import pandas as pd 
from tqdm import tqdm, tqdm_notebook
tqdm.pandas()

#reading the tab-seperated file of the unlabelled dataset
df = pd.read_csv("/content/gdrive/My Drive/reviews.tsv", sep='\t', error_bad_lines=False)

# preliminary data cleaning to create embeddings in the next step
df.drop(['marketplace','customer_id','review_id','product_id','product_parent','product_title','helpful_votes','total_votes','vine','review_date'], axis=1, inplace=True)

df['text'] = df.progress_apply(lambda x: str(x['review_headline']) + ". " + str(x['review_body']), axis=1)
df.drop(['review_headline','review_body'], axis=1, inplace = True)

def clean_rating_col(row):
  rating = str(row['star_rating'])
  return rating[:-2]

df['star_rating'] = df.progress_apply(lambda x: clean_rating_col(x), axis=1)

def concatenate_categorical_values_in_text(row):
  rating = row['star_rating']
  verified = row['verified_purchase']
  category = row['product_category']
  verified_string = ""
  text = row['text']

  new_string = "The rating is {}. ".format(rating)
  if (verified == "Y"):
    new_string += "It is a verified purchase with product category {}. ".format(category)
  else:
    new_string += "It is not a verified purchase with product category {}. ".format(category)
  new_string += text

  return new_string

df['text'] = df.progress_apply(lambda x: concatenate_categorical_values_in_text(x), axis=1)
df.drop(['star_rating','verified_purchase','product_category'], axis=1, inplace = True)

#writing the cleaned dataset to csv file for creating embeddings
df.to_csv('df_clean_wo_embeddings.csv')
