# 3.3 Creating sentence embeddings for cleaned unlabelled datasets
# implemented batchwise on each partition to avoid memory issue

#install the  numpy, pandas, sentence tranformer and datasets packages using preferred installer program (pip)
pip install numpy # https://numpy.org/
pip install pandas # https://pandas.pydata.org/
pip install -U sentence-transformers # https://www.sbert.net/
pip install datasets #https://pypi.org/project/datasets/

#import the required libraries
import numpy as np
import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer

#sentence transformer to convert review text 
sentence_transformer_model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1', device='cuda')
def get_paragraph_embeddings(row):
  row['embeddings'] = sentence_transformer_model.encode(row['text'])
  return row

#creating embeddings for each partitions and writing them to csv file for further use
df1 = Dataset.from_csv("df_partition_1_teammember.csv")
df_e1 = df1.map(get_paragraph_embeddings, batched=True)
df_e1.to_csv('df_w_embeddings_part_1.csv')

df2 = Dataset.from_csv("df_partition_2_teammember.csv")
df_e2 = df2.map(get_paragraph_embeddings, batched=True)
df_e2.to_csv('df_w_embeddings_part_2.csv')

df3 = Dataset.from_csv("df_partition_3_teammember.csv")
df_e3 = df3.map(get_paragraph_embeddings, batched=True)
df_e3.to_csv('df_w_embeddings_part_3.csv')

df4 = Dataset.from_csv("df_partition_4_teammember.csv")
df_e4 = df4.map(get_paragraph_embeddings, batched=True)
df_e4.to_csv('df_w_embeddings_part_4.csv')
