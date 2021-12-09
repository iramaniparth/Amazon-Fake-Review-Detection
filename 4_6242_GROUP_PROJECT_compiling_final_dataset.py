# 4. creating the final dataset for visualization after generation of pseudo_labels

import numpy as np
import pandas as pd

pip install -U sentence-transformers
pip install datasets

from sentence_transformers import SentenceTransformer
sentence_transformer_model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1', device='cuda')

from datasets import Dataset
df = Dataset.from_csv("/content/gdrive/My Drive/df.csv")

df = pd.read_csv("/content/gdrive/My Drive/jenna_partition/df_jenna.csv")

df1, df2, df3, df4 = np.array_split(df, 4)

df1.to_csv("df_part1.csv")
!cp df_part1.csv "/content/gdrive/My Drive/jenna_partition"

df2.to_csv("df_part2.csv")
!cp df_part2.csv "/content/gdrive/My Drive/jenna_partition"

df3.to_csv("df_part3.csv")
!cp df_part3.csv "/content/gdrive/My Drive/jenna_partition"

df4.to_csv("df_part4.csv")
!cp df_part4.csv "/content/gdrive/My Drive/jenna_partition"

# Paragraph Embeddings

import numpy as np
import pandas as pd

from google.colab import drive
drive.mount("/content/gdrive")

!pip install -U sentence-transformers
!pip install datasets

from sentence_transformers import SentenceTransformer
sentence_transformer_model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1', device='cuda')

from datasets import Dataset
df = Dataset.from_csv("/content/gdrive/My Drive/df_part4.csv")

def get_paragraph_embeddings(row):
  row['embeddings'] = sentence_transformer_model.encode(row['text'])
  return row

df_e = df.map(get_paragraph_embeddings, batched=True)
df_e.to_csv('df_part4_w_embeddings_jenna.csv')
!cp df_part4_w_embeddings_jenna.csv "/content/gdrive/My Drive/"

## BERT

# Create Pseudo-Label Dataframe and Find Prediction Probabilities

import numpy as np
import pandas as pd
import glob
from scipy.special import softmax

dfs = {}
for file in glob.glob("/Users/jennagottschalk/Documents/CSE6242/project/pseudo labels/predictions/*"):
    name = (file.split('_')[1][:-4])
    if name in dfs:
        pass
    else:
        dfs[name] = pd.read_csv(file, header=None, names=['real_logit', 'fake_logit'])

order = ['Parth1', 'Parth2', 'Parth3', 'Parth4', \
         'Parth5', 'Parth6', 'Parth7', 'Parth8', \
         'Parth9', 'Parth10', 'Zoe1', 'Zoe2', \
         'Zoe3', 'Zoe4', 'Jenna1', 'Jenna2', \
         'Jenna3', 'Jenna4', 'Mugundhan1', 'Mugundhan2', \
         'Mugundhan3', 'Mugundhan4', 'Sittun1', 'Sittun2', \
         'Sittun3', 'Sittun4', 'Atrima1', 'Atrima2', \
         'Atrima3', 'Atrima4', 'Parth11']


df1 = pd.DataFrame(columns=['real_logit', 'fake_logit'])

for name in order:
    df1 = df1.append(dfs[name])

df2 = pd.DataFrame(softmax(df1, axis=1)).rename(columns={'real_logit': 'real_prob', 'fake_logit': 'fake_prob'})
df = pd.concat([df1, df2], axis=1)
df['confidence'] = df[['real_prob', 'fake_prob']].max(axis=1)

def conf_level(x):
    if x['confidence'] >= 0.9:
        return '90-100'
    elif x['confidence'] >= 0.8:
        return'80-90'
    elif x['confidence'] >= 0.7:
        return'70-80'
    else:
        return'00-70'
df['conf_level'] = df.apply(conf_level, axis=1)

df['label'] = pd.DataFrame(np.argmax(np.array(df2), axis=-1)).replace({0: 'real', 1: 'fake'})

# Create CSV Files
clean_text = pd.read_csv('df_clean_wo_embeddings.csv')
pseudo_labels = pd.read_csv('pseudo_labels.csv')
pseudo_labels['label'] = pseudo_labels['label'].map({'real': 0, 'fake':1})
df_complete = pd.concat([clean_text, pseudo_labels], axis=1)
df_complete.drop(columns=['Unnamed: 0'], inplace=True)
df_complete.to_csv('text_pseudo_labels_complete.csv')
df_text_plabel_prob = df_complete[['text', 'label', 'confidence']].rename(columns={'confidence': 'probability'})
df_text_plabel_prob.to_csv('text_pseudo_label_prob.csv')
df_text_plabel_prob90 = df_text_plabel_prob[df_text_plabel_prob['probability'] >= 0.9].drop(columns=['probability'])
df_text_plabel_prob90.to_csv('text_pseudo_label_prob90.csv')