# 1. Supervised Learning
# Run below cell to install required modules

pip install -U sentence-transformers
pip install datasets


import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
df_kaggle = pd.read_csv("/content/gdrive/My Drive/Project 6242/amazon_reviews.txt",sep="\t")

df_kaggle_text = df_kaggle[['RATING','VERIFIED_PURCHASE','PRODUCT_CATEGORY','REVIEW_TITLE', 'REVIEW_TEXT','LABEL']].copy()
df_kaggle_text['TEXT'] = df_kaggle_text.apply(lambda x: x['REVIEW_TITLE'] + ". " + x['REVIEW_TEXT'], axis=1)
df_kaggle_text.drop(['REVIEW_TITLE','REVIEW_TEXT'], axis=1, inplace=True)
df_kaggle_text = df_kaggle_text[['RATING','VERIFIED_PURCHASE','PRODUCT_CATEGORY','TEXT','LABEL']]


def parse_label(label):
    if label == '__label2__':
        return 0
    else:
        return 1
df_kaggle_text['LABEL'] = df_kaggle_text['LABEL'].apply(lambda x: parse_label(x))



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
df_kaggle_mod["TEXT"] = df_kaggle_mod.apply(lambda x: concatenate_categorical_values_in_text(x), axis=1)
df_kaggle_mod.drop(["RATING","VERIFIED_PURCHASE","PRODUCT_CATEGORY"], axis=1, inplace = True)

#importing required libraries
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments
from datasets import Dataset

from sklearn.model_selection import train_test_split

data_size = 21000
train_split = 0.7
sup_val_split = 0.1
unsup_val_split = 0.1
test_val_split = 0.1

X_train, X_test_1, y_train, y_test_1 = train_test_split(df_kaggle_mod['TEXT'], df_kaggle_mod['LABEL'], test_size=(1-train_split), random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test_1, y_test_1, test_size=0.333, random_state=42)
X_val_sup, X_val_unsup, y_val_sup, y_val_unsup = train_test_split(X_val, y_val, test_size=0.5, random_state=42)


train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
val_sup = pd.concat([X_val_sup, y_val_sup], axis=1).reset_index(drop=True)
val_unsup = pd.concat([X_val_unsup, y_val_unsup], axis=1).reset_index(drop=True)
test = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

train = Dataset.from_pandas(train)
val_sup = Dataset.from_pandas(val_sup)
val_unsup = Dataset.from_pandas(val_unsup)
test = Dataset.from_pandas(test)

#Tokenizing using BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["TEXT"], padding="max_length", truncation=True)

train1 = train.map(tokenize_function, batched = True)
val_sup1 = val_sup.map(tokenize_function, batched = True)
val_unsup1 = val_unsup.map(tokenize_function, batched = True)
test1 = test.map(tokenize_function, batched = True)

train2 = train1.remove_columns(["TEXT"])
val_sup2 = val_sup1.remove_columns(["TEXT"])
val_unsup2 = val_unsup1.remove_columns(["TEXT"])
test2 = test1.remove_columns(["TEXT"])

train3 = train2.rename_column("LABEL", "labels")
val_sup3 = val_sup2.rename_column("LABEL", "labels")
val_unsup3 = val_unsup2.rename_column("LABEL", "labels")
test3 = test2.rename_column("LABEL", "labels")


train3 = train3.with_format("torch")
val_sup3 = val_sup3.with_format("torch")
val_unsup3 = val_unsup3.with_format("torch")
test3 = test3.with_format("torch")

import numpy as np
from datasets import load_metric

#Loading the metric object for defining parameter
metric = load_metric("glue", "mrpc")

#Trainer parameter - compute_metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

#Defining the model 

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

#Defining the training arguments 
training_args = TrainingArguments(output_dir = '/content/gdrive/My Drive/Project1 6242/model_checkpoints',
                                  logging_dir ='/content/gdrive/My Drive/Project1 6242/model_logs',
                                  per_device_train_batch_size=16,
                                  per_device_eval_batch_size=2,
                                  num_train_epochs = 5,
                                  learning_rate = 2e-5,
                                  weight_decay = 0.01,
                                #   evaluation_strategy="epoch",
                                  save_strategy = "epoch",
                                #   load_best_model_at_end = True
                                  )
#Defining the parameters for the trainer
trainer = Trainer(
    model=model, args=training_args, train_dataset=train3, eval_dataset=val_sup3, compute_metrics=compute_metrics
)

#training the model
trainer.train()
#Evaluate the model using compute_metrics using the eval_dataset given in the trainer parameters
trainer.evaluate()


 




