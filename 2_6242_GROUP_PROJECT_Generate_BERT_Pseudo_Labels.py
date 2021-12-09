# 2. Generation of pseudo-labels

#importing the required libraries
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments
from datasets import Dataset

#reading the cleaned csv file - file obtained from 31clean_final_dataset.py
df = Dataset.from_csv("/content/gdrive/My Drive/df_clean_wo_embeddings.csv")

df = df.remove_columns(list(df.features)[0])

#Creating partition for each member to avoid memory issues
len_part = 200000

partition = {
'Parth1': range(0, len_part),
'Parth2': range(len_part*1, len_part*2),
'Parth3': range(len_part*2, len_part*3),
'Parth4': range(len_part*3, len_part*4),
'Parth5': range(len_part*4, len_part*5),
'Parth6': range(len_part*5, len_part*6),
'Parth7': range(len_part*6, len_part*7),
'Parth8': range(len_part*7, len_part*8),
'Parth9': range(len_part*8, len_part*9),
'Parth10': range(len_part*9, len_part*10),
'Zoe1': range(len_part*10, len_part*11),
'Zoe2': range(len_part*11, len_part*12),
'Zoe3': range(len_part*12, len_part*13),
'Zoe4': range(len_part*13, len_part*14),
'Jenna1': range(len_part*14, len_part*15),
'Jenna2': range(len_part*15, len_part*16),
'Jenna3': range(len_part*16, len_part*17),
'Jenna4': range(len_part*17, len_part*18),
'Mugundhan1': range(len_part*18, len_part*19),
'Mugundhan2': range(len_part*19, len_part*20),
'Mugundhan3': range(len_part*20, len_part*21),
'Mugundhan4': range(len_part*21, len_part*22),
'Sittun1': range(len_part*22, len_part*23),
'Sittun2': range(len_part*23, len_part*24),
'Sittun3': range(len_part*24, len_part*25),
'Sittun4': range(len_part*25, len_part*26),
'Atrima1': range(len_part*26, len_part*27),
'Atrima2': range(len_part*27, len_part*28),
'Atrima3': range(len_part*28, len_part*29),
'Atrima4': range(len_part*29, len_part*30),
'Parth11': range(len_part*30, df.num_rows)
}

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


df1 = df.select(partition['Parth11']).map(tokenize_function, batched = True)

df2 = df1.remove_columns(["text"])
df3 = df2.with_format("torch")

model = AutoModelForSequenceClassification.from_pretrained("/content/gdrive/My Drive/Project1 6242/model_checkpoints/cp1", num_labels=2)
training_args = TrainingArguments(output_dir = '/content/gdrive/My Drive/Project1 6242/model_checkpoints',
                                  logging_dir ='/content/gdrive/My Drive/Project1 6242/model_logs',
                                  per_device_train_batch_size=16,
                                  per_device_eval_batch_size=64,
                                  num_train_epochs = 5,
                                  learning_rate = 2e-5,
                                  weight_decay = 0.01,
                                #   evaluation_strategy="epoch",
                                #   save_strategy = "epoch",
                                #   load_best_model_at_end = True
                                  )
trainer = Trainer(
    model=model, args=training_args
)

# predicting the pesudo-labels
pred = trainer.predict(df3)


## Don't forget to name your file as pred_name.csv (including partion number). For example, pred_Parth1.csv

np.savetxt("/content/gdrive/My Drive/Project1 6242/pred_Parth11.csv", pred.predictions,delimiter = ',')

