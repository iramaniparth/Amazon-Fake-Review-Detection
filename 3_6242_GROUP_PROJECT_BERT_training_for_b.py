#3. Training the model using pseudo-labels
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments
from datasets import Dataset, load_metric

df = Dataset.from_csv("/home/zmasood3/review_data/text_pseudo_label_prob90.csv")
df = df.remove_columns(list(set(df.features) -  {'text', 'label'}))

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

df1 = df.map(tokenize_function, batched = True)
df2 = df1.rename_column("label", "labels")
df3 = df2.remove_columns(["text"])
df4 = df3.with_format("torch")


metric = load_metric("glue", "mrpc")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
training_args = TrainingArguments(output_dir = 'ModelB4_checkpoints',
                                  logging_dir ='ModelB4_logs',
                                  per_device_train_batch_size=32,
                                  per_device_eval_batch_size=64,
                                  num_train_epochs = 5,
                                  learning_rate = 2e-5,
                                  weight_decay = 0.01,
                                #   evaluation_strategy="epoch",
                                  save_strategy = "epoch",
                                #   load_best_model_at_end = True
                                  )


trainer = Trainer(
    model=model, args=training_args, train_dataset=df4.shuffle(seed=42).select(range(2000000))
)

trainer.train()

#saving the trained model file
model.save_pretrained('ModelB/cp4')