

import transformers
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
import torchtext
from torchtext import data
import tez
from sklearn import metrics, model_selection
from transformers import AdamW, get_linear_schedule_with_warmup

!pip install tez
!pip install transformers

# Dataloader
class BERTDataset:
    def __init__(self, review, target):
        self.review = review
        self.target = target
        self.tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.max_len = 64

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        review = str(self.review[item])
        review = " ".join(review.split())

        inputs = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.target[item], dtype=torch.float),
        }

class BERTBaseUncased(tez.Model):
    def __init__(self, num_train_steps):
        super().__init__()
        self.tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.bert = transformers.BertModel.from_pretrained("bert-base-uncased", return_dict=False)
        self.bert_drop = nn.Dropout(0.5)
        self.out = nn.Linear(768, 1)
        self.num_train_steps = num_train_steps
        self.step_scheduler_after = "batch"

    def fetch_optimizer(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [{
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,}
        ]
        opt = AdamW(optimizer_parameters, lr=3e-5)
        return opt

    def fetch_scheduler(self):
        sch = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=self.num_train_steps
        )
        return sch

    def loss(self, outputs, targets):
        if targets is None:
            return None
        return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}
        outputs = torch.sigmoid(outputs).cpu().detach().numpy() >= 0.5
        targets = targets.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(targets, outputs)
        return {"accuracy": accuracy}

    def forward(self, ids, mask, token_type_ids, targets=None):
        _, out = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        outp = self.bert_drop(out)
        output = self.out(outp)
        loss = self.loss(output, targets)
        acc = self.monitor_metrics(output, targets)
        return output, loss, acc

train_dataset = BERTDataset( review=train_df.review.values, target=train_df.sentiment.values)
valid_dataset = BERTDataset(review=valid_df.review.values, target=valid_df.sentiment.values)

n_train_steps = int(len(train_df) / 32 * 10)
model = BERTBaseUncased(num_train_steps=n_train_steps)

es = tez.callbacks.EarlyStopping(monitor="valid_loss", model_path="model.bin")
model.fit(
    train_dataset,
    valid_dataset=valid_dataset,
    train_bs=64,
    device="cuda",
    epochs=50,
    callbacks=[es],
    fp16=True,
)
model.save("model.bin")

"""BiLSTM"""

from keras.layers import Dropout, Dense, Embedding, LSTM, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import shuffle
import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

n_unique_words = 10000 # cut texts after this number of words
 maxlen = 200
 batch_size = 128

!pip install zeugma

from zeugma.embeddings import EmbeddingTransformer

glove = EmbeddingTransformer('glove')
x2 = glove.transform(df.reviews)

X = x2
y = df.sentiment
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
 x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
 y_train = np.array(y_train)
 y_test = np.array(y_test)

model = Sequential()
 model.add(Embedding(n_unique_words, 128, input_length=maxlen))
 model.add(Bidirectional(LSTM(64)))
 model.add(Dropout(0.5))
 model.add(Dense(1, activation='sigmoid'))
 model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history=model.fit(x_train, y_train,
           batch_size=256,
           epochs=2,
           validation_data=[x_test, y_test])
 print(history.history['loss'])
 print(history.history['accuracy'])

from matplotlib import pyplot
 pyplot.plot(history.history['loss'])
 pyplot.plot(history.history['accuracy'])
 pyplot.title('model loss vs accuracy')
 pyplot.xlabel('epoch')
 pyplot.legend(['loss', 'accuracy'], loc='upper right')
 pyplot.show()

def get_eval_report(labels, preds):
    mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    precision = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)
    f1 = (2*(precision*recall))/(precision+recall)
    return {
        "mcc": mcc,
        "true positive": tp,
        "true negative": tn,
        "false positive": fp,
        "false negative": fn,
        "pricision" : precision,
        "recall" : recall,
        "F1" : f1,
        "accuracy": (tp+tn)/(tp+tn+fp+fn)
    }
def compute_metrics(labels, preds):
    assert len(preds) == len(labels)
    return get_eval_report(labels, preds)
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string], '')
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

print("\n Evaluating Model ... \n")
predicted = model.predict_classes(x_test)
print(metrics.classification_report(y_test, predicted))
print("\n")
logger = logging.getLogger("logger")
result = compute_metrics(y_test, predicted)
for key in (result.keys()):
    logger.info("  %s = %s", key, str(result[key]))

