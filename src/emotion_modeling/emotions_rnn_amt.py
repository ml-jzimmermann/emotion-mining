import pandas as pd
import tensorflow.keras.utils as utils
import numpy as np
import os


csv = pd.read_csv("../data/amt_results.csv")

np_results = np.array(csv)
text = np_results[:,27]
results = np_results[:,28]

# parameters
split_factor = 0.85
dropout = 0.6
units = 32
add_hidden_layers = 0
batch = 16
epochs = 75
shuffle = True
use_tensorboard = True
save_embeddings = False

log_dir = "../../../tensorboard/log/emotion-amt-epochs_" + str(epochs) + "_layers_" + str(add_hidden_layers + 2) + "_units_" + str(units) + "-dropout-" + str(dropout) + "_shuffle_e/"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


text_ = []
results_ = []
for i in range(0, len(results)):
    if results[i] < 9:
        text_.append(text[i])
        results_.append(results[i])
text = text_
results = results_

import re as reg

def clean(input):
    input = input.replace(",", " ").replace(".", " ")
    pattern = reg.compile("[^a-zA-Z0-9 ]")
    text = pattern.sub("", input)
    for i in range(10):
        text = text.replace("  ", " ")
    return text.lower()


# clean the strings
stoplist = set('for a of the on to it'.split())
x = [clean(t) for t in text]

from gensim import corpora as cp

# split into word lists
text = [[word for word in t.lower().split() if word not in stoplist] for t in x]

from collections import defaultdict

frequency = defaultdict(int)
frequency_limit = 1
for t in text:
    for token in t:
        frequency[token] += 1

text = [[token for token in t if frequency[token] > frequency_limit] for t in text]

# create dictionary
dict = cp.Dictionary(text)
#dict.save("../../data/dicts/emotions-amt_batch_wo-neutral.dict")

# match text with dictionary
corpus = [dict.doc2idx(t, unknown_word_index = 0) for t in text]
vocabulary = len(dict)

def max_text(c):
    m = 0
    for t in c:
        if len(t) > m:
            m = len(t)
    return m

# pad the texts
from tensorflow.keras.preprocessing import sequence
limit = max_text(corpus)
print("limit: " + str(limit))
corpus = sequence.pad_sequences(corpus, maxlen=limit)

# split into training and test data
Y = utils.to_categorical(results)
train_split = int(len(Y) * split_factor)
print("train_split: " + str(train_split))
np_x = np.array(corpus)

if shuffle:
    sdata = np.concatenate((np_x, Y.astype(int)), 1)
    np.random.shuffle(sdata)

    x = []
    y = []
    for line in sdata:
        x.append(line[:limit])
        y.append(line[limit:])
    Y = np.array(y)
    np_x = np.array(x)

x_train = np_x[:train_split]
y_train = Y[:train_split]

x_val = np_x[train_split:]
y_val = Y[train_split:]


import keras.layers as layers
import keras.models as models
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard


model = models.Sequential()

print("vocabulary: " + str(vocabulary))
embeddings = 100
model.add(layers.Embedding(vocabulary, embeddings, input_length=limit, name="embeddings"))
model.add(layers.Bidirectional(layers.CuDNNLSTM(units=units, return_sequences=True)))
model.add(layers.Dropout(dropout))
for i in range(add_hidden_layers):
    model.add(layers.Bidirectional(layers.CuDNNLSTM(units=units, return_sequences=True)))
    model.add(layers.Dropout(dropout))
model.add(layers.Bidirectional(layers.CuDNNLSTM(units=units)))
model.add(layers.Dropout(dropout))
#model.add(layers.Dense(128))
model.add(layers.Dense(y_train.shape[1], activation = "softmax"))

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
print(model.summary())

if save_embeddings:
    metadata = {"embeddings":"embeddings_meta.tsv"}
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1, embeddings_freq=10, embeddings_layer_names=["embeddings"], embeddings_metadata=metadata, embeddings_data=edata)
else:
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

#filepath="weights/emotions_batch_emotion-dfe_wo-n_bi_512_2_s0.85_weights-improvement-{epoch:02d}-{val_categorical_accuracy:.4f}.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
#callbacks_list = [checkpoint, tensorboard]
callbacks_list = []
if use_tensorboard:
    callbacks_list = [tensorboard]

history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch, validation_data=(x_val, y_val), verbose=1, callbacks=callbacks_list)

path = "../models/model_rmsprop_emotion-amt_batch_wo-n_bidirectional_embeddings_" + str(add_hidden_layers + 2) + "l_" + str(units) + "u_" + str(epochs) + "e_" + str(split_factor) + "sf_" + str(dropout) + "do.save"
# models.save_model(model, path)

with open("../models/model.summary.txt", "a") as file:
    file.write('\n')
    file.write(path)
    file.write('\n')
    file.write(str(history.history))
    file.write('\n')

print(42)
