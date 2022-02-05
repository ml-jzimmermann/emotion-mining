import pandas as pd
import tensorflow.keras.utils as utils
import numpy as np
import os


csv = pd.read_csv("../data/primary-plutchik-wheel-DFE.csv")
dropout = 0.5
units = 128
epochs = 50
shuffle = True
use_tensorboard = True
save_embeddings = False
log_dir = "../../../tensorboard/log/emotion-dfe-epochs_" + str(epochs) + "-units_" + str(units) + "-dropout-" + str(dropout) + "/"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

sentences = csv["sentence"]
results = csv["emotion"]
confidence = csv["emotion:confidence"]
labels = ["Anticipation", "Anger", "Disgust", "Sadness", "Surprise", "Fear", "Trust", "Joy", "Neutral"]

text_ = []
results_ = []
for emotion in range(0, len(results)):
    if results[emotion] in labels:
        text_.append(sentences[emotion])
        results_.append(results[emotion])
text = text_
results = results_

#print(len(text))
#print(len(results))
rs_ = []
for emotion in results:
    if emotion == "Anticipation":
        rs_.append(1)
    if emotion == "Anger":
        rs_.append(2)
    if emotion == "Disgust":
        rs_.append(3)
    if emotion == "Sadness":
        rs_.append(4)
    if emotion == "Surprise":
        rs_.append(5)
    if emotion == "Fear":
        rs_.append(6)
    if emotion == "Trust":
        rs_.append(7)
    if emotion == "Joy":
        rs_.append(8)
    if emotion == "Neutral":
        rs_.append(9)
results = rs_

# file = open("../dfe_input.data", "w")
# file.write("sentence,label")
# file.write("\n")
# for i in range(len(results)):
#     file.write('"' + str(sentences[i]) + '",' + str(results[i]) + "\n")
# file.flush()
# file.close()
# exit()

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
frequency_limit = 0
for t in text:
    for token in t:
        frequency[token] += 1
print(len(frequency))

# create and save metadata for embedding layer
if save_embeddings:
    file = open(log_dir + "embeddings_meta.tsv", "w")
    file.write("Word\tFrequency\n")
    print("meta_len: " + str(len(frequency)))
    for key in frequency.keys():
        file.write(str(key) + "\t" + str(frequency[key]) + "\n")
    file.flush()
    file.close()

text = [[token for token in t if frequency[token] > frequency_limit] for t in text]

# create dictionary
dict = cp.Dictionary(text)
#dict.save("../data/dicts/emotions_batch_wo-neutral-dfe.dict")

# match text with dictionary
corpus = [dict.doc2idx(t, unknown_word_index = 0) for t in text]
vocabulary = len(dict)
print("vocabulary: " + str(vocabulary))

def max_text(c):
    m = 0
    for t in c:
        if len(t) > m:
            m = len(t)
    return m

# pad the texts
limit = max_text(corpus)
print("limit: " + str(limit))

if save_embeddings:
    edata = []
    for line in corpus:
        for symbol in line:
            edata.append(symbol)

    print(len(edata))
    edata = set(edata)
    print(len(edata))
    list = []
    for e in edata:
        list.append(e)
    edata = np.array(list)
    edata = edata.reshape((vocabulary,1))
    print(edata)

from keras.preprocessing import sequence
corpus = sequence.pad_sequences(corpus, maxlen=limit)

# split into training and test data
split_factor = 0.85
Y = utils.to_categorical(results)
train_split = int(len(Y) * split_factor)
print("train_split: " + str(train_split))
#val = 50
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
print(x_train)
print("x_train:", x_train.shape)
print("x_val:", x_val.shape)

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

filepath="weights/emotions_batch_emotion-dfe_wo-n_bi_512_2_s0.85_weights-improvement-{epoch:02d}-{val_categorical_accuracy:.4f}.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
#callbacks_list = [checkpoint, tensorboard]
callbacks_list = []
if use_tensorboard:
    callbacks_list = [tensorboard]

history = model.fit(x_train, y_train, epochs=epochs, batch_size=16, validation_data=(x_val, y_val), verbose=1, callbacks=callbacks_list)

path = "../models/model_rmsprop_emotion-dfe_batch_wo-n_bidirectional_embeddings_" + str(units) + "_" + str(epochs) + "_split_" + str(split_factor) + "_" + str(dropout) + ".save"
#models.save_model(model, path)

with open("../models/model.summary.txt", "a") as file:
    file.write('\n')
    file.write(path)
    file.write('\n')
    file.write(str(history.history))
    file.write('\n')

print(42)
