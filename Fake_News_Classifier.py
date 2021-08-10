#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import pandas as pd
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split


#preprocess data
fake = pd.read_csv('/Users/mattgevercer/Downloads/archive/Fake.csv')
fake['label'] = 1
real = pd.read_csv('/Users/mattgevercer/Downloads/archive/True.csv')
real['label'] = 0
data = pd.concat((fake,real))

sentences = data['text'].tolist()
labels = data['label'].tolist()

#split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    sentences, labels, test_size = 0.2)

y_train = np.array(y_train)
y_test = np.array(y_test)


#hyperparameters
vocab_size = 10000
embedding_dim = 16
max_length = 300
trunc_type = 'post'
oov_tok = '<OOV>'

#tokenize and pad
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(x_train)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(x_train)
padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(x_test)
testing_padded = pad_sequences(testing_sequences,maxlen = max_length, truncating = trunc_type)

#single layer LSTM
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

#GRU model
# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
#     tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
#     tf.keras.layers.Dense(6, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

model.compile(
    loss = 'binary_crossentropy',optimizer = 'adam',metrics = ['accuracy']
)
model.summary()
history = model.fit(
    padded, y_train, 
    epochs = 5, 
    validation_data = (testing_padded, y_test)
)


def IsFake(text):
    text=[text]
    sequences = tokenizer.texts_to_sequences(text)
    padded = pad_sequences(
        sequences,maxlen = max_length, truncating = trunc_type
    )
    pred = model.predict(padded)
    if pred > 0.5:
        print("This article is fake news.")
    else:
        print("This article is not fake news.")


reuters0= 'Aug 10 (Reuters) - The Texas Supreme Court on Tuesday blocked a rule protecting state Democratic lawmakers from arrest after they went to Washington to avoid a quorum as state Republicans attempt to pass voting restriction measures, the Washington Post reported. Texas Governor Greg Abbott, a Republican, vowed to arrest more than 50 Democratic lawmakers who left the state on July 12, denying the state House of Representatives the quorum required to approve the voting limits and other measures on his special-session agenda. On Sunday, Texas District Judge Brad Urrutia issued a temporary restraining order in a case filed by 19 Texas House Democrats against Abbott, challenging the state  s power to arrest them for political purposes, the Washington Post reported. Texas is among several Republican-led states pursuing new voting restrictions in the name of enhancing election security. Former President Donald Trump has claimed falsely that the presidential election last November was stolen from him through widespread fraud. '
reuters1= 'WASHINGTON, Aug 10 (Reuters) - Next January will be the earliest possible start of any trial for members and associates of the Oath Keepers militia movement facing charges for rioting at the U.S. Capitol, giving prosecutors and defense lawyers time to examine evidence and prepare, attorneys told a court hearing on Tuesday. Four people connected to the right-wing Oath Keepers have already pleaded guilty to riot-related charges. Lawyers for 16 people facing felony riot charges appeared at a status hearing before U.S. District Judge Amit Mehta on Tuesday. Eugene Rossi, a lawyer for defendant William Isaacs, said prosecutors had deluged defense lawyers with an "avalanche of documents" like the flow "out of a fire hydrant" and extensive videos from Jan. 6, when supporters of then-President Donald Trump sought to block Congress from certifying Joe Biden s election victory. Michelle Peterson, a public defender representing accused rioter Jessica Watkins, who is still in pre-trial custody, complained that there were areas of the Capitol without government video cameras but where amateur videographers may have captured "exculpatory" behavior by some participants.  A government affidavit said defendants traveled to Washington carrying paramilitary gear and wearing clothes with Oath Keepers insignias. At the Capitol, investigators said, 10 defendants formed a "stack" which marched single file up stairs into the building, with hands on the person in front, then forced their way through the central Rotunda. More than 535 people face charges from Jan 6. Also on Tuesday, Virginia residents Douglas Sweet, 59, and Cindy Fitchett, 60, who posted a video of herself in the Capitol declaring "Patriots Arise", pleaded guilty before U.S. District Judge Carl Nichols to misdemeanor charges carrying maximum six month prison terms. Sentencing was set for November. Illinois resident Bradley Rukstales also signed a plea agreement, but it was not officially entered. '
IsFake(reuters0)


def getProb(text):
    text=[text]
    sequences = tokenizer.texts_to_sequences(text)
    padded = pad_sequences(
        sequences,maxlen = max_length, truncating = trunc_type
    )
    print(model.predict(padded))

getProb(reuters0)






