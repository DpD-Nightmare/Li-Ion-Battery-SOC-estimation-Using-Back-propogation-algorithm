#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 22:00:27 2019

@author: dhrupad
"""
from keras.datasets import imdb
(train_data, train_labels),(test_data, test_labels)=imdb.load_data(num_words=10000)
print(train_data[0])
print(train_labels[0])

#word decoding
word_index=imdb.get_word_index()
reverse_word_index=dict([(value,key) for (value,key) in word_index.items()])
decord_review=' '.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])
print(decord_review)

import numpy as np

def vectorize_sequence(sequences,dimension=10000):
    results=np.zeros((len(sequences),dimension))
    for i, sequences in enumerate(sequences):
        results[i, sequences]=1.
    return results
x_train=vectorize_sequence(train_data)
x_test=vectorize_sequence(test_data)

print(x_train)
print(x_test)
y_train=np.asarray(train_labels).astype('float32')
y_test=np.asarray(test_labels).astype('float32')
