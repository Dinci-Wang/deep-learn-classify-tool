# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 17:19:12 2020

@author: Eddie
"""
# import math, copy
import itertools
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import models
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPool1D, Flatten, Activation, BatchNormalization
from tensorflow.keras.layers import TimeDistributed, GRU, LSTM, Dropout, Bidirectional, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint

from AttentionWithContext import AttentionWithContext


import os
# print(os.listdir())
# #%%

# df = pd.read_csv("train_data.csv", header=None)#22396資料1400點1label
# label = pd.read_csv("multi_label.csv", header=None)#22396資料108病徵

# df.head()
# df.info()

# # label = df[1400].value_counts()
# # print(label)

# M = df.values
# X = M[:, :-1]#22396資料1400點
# # y = M[:, -1].astype(str)
# y = label

# del df
# #del df2
# del M

#%%
data = np.load(r"../fornn\p_wave_list.npy", allow_pickle=True)
label = np.load(r"../fornn\p_wave_level.npy")
#%%
"""# Split"""
dataII = data[:,:,:,0]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) #contrl label cat

X_train, X_test, y_train, y_test = train_test_split(dataII, label, test_size=0.20) #contrl label cat
#
X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)





# X_train = X_train.astype('float64')
# X_test = X_test.astype('float64')



print("X_train :", X_train.shape)
# print("NSR_y_train :", NSR_y_train.shape)
print("X_test :", X_test.shape)
# print("NSR_y_test :", NSR_y_test.shape)

#%%
"""# Model
"""
epochs = 50
batch_size = 175

n_obs, seq, feature = X_train.shape

# feature=700
K.clear_session()
model = models.Sequential()
# model.add(Input(shape=(seq, feature, depth)))
model.add(Masking(mask_value=-999.0, input_shape=(seq, feature)))

model.add(LSTM(256, return_sequences=True, input_shape=(seq, feature)))

model.add(AttentionWithContext())
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation='relu'))

#model.add(BatchNormalization())
#model.add(Dense(256, activation='relu'))

#model.add(Dense(128, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.summary()

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
 
    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[f1, 'accuracy'])

history = model.fit(X_train, y_train, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    verbose=1,
                    validation_data=(X_test, y_test))

y_pred = model.predict(X_test)
y_pred =(y_pred>0.4)
list(y_pred)




# print(classification_report(NSR_y_test.argmax(axis=1), y_pred.argmax(axis=1)))
print(classification_report(y_test, y_pred))


#%%

"""Result
"""

# model.save("NSR1.h5")

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(1)
plot_confusion_matrix(cnf_matrix, classes=["0","1","2"],
                      title='Confusion matrix, without normalization')
plt.show()

plt.figure(2)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

# test_loss, test_acc = model.evaluate(X_test, NSR_y_test, verbose=2)
plt.show()

# print(test_acc)
