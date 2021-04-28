#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy import array
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from sklearn.model_selection import train_test_split
import re
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


# In[2]:


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
        logical_gpus=tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus),"Physical GPUS",len(logical_gpus),"Logical GPUs")
    except RuntimeError as e:
        print(e)


# In[3]:


train = pd.read_csv("train.csv")
train['tweet']=train['tweet'].apply(lambda x:x.lower())
train['tweet']=train['tweet'].apply((lambda x:re.sub('[^a-zA-z0-9-9\s]','',x)))


# In[4]:


tweet_data = [i for i in train['tweet']]
response_label = [i for i in train['label']]


# In[ ]:


#max_words parameter
max_words = 5000
t = Tokenizer(num_words=max_words)
t.fit_on_texts(tweet_data)
sequences= t.texts_to_sequences(tweet_data)
#one_hot_results = t.texts_to_matrix(tweet_data,mode='binary')
word_index = t.word_index

#paramaters for embedding layer
vocab_size = len(word_index)+1
max_len = 50
embedding_dim = 100

data = pad_sequences(sequences,maxlen=max_len,padding='post')
labels = np.asarray(response_label)

#split into train and test data
x_train,x_val,y_train,y_val = train_test_split(data,labels,test_size =.3, random_state=30)


# In[ ]:


# for embedding input -> vocab_size, embedding_dim, input_length=maxlen)
#initial model without pretrained embedding

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim,input_length = max_len))
model.add(LSTM(256, dropout=.3,recurrent_activation='sigmoid',activation='tanh',use_bias=True,unroll=False))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))
model.summary()
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history = model.fit(x_train,y_train,epochs=10,batch_size=32,validation_data=(x_val,y_val))


# In[ ]:


#plot for model1, no pretrained embedding
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[ ]:


#import pretrained embedding to constraint weights for embedding matrix
import os
glove_dir = '/Users/12038/Downloads/glove.6B'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'),encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


# In[ ]:


#same model with pretrained embedding
#for pretrained embedding, vocab_size must equal max_words
#max_words parameter

vocab_size = max_words
model2 = Sequential()
model2.add(Embedding(vocab_size, embedding_dim,input_length = max_len))
model2.add(LSTM(256, dropout=.3,recurrent_activation='sigmoid',activation='tanh',use_bias=True,unroll=False))
model2.add(Flatten())
model2.add(Dense(1,activation='sigmoid'))
model2.summary()
model2.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history2 = model2.fit(x_train,y_train,epochs=10,batch_size=32,validation_data=(x_val,y_val))
model2.layers[0].set_weights([embedding_matrix])
model2.layers[0].trainable = False


# In[ ]:


#plot for model2, pretrained embedding
import matplotlib.pyplot as plt
acc = history2.history['acc']
val_acc = history2.history['val_acc']
loss = history2.history['loss']
val_loss = history2.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[ ]:


# for embedding input -> vocab_size, embedding_dim, input_length=maxlen)
#initial model without pretrained embedding
#doubling number of hidden units
vocab_size = len(word_index)+1
model3 = Sequential()
model3.add(Embedding(round(vocab_size*2), embedding_dim,input_length = max_len))
model3.add(LSTM(256, dropout=.3,recurrent_activation='sigmoid',activation='tanh',use_bias=True,unroll=False))
model3.add(Flatten())
model3.add(Dense(1,activation='sigmoid'))
model3.summary()
model3.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history3 = model3.fit(x_train,y_train,epochs=10,batch_size=32,validation_data=(x_val,y_val))


# In[ ]:


#plot for model3, pretrained embedding with doubled hidden units
import matplotlib.pyplot as plt
acc = history3.history['acc']
val_acc = history3.history['val_acc']
loss = history3.history['loss']
val_loss = history3.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[ ]:


#halving number of hidden units
model4 = Sequential()
model4.add(Embedding(round(vocab_size/2), embedding_dim,input_length = max_len))
model4.add(LSTM(256, dropout=.3,recurrent_activation='sigmoid',activation='tanh',use_bias=True,unroll=False))
model4.add(Flatten())
model4.add(Dense(1,activation='sigmoid'))
model4.summary()
model4.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history4 = model4.fit(x_train,y_train,epochs=10,batch_size=32,validation_data=(x_val,y_val))


# In[ ]:


#plot for model4, pretrained embedding with halvved hidden units
import matplotlib.pyplot as plt
acc = history4.history['acc']
val_acc = history4.history['val_acc']
loss = history4.history['loss']
val_loss = history4.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[ ]:


#model with different numbers of all hidden units, including vocab_size, embedding dim, maxlen, and ltsm

#max_words parameter,increased
max_words = 10000
t = Tokenizer(num_words=max_words)
t.fit_on_texts(tweet_data)
sequences= t.texts_to_sequences(tweet_data)
one_hot_results = t.texts_to_matrix(tweet_data,mode='binary')
word_index = t.word_index

#paramaters for embedding layer, increased
vocab_size = len(word_index)+1
max_len = 45
embedding_dim = 60

data = pad_sequences(sequences,maxlen=max_len,padding='post')
labels = np.asarray(response_label)

#split into train and test data
x_train,x_val,y_train,y_val = train_test_split(data,labels,test_size =.3, random_state=30)

model5 = Sequential()
model5.add(Embedding(vocab_size, embedding_dim,input_length = max_len))
model5.add(LSTM(128, dropout=.3,recurrent_activation='sigmoid',activation='tanh',use_bias=True,unroll=False))
model5.add(Flatten())
model5.add(Dense(1,activation='sigmoid'))
model5.summary()
model5.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history5 = model5.fit(x_train,y_train,epochs=10,batch_size=32,validation_data=(x_val,y_val))


# In[ ]:


import matplotlib.pyplot as plt
acc = history5.history['acc']
val_acc = history5.history['val_acc']
loss = history5.history['loss']
val_loss = history5.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[ ]:


#model with different numbers of all hidden units, including vocab_size, embedding dim, maxlen, and ltsm

#max_words parameter,decreased
max_words = 4000
t = Tokenizer(num_words=max_words)
t.fit_on_texts(tweet_data)
sequences= t.texts_to_sequences(tweet_data)
one_hot_results = t.texts_to_matrix(tweet_data,mode='binary')
word_index = t.word_index

#paramaters for embedding layer, decreased
vocab_size = len(word_index)+1
max_len = 40
embedding_dim = 50

data = pad_sequences(sequences,maxlen=max_len,padding='post')
labels = np.asarray(response_label)

#split into train and test data
x_train,x_val,y_train,y_val = train_test_split(data,labels,test_size =.3, random_state=30)

model6 = Sequential()
model6.add(Embedding(vocab_size, embedding_dim,input_length = max_len))
model6.add(LSTM(128, dropout=.3,recurrent_activation='sigmoid',activation='tanh',use_bias=True,unroll=False))
model6.add(Flatten())
model6.add(Dense(1,activation='sigmoid'))
model6.summary()
model6.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history6 = model6.fit(x_train,y_train,epochs=10,batch_size=32,validation_data=(x_val,y_val))


# In[ ]:


import matplotlib.pyplot as plt
acc = history6.history['acc']
val_acc = history6.history['val_acc']
loss = history6.history['loss']
val_loss = history6.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[ ]:


#stacking ltsm layers
#model with different numbers of all hidden units, including vocab_size, embedding dim, maxlen, and ltsm

#max_words parameter
max_words = 10000
t = Tokenizer(num_words=max_words)
t.fit_on_texts(tweet_data)
sequences= t.texts_to_sequences(tweet_data)
one_hot_results = t.texts_to_matrix(tweet_data,mode='binary')
word_index = t.word_index

#paramaters for embedding layer
vocab_size = len(word_index)+1
max_len = 50
embedding_dim = 50

data = pad_sequences(sequences,maxlen=max_len,padding='post')
labels = np.asarray(response_label)

#split into train and test data
x_train,x_val,y_train,y_val = train_test_split(data,labels,test_size =.3, random_state=30)

model7 = Sequential()
model7.add(Embedding(vocab_size, embedding_dim,input_length = max_len))
model7.add(LSTM(32, dropout=.3,recurrent_activation='sigmoid',activation='tanh',use_bias=True,unroll=False,return_sequences=True))
model7.add(LSTM(64, dropout=.3,recurrent_activation='sigmoid',activation='tanh',use_bias=True,unroll=False))
model7.add(Flatten())
model7.add(Dense(1,activation='sigmoid'))
model7.summary()
model7.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history7 = model7.fit(x_train,y_train,epochs=10,batch_size=32,validation_data=(x_val,y_val))


# In[ ]:


import matplotlib.pyplot as plt
acc = history7.history['acc']
val_acc = history7.history['val_acc']
loss = history7.history['loss']
val_loss = history7.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[5]:


#adding dense layer
#max_words parameter
max_words = 5000
t = Tokenizer(num_words=max_words)
t.fit_on_texts(tweet_data)
sequences= t.texts_to_sequences(tweet_data)
#one_hot_results = t.texts_to_matrix(tweet_data,mode='binary')
word_index = t.word_index

#paramaters for embedding layer
vocab_size = len(word_index)+1
max_len = 50
embedding_dim = 50

data = pad_sequences(sequences,maxlen=max_len,padding='post')
labels = np.asarray(response_label)

#split into train and test data
x_train,x_val,y_train,y_val = train_test_split(data,labels,test_size =.3, random_state=30)
model8 = Sequential()
model8.add(Embedding(vocab_size, embedding_dim,input_length = max_len))
model8.add(LSTM(32, dropout=.3,recurrent_activation='sigmoid',activation='tanh',use_bias=True,unroll=False,return_sequences=True))
model8.add(LSTM(64, dropout=.3,recurrent_activation='sigmoid',activation='tanh',use_bias=True,unroll=False))
model8.add(Dense(64,activation='sigmoid'))
model8.add(Flatten())
model8.add(Dense(1,activation='sigmoid'))
model8.summary()
model8.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history8 = model8.fit(x_train,y_train,epochs=10,batch_size=32,validation_data=(x_val,y_val))


# In[6]:


import matplotlib.pyplot as plt
acc = history8.history['acc']
val_acc = history8.history['val_acc']
loss = history8.history['loss']
val_loss = history8.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[7]:


#halving sequence length only, previous was 50
max_len = 25
# for embedding input -> vocab_size, embedding_dim, input_length=maxlen)
data = pad_sequences(sequences,maxlen=max_len,padding='post')
labels = np.asarray(response_label)

#split into train and test data
x_train,x_val,y_train,y_val = train_test_split(data,labels,test_size =.3, random_state=30)
model9 = Sequential()
model9.add(Embedding(vocab_size, embedding_dim,input_length = max_len))
model9.add(LSTM(256, dropout=.3,recurrent_activation='sigmoid',activation='tanh',use_bias=True,unroll=False))
model9.add(Flatten())
model9.add(Dense(1,activation='sigmoid'))
model9.summary()
model9.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history9 = model9.fit(x_train,y_train,epochs=10,batch_size=32,validation_data=(x_val,y_val))


# In[8]:


import matplotlib.pyplot as plt
acc = history9.history['acc']
val_acc = history9.history['val_acc']
loss = history9.history['loss']
val_loss = history9.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[9]:


#doubling from 100
max_len = 200
# for embedding input -> vocab_size, embedding_dim, input_length=maxlen)
data = pad_sequences(sequences,maxlen=max_len,padding='post')
labels = np.asarray(response_label)

#split into train and test data
x_train,x_val,y_train,y_val = train_test_split(data,labels,test_size =.3, random_state=30)
model10 = Sequential()
model10.add(Embedding(vocab_size, 250,input_length = max_len))
model10.add(LSTM(256, dropout=.3,recurrent_activation='sigmoid',activation='tanh',use_bias=True,unroll=False))
model10.add(Flatten())
model10.add(Dense(1,activation='sigmoid'))
model10.summary()
model10.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history10 = model10.fit(x_train,y_train,epochs=10,batch_size=32,validation_data=(x_val,y_val))


# In[10]:


import matplotlib.pyplot as plt
acc = history10.history['acc']
val_acc = history10.history['val_acc']
loss = history10.history['loss']
val_loss = history10.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[ ]:


#confusion matrices from best 3 models
#models 6,3,2 and 1 performed the best on the test set

# model 1 confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
predictions = model.predict(x_val)
y_pred = (predictions > .5)
matrix = confusion_matrix(y_val,y_pred)
labels = [1,0]

ax= plt.subplot()
sns.heatmap(matrix, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['1', '0']); ax.yaxis.set_ticklabels(['0', '1']);

# model1 predictions
twt = ['women are great #selfaffirmation']
twt = t.texts_to_sequences(twt)
twt=pad_sequences(twt,maxlen=50)
sentiment = model.predict(twt,batch_size=1)
print(sentiment)


# In[ ]:


#confusion matrices from best 3 models
#models 6,3,2 and 1 performed the best on the test set

# model 2 confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
predictions = model2.predict(x_val)
y_pred = (predictions > .5)
matrix = confusion_matrix(y_val,y_pred)
labels = [1,0]

ax= plt.subplot()
sns.heatmap(matrix, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['1', '0']); ax.yaxis.set_ticklabels(['0', '1']);

# model2 predictions
max_len=50
twt = ['its unbelievable that in the 21st century wed need something like this. again. #neverump  #xenophobia ',' i was into "pimps up hoes down" "hookers at the point" documentaries by brent owens and the cathouse series as a kid lol']
twt = t.texts_to_sequences(twt)
twt=pad_sequences(twt,maxlen=max_len)
sentiment = model2.predict(twt,batch_size=1)
print(sentiment)


# In[ ]:


# model 2 confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
predictions = model3.predict(x_val)
y_pred = (predictions > .5)
matrix = confusion_matrix(y_val,y_pred)
labels = [1,0]

ax= plt.subplot()
sns.heatmap(matrix, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['1', '0']); ax.yaxis.set_ticklabels(['0', '1']);

# model3 predictions
max_len=50
twt = ['its unbelievable that in the 21st century wed need something like this. again. #neverump  #xenophobia ',' i was into "pimps up hoes down" "hookers at the point" documentaries by brent owens and the cathouse series as a kid lol']
twt = t.texts_to_sequences(twt)
twt=pad_sequences(twt,maxlen=max_len)
sentiment = model2.predict(twt,batch_size=1)
for i in sentiment:
    if i > .5:
        print('tweet is negative')
    else:
        print('tweet is positive')


# In[ ]:




