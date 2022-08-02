#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[210]:


# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Common Imports
import os
import string
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig as save_fig
get_ipython().run_line_magic('matplotlib', 'inline')
from collections import Counter
import pathlib
import random

import warnings
warnings.filterwarnings("ignore")

# Text Preprocessing
import re, string, unicodedata
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import SnowballStemmer
from string import punctuation

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

# For Model
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, GlobalAveragePooling1D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import AveragePooling1D, MaxPooling1D, Conv1D
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization

# Split Dataset and Standarize the Datasets
from sklearn.model_selection import train_test_split

# Imaging
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Evaluate Regression Models
from sklearn.metrics import accuracy_score, classification_report

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)


# In[91]:


# Checking Tensorflow version
tf.__version__


# In[92]:


# Checking Keras version
keras.__version__


# # Data Loading

# In[93]:


# Data Loading & Data Head
data = pd.read_csv('maps_data_with_target.csv')
data.head()


# In[94]:


# Data Tail

data.tail()


# In[95]:


# Copy data original

data_copy = data.copy()


# # Data Cleaning

# ### Drop Unuseful Variable

# In[96]:


# We will not use "Unnamed: 0" for analysis, so we will delete it
data.drop('Unnamed: 0', axis = 1, inplace = True)


# ### Data Shape (Num of Rows and Columns) & Data Type

# In[97]:


# Shape Data
print("Data Shape : ", data.shape)


# In[98]:


# Check Dataset columns type, missing values
data.info()


# This dataset has missing values, *RangeIndex* from 0 to 551, its type of data type are float and object.

# # Exploratory Data Analysis

# ## General Exploratory

# ### Check imbalance data target

# In[247]:


# plot imbalance on target class
sns.set_theme(style="darkgrid")
ax = sns.countplot(x="target", data = data)


# Here, as it can be seen, the data is imbalance.

# ### Explore the Categorical Features

# In[99]:


# initialize categorical_features
categorical_features = [feature for feature in data.columns if ((data[feature].dtypes=='O') & (feature not in ['fraudulent']))]

# length each feature
for feature in categorical_features:
    print('This {} feature has {} values'.format(feature,len(data[feature].unique())))

print("Len of categorical features :",len(categorical_features))


# In[100]:


# list of numerical variables
numerical_features = data.select_dtypes(exclude='object').columns
print('Number of numerical variables: ', len(numerical_features))

# visualise the numerical variables
data[numerical_features].head()


# ### Numerical Features Distribution and Anomaly

# In[101]:


# Visualizing data distribution with distplot 
x = plt.figure(figsize=(15, 10))

for i, j in enumerate(numerical_features):
    x.add_subplot(5, 2, i+1)
    sns.distplot(data[j], bins=15)
    x.tight_layout()

plt.tight_layout()


# In[102]:


# Visualizing data distribution with boxplot // checking outliers
x = plt.figure(figsize=(15, 10))
for i, j in enumerate(numerical_features):
    x.add_subplot(5, 2, i+1)
    sns.boxplot(data[j])
    x.tight_layout()
plt.tight_layout()


# ###  Relation between Numerical Features and Target Feature

# In[103]:


#boxplot to show target distribution with respect numerical features
plt.figure(figsize = (20,60), facecolor = 'white')
plotnumber = 1
for feature in numerical_features:
    ax = plt.subplot(12, 3, plotnumber)
    sns.boxplot(x = "target", y = data[feature], data = data)
    plt.xlabel(feature)
    plotnumber += 1
plt.show()


# ### Correlation between Numerical Features

# In[104]:


# Checking for correlation by heatmap
corr = data[numerical_features].corr() 
plt.figure(figsize=(15, 10))

sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 
            cmap='GnBu', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);


# ## Text Exploratory

# In[105]:


# we will fill nan values with space before we do the exploration
data.fillna(" ",inplace = True)


# In[106]:


# Merging all object variables except the variables with link to website
data['text'] = data['name'] + ' ' + data['type'] + ' ' + data['subtypes'] + ' ' + data['description'] + ' ' + data['owner_title']
data['text'][0]


# In[107]:


# initialize new dataframe from original data with only text and target variable
datax = data.copy()
del datax['name']
del datax['type']
del datax['subtypes']
del datax['city']
del datax['latitude']
del datax['longitude']
del datax['rating']
del datax['reviews']
del datax['photo']
del datax['street_view']
del datax['description']
del datax['owner_title']


# In[108]:


datax.head()


# In[109]:


# filtering
sea_post = datax[datax['target'] == 'Sea']
buildings_post = datax[datax['target'] == 'Buildings']
forest_post = datax[datax['target'] == 'Forest']
mountain_post = datax[datax['target'] == 'Mountain']
street_post = datax[datax['target'] == 'Street']


# ### Wordclouds

# In[110]:


# Defining function wor wordcloud
def show_wordcloud(data, name):
    # Collecting all words in the description
    joineddata = " ".join(desc for desc in data.text)
    print ("There are {} words in the combination of all {} place description.".format(len(joineddata), name))
    
    # Generate wordcloud
    wordcloud = WordCloud(background_color='white',
                    stopwords = set(STOPWORDS),
                    max_words = 1000,
                    random_state = 42,
                    width=1000, 
                    height=500)
    wordcloud.generate(joineddata)

    # display the word cloud
    plt.figure(figsize = (15,10), facecolor = 'white')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'"Wordcloud of {name} place description"', fontsize = 15)
    plt.show()


# #### *Wordcloud of sea class*

# In[111]:


# wordcloud of class target sea
show_wordcloud(sea_post, 'sea')


# #### *Wordcloud of buildings class*

# In[112]:


# wordcloud of class target buildings
show_wordcloud(buildings_post, 'buildings')


# #### *Wordcloud of forest class*

# In[115]:


# wordcloud of class target forest
show_wordcloud(forest_post, 'forest')


# #### *Wordcloud of mountain class*

# In[116]:


# wordcloud of class target mountain
show_wordcloud(mountain_post, 'mountain')


# #### *Wordcloud of street class*

# In[117]:


# wordcloud of class target street
show_wordcloud(street_post, 'street')


# # Data Preprocessing

# ### *Get Data for Model Inference*

# In[118]:


# Get Data for Model Inference

data_inf = data.sample(20, random_state = 0)


# In[119]:


# Remove Inference-Set from Dataset

data_train_test = data.drop(data_inf.index)


# In[120]:


# Reset Index

data_train_test.reset_index(drop = True, inplace = True)
data_inf.reset_index(drop = True, inplace = True)


# ### *Feature Selection*

# We have done feature selection on text exploration, merge all object column in `text` column and define new dataframe named `datax`.

# In[121]:


datax.head()


# In[212]:


# shuffling the data
datax = shuffle(datax)
datax.reset_index(inplace = True, drop = True)


# ### *Handling Missing Values*

# In[214]:


# Check Missing Values on data
datax.isnull().sum()


# ### *Cleaning Data*

# In[215]:


# Get the Independent Features
X = datax.drop('target',axis=1)

# Get the Dependent features
y = datax['target']


# In[124]:


# Function for cleaning the data
def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower() # text to lower case
    text = re.sub('\[.*?\]', '', text) # dropping text in square brackets
    text = re.sub('https?://\S+|www\.\S+', '', text) # drop links
    text = re.sub('<.*?>+', '', text) # drop text in <>
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # drop punctuatuion 
    text = re.sub('\n', '', text) # drop enter / new line
    text = re.sub('\w*\d\w*', '', text)
    return text


# In[217]:


# Applying clean_text function to data
uncleaned_corpus = X.copy()
uncleaned_corpus['text'] = uncleaned_corpus['text'].apply(lambda x:clean_text(x)) 


# In[218]:


# Defining corpus with cleaned data
ss = SnowballStemmer(language='english') 
corpus = []
for i in range(0, len(uncleaned_corpus)):
  decsr = uncleaned_corpus['text'][i]
  decsr = decsr.split()  # splitting data
  decsr = [ss.stem(word) for word in decsr if not word in stopwords.words('english')] # steeming setiap huruf dengan pengecualian kata yang ada dalam stopwords
  decsr = ' '.join(decsr)
  corpus.append(decsr)


# Why we use `SnowballSteemer` instead of using `PorterSteemer`?
# 
# ```
# Martin Porter (who invented the Porter Stemmer or Porter algorithm in 1980) also created Snowball Stemmer. The method utilized in this instance is more precise and is referred to as “English Stemmer” or “Porter2 Stemmer.” It is somewhat faster and more logical than the original Porter Stemmer.
# ```

# In[219]:


# defining corpus to dataframe
X['corpus'] = corpus
X.reset_index(inplace=True)


# ### *Feature Encoding*

# In[222]:


#defined vocabulary size 
voc_size = 400

# Encoding corpus with tf.one_hot encoder
enc_corps = [one_hot(words, voc_size) for words in corpus] 


# In[223]:


# Checking len corpus and enc_corps
print(len(enc_corps[0]))
print(len(corpus[0].split(' ')))


# In[224]:


# Encoding y target
y_enc = LabelEncoder().fit_transform(y)


# ### *Splitting Data*

# In[225]:


# Split between `X` final (Features) and `y` final (Target)

X_trains, X_test, y_trains, y_test = train_test_split(X['corpus'], np.array(y_enc), test_size=0.2, random_state = 0)


# ### *Tokenization*

# In[226]:


# Embedding corpus after encoded with equal lenght = 40
descr_length = 20 # set maximum length of all sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
vocab = tokenizer.word_index

# Only top num_words-1 most frequent words will be taken into account. Only words known by the tokenizer will be taken into account.
X_trains_word_idx = tokenizer.texts_to_sequences(X_trains) 
X_test_word_idx = tokenizer.texts_to_sequences(X_test)

# padding sequences
X_trains_padded_seqs = pad_sequences(X_trains_word_idx, maxlen = descr_length)
X_test_padded_seqs = pad_sequences(X_test_word_idx, maxlen = descr_length)


# ### *Define Validation Set*

# In[227]:


# Mengambil data dari X_trains untuk dijadikan sebagai data validasi
X_train, X_val = X_trains_padded_seqs[:360], X_trains_padded_seqs[360:]
y_train, y_val = y_trains[:360], y_trains[360:]


# # Model Defining

# ### *Building Pipeline*

# In[228]:


# Setting Autotune
AUTOTUNE = tf.data.AUTOTUNE

#Building a pipeline from a data that exists in memory
training_batches = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1024).batch(64).cache().prefetch(AUTOTUNE)
validation_batches = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64).cache().prefetch(AUTOTUNE)
testing_batches = tf.data.Dataset.from_tensor_slices((X_test_padded_seqs, y_test)).batch(64).cache().prefetch(AUTOTUNE)


# In[230]:


# Creating model lstm only
embedding_vector_features = 100 # output dimension
model = Sequential()
model.add(Embedding(len(vocab)+1, embedding_vector_features, input_length = descr_length)) # len(vocab)+1 is an input shape, we have initialized vocab variabel in tokenization section
model.add(LSTM(100))        # model lstm with 100 output dimension
model.add(Dropout(0.3))     # dropout layers
model.add(Dense(5, activation='softmax')) # output layer

model.compile(loss = 'sparse_categorical_crossentropy', optimizer= 'adam', metrics = 'accuracy')
# loss function with categorical_crossentropy because it's multiclass case

print(model.summary())


# # Model Training

# In[241]:


# ClearSession
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

# Compiling model with Seq-API
model_learn = model.fit(training_batches, epochs = 100, validation_data = validation_batches)


# In[242]:


# Graph plot of train process model
pd.DataFrame(model_learn.history).plot(figsize=(12, 8))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()


# Here as we can see, the loss on validation increases and exceeds 100% loss tresshold while accuracy shows a stable movement. On training, the model seems to perform well with zero loss and 100% accuracy. The increased movement of loss on validation perhaps caused by the data size and the imbalance of class data target.
# 
# With this result, maybe there should be a way to improve the model perfomance. So, let's try model improvement.

# # Model Improvement

# ### Defining Model Improvement

# In[234]:


# creating model lstm + depths CNN
main_input = Input(shape = (descr_length, ), dtype = 'int32') # set input

# embedding
embedder = Embedding(len(vocab)+1, embedding_vector_features, input_length = descr_length)
embed = embedder(main_input)

# lstm model
lstm = LSTM(100)(embed) # inisialisasi model lstm dengan output dimension 100
lstm = Dropout(0.3)(lstm) # dropout layer

# cnn part
cnn = Conv1D(256, 5, padding='same')(embed)
cnn = MaxPooling1D(3, 3, padding='same')(cnn)
cnn = Conv1D(128, 5, padding='same')(cnn)
cnn = MaxPooling1D(3, 3, padding='same')(cnn)
cnn = Conv1D(64, 3, padding='same')(cnn)
cnn = Flatten()(cnn)
cnn = Dropout(0.1)(cnn)
cnn = BatchNormalization()(cnn)
cnn = Dense(100, activation='relu')(cnn)
cnn = Dropout(0.1)(cnn)

# concat
lstm_cnn = concatenate([lstm, cnn], axis = -1)
flat = Flatten()(lstm_cnn)  # Flattening
drop = Dropout(0.2)(flat)
main_output = Dense(5, activation='softmax')(drop)
model_imp = Model(inputs = main_input, outputs = main_output)

# compiling
model_imp.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# summary
print(model_imp.summary())


# ### Training Model Improvement

# In[239]:


# ClearSession
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

# Compiling model with Seq-API
model_learn_imp = model_imp.fit(training_batches, epochs = 100, validation_data = validation_batches)


# In[240]:


# Graph plot of train process model
pd.DataFrame(model_learn_imp.history).plot(figsize=(15, 10))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()


# After improvement, the loss on validation appears to decrease while the accuracy increases from 0.7 to almost 0.9. Is this a good improvement, let's have a look on the classification report of both models.

# # Model Evaluation

# In[257]:


# Dictionary of each class
classes_dict = { 0:'Buildings',
                 1:'Forest', 
                 2:'Mountain', 
                 3:'Sea', 
                 4:'Street' }


# ### *Classification Report on Train Set*

# In[248]:


# predicting
y_pred = model.predict(X_train)
y_pred_imp = model_imp.predict(X_train)
y_pred = np.argmax(y_pred, axis = 1)
y_pred_imp = np.argmax(y_pred_imp, axis = 1)

# Classification all models (base and improved)
print("Sequential base model evaluation: \n", classification_report(y_train, y_pred))
print('-'*55)
print("Model improvement evaluation: \n", classification_report(y_train, y_pred_imp))
print('-'*55)


# ### *Classification Report on Test Set*

# In[249]:


# predicting
y_pred_test = model.predict(X_test_padded_seqs)
y_pred_imp_test = model_imp.predict(X_test_padded_seqs)
y_pred_test = np.argmax(y_pred_test, axis = 1)
y_pred_imp_test = np.argmax(y_pred_imp_test, axis = 1)

# Classification all models (base and improved)
print("Sequential base model evaluation: \n", classification_report(y_test, y_pred_test))
print('-'*55)
print("Model improvement evaluation: \n", classification_report(y_test, y_pred_imp_test))
print('-'*55)


# On training set, both models show a good performances with 100% score on all metrics. However, on training set, `f1-score` on improved model looks better than base model except for `street` class or `target` = 4 (there are only 2 data of this class on testing set). While on base model the class of `street` has one predicted correctly out of two data.
# 
# With the consideration of decreasing loss on validation, we will select the improved model for data inference.

# # Model Saving

# In[258]:


# Save the model in HDF5 format // to use for data inference

model_imp.save("text_classification.h5")


# In[261]:


# freeze model
for layer in model_imp.layers:
  layer.trainable = False

# Save model for backend
model_imp.save("text_classification")


# # Model Inference

# In[259]:


# Display Inference-Set

data_inf


# In[260]:


# feature selecting
data_inf['text'] = data_inf['name'] + ' ' + data_inf['type'] + ' ' + data_inf['subtypes'] + ' ' + data_inf['description'] + ' ' + data_inf['owner_title']
data_inf_final = data_inf[['target', 'text']]

# cleaning data
infdat = data_inf_final.drop('target', axis = 1)
infdat['text'] = infdat['text'].apply(lambda x:clean_text(x))

corpusinf = []
for i in range(0, len(infdat)):
  decsr = infdat['text'][i]
  decsr = decsr.split()  # splitting data
  decsr = [ss.stem(word) for word in decsr if not word in stopwords.words('english')] # steeming setiap huruf dengan pengecualian kata yang ada dalam stopwords
  decsr = ' '.join(decsr)
  corpusinf.append(decsr)

infdat['corpusinf'] = corpusinf
infdat.reset_index(inplace = True)

# encoding
inf_enc_corps = [one_hot(words, voc_size) for words in corpusinf]

# Tokenization
inf_word_idx = tokenizer.texts_to_sequences(infdat['corpusinf'])
inf_padded_seqs = pad_sequences(inf_word_idx, maxlen = descr_length)

# Loading model
model_ = keras.models.load_model("text_classification.h5")

# Predicting
y_pred_inf = model_.predict(inf_padded_seqs)
y_pred_inf = np.argmax(y_pred_inf, axis = 1)

# Concate between Inference-Set and Prediction
data_inf_finaldf = pd.concat([data_inf, pd.DataFrame(y_pred_inf, columns=['Prediction'])], axis=1)
data_inf_finaldf


# For model improvement :
# - We will collect more complete data (the data that we've been using for this project is scraped from google maps with many limitations), or collaborating with the local goverment to get more detailed information about the tourist attraction.

# ---
# 
# ## *`References`*
# **Exploratory Data Analysis :**
# 
# >   https://www.kaggle.com/code/aayush895/text-classification-using-keras <br>
#     https://www.kaggle.com/code/aashita/word-clouds-of-various-shapes/notebook<br>
#     (https://www.kaggle.com/code/junedism/spaceship-titanic-exploratory-data-analysis)
# 
# **Data Preprocessing Text :** 
# 
# >   https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/<br>
#     https://machinelearningmastery.com/clean-text-machine-learning-python/<br>
#     https://keras.io/api/preprocessing/text/#one_hot<br>
#     https://www.analyticsvidhya.com/blog/2021/11/an-introduction-to-stemming-in-natural-language-processing/<br>
#     https://www.datacamp.com/tutorial/wordcloud-python<br>
#     https://coderpad.io/regular-expression-cheat-sheet/<br>
#     https://towardsdatascience.com/building-a-one-hot-encoding-layer-with-tensorflow-f907d686bf39<br>
#     https://www.goeduhub.com/10643/practical-approach-word-embedding-simple-embedding-example<br>
#     https://www.tensorflow.org/api_docs/python/tf/keras/utils/pad_sequences<br>
#     https://colab.research.google.com/drive/1quIzzM4444f41LvSGMIDzNhZbBggUFZo#scrollTo=UbOU7Zh_RmXK (hacktiv8 material course)<br>
# 
# **Modelling :** 
# 
# >   https://www.tensorflow.org/tfx/tutorials/transform/census<br>
#     https://www.kaggle.com/code/sardiirfansyah/tensorflow-input-pipeline-prefetch-tf-data<br>
#     https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle<br>
#     https://medium.com/@ashraf.dasa/shuffle-the-batched-or-batch-the-shuffled-this-is-the-question-34bbc61a341f<br>
#     https://stackoverflow.com/questions/56227671/how-can-i-one-hot-encode-a-list-of-strings-with-keras<br>
#     https://towardsdatascience.com/cross-entropy-loss-function-f38c4ec8643e<br>
#     https://keras.io/api/layers/core_layers/embedding/<br>
#     https://www.baeldung.com/cs/bidirectional-vs-unidirectional-lstm<br>
#     https://www.kaggle.com/code/fanyuanlai/textcnn<br>
#     https://www.kaggle.com/code/tanvikurade/fake-job-postings-using-bidirectional-lstm/notebook<br>
#     https://medium.com/deep-learning-with-keras/lstm-understanding-the-number-of-parameters-c4e087575756<br>
#     https://medium.com/geekculture/10-hyperparameters-to-keep-an-eye-on-for-your-lstm-model-and-other-tips-f0ff5b63fcd4<br>
#     https://towardsdatascience.com/lstm-framework-for-univariate-time-series-prediction-d9e7252699e<br>
#     https://medium.com/@kangeugine/long-short-term-memory-lstm-concept-cb3283934359<br>
#     https://medium.com/ai-ml-at-symantec/should-we-abandon-lstm-for-cnn-83accaeb93d6<br>
#     https://analyticsindiamag.com/guide-to-text-classification-using-textcnn/<br>
#     https://keras.io/api/layers/pooling_layers/max_pooling1d/<br>
