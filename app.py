# Name: Sashen Moodley
# Student Number: 219006946

# PLEASE NOTE: AS THE SUBMISSION SIZE IS LIMITED TO 50MB I HAVE REMOVED THE "sentiment_dataset.csv" AND
# "glove.6B.300d.txt" FILES. THESE ARE REQUIRED TO RUN THE PROGRAM.
# THE "sentiment_dataset.csv" CAN BE FOUND ON THE ONEDRIVE LINK: https://stuukznac-my.sharepoint.com/personal/jemberee_ukzn_ac_za/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fjemberee%5Fukzn%5Fac%5Fza%2FDocuments%2FCOMP703%20Assignment%20II%20Dataset&ga=1
# OR EQUIVALENTLY FROM: https://www.kaggle.com/datasets/kazanova/sentiment140

# THE REQUIRED "glove.6B.300d.txt" FILE CAN BE found at: http://nlp.stanford.edu/data/glove.6B.zip
# ONCE UNZIPPED, only the "glove.6B.300d.txt" FILE IS REQUIRED

import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from typing import List
from wordcloud import WordCloud
from keras import backend as K
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Dense, Dropout, LSTM, GRU
from keras import Sequential
from keras.optimizers import adam_v2
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
# from nltk.corpus import stopwords
# from nltk.stem.wordnet import WordNetLemmatizer  # Not required in final build

rcParams.update({'figure.autolayout': True})  # To ensure proper scaling of visuals

# --------------Global Parameters----------------------
MAX_NO_WORDS = 100000  # Only 100000 from the processed vocab will be considered
EMBEDDING_DIM = 300  # Inline with GloVe embedding file (300 dim)
MAX_LENGTH = 30  # Maximum sentence length (used for padding)
EPOCHS = 80
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

# --------------Plotting Function------------------------
def plot_graphs(history, string, title):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.title(title)
    plt.show()


# ----------Legacy Metric Functions------------------------------
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# -------------- Denoising Function ----------------------
def denoise(tweets: List[str]) -> List[str]:
    processed_tweets = []
    # stop_words = set(stopwords.words('english'))
    # word_lemmatizer = WordNetLemmatizer()

    # Defining dictionary containing all emojis with their meanings - provided by referenced person in report
    emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
              ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
              ':-@': 'shocked', ':@': 'shocked', ':-$': 'confused', ':\\': 'annoyed',
              ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
              '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
              '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
              ';-)': 'wink', 'O:-)': 'angel', 'O*-)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'}

    for tweet in tweets:
        tweet = tweet.lower()  # Tokenizer will also transform to lowercase
        tweet = re.sub(r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)", '', tweet)  # removing links
        tweet = re.sub(r"\@\w+|\#", '', tweet)  # remove mentions(and what follows) and hashtags(just the '#')
        tweet = re.sub("[^a-zA-Z0-9]", ' ', tweet)  # removing all non-alphabets
        tweet = re.sub(r"(.)\1\1+", r"\1\1", tweet)  # replacing 3 or more consecutive letters with 2

        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])

        processed_tweets.append(tweet)

        # This has been commented out to show the word-stop and lemmatisation process (if it were to be considered)

        #tweet = ' '.join([word for word in tweet.split() if word not in stop_words])  # removing stop words

        # final_tweet_words = ''
        # for word in tweet.split():
        #     if len(word) > 1:  # removing short words (less than 2 letters in length)
        #         word = word_lemmatizer.lemmatize(word)
        #         final_tweet_words += (word+' ')

        # processed_tweets.append(tweet)

    return processed_tweets


# --------------DATA PRE-PROCESSING------------------------
data = pd.read_csv('sentiment_dataset.csv',
                   names=['Target', 'ID', 'Date', 'Flag', 'User', 'Text'],
                   encoding='ISO-8859-1')
print("CSV data read into dataframe")

# Converting dataframe subset into list of tuples for randomized sklearn train/test split
data = data[['Target', 'Text']]  # subset of dataframe where only necessary columns selected
data['Target'] = data['Target'].replace(4, 1)  # Represent positive sentiment as 1 rather than 4

# Getting data information
print(data['Target'].value_counts())
data_dist = data.groupby('Target').count().plot(kind='bar', legend=False, title='Distribution of Sentiment Labels')
data_dist.set_xticklabels(['Negative (0)', 'Positive (1)'], rotation=0)
plt.show()

# Separating data into separate lists
tweets, labels = list(data['Text']), list(data['Target'])

# Distribution of sentence lengths before processing
pre_sentence_lengths = [len(tweet.split()) for tweet in tweets]
print(f"Total words/tokens: {sum(pre_sentence_lengths)}")
print(f"Max tweet length: {max(pre_sentence_lengths)}")
print(f"Min tweet length: {min(pre_sentence_lengths)}")
print(f"Average tweet length: {sum(pre_sentence_lengths)/len(pre_sentence_lengths)}")
pre_length_hist = pd.DataFrame(data=pre_sentence_lengths, columns=["Preprocessed Sentence Lengths"])
pre_length_hist.hist(bins=60)
plt.show()

# Denoising the tweets
print("Denoising tweets . . .")
processed_tweets = denoise(tweets)
data['Text'] = processed_tweets
print("Denoising complete-------")

# Distribution of sentence lengths after processing
post_sentence_lengths = [len(tweet.split()) for tweet in processed_tweets]
print(f"Total words/tokens: {sum(post_sentence_lengths)}")
print(f"Max tweet length: {max(post_sentence_lengths)}")
print(f"Min tweet length: {min(post_sentence_lengths)}")
print(f"Average tweet length: {sum(post_sentence_lengths)/len(post_sentence_lengths)}")
post_length_hist = pd.DataFrame(data=post_sentence_lengths, columns=["Processed Sentence Lengths"])
post_length_hist.hist(bins=60)
plt.show()

# visualizing some keywords for positive and negative sentiment before proper word_indexing
print("Performing word cloud visualizations. . .")
neg_word_cloud = WordCloud(max_words=1000, width=1600, height=800, collocations=False).generate(" ".join(processed_tweets[:800000]))
plt.figure(figsize=(20, 20))
plt.imshow(neg_word_cloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud for Negative Sentiment", fontsize=40)
plt.show()
pos_word_cloud = WordCloud(max_words=1000, width=1600, height=800, collocations=False).generate(" ".join(processed_tweets[800000:]))
plt.figure(figsize=(20, 20))
plt.imshow(pos_word_cloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud for Positive Sentiment", fontsize=40)
plt.show()
print("Word cloud visualizations complete --------")

data_tuple_list = list(data.itertuples(index=False, name=None))  # Converting dataframe to list of tuples
training_set, testing_set = train_test_split(data_tuple_list, train_size=0.90, test_size=0.10, random_state=27)

training_tweets = []
training_labels = []
testing_tweets = []
testing_labels = []

print(f"Training set size: {len(training_set)}")
print(f"Testing set size: {len(testing_set)}")

# Separating the labels from sentences for both training and testing
for label, sentence in training_set:
    training_labels.append(label)
    training_tweets.append(str(sentence))
for label, sentence in testing_set:
    testing_labels.append(label)
    testing_tweets.append(str(sentence))

# Converting to type numpy array for keras
training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)

# Tokenization and padding
print("Beginning tokenization and padding")
tokenizer = Tokenizer(num_words=MAX_NO_WORDS, lower=True, oov_token="<OOV>")
# This method creates the vocabulary index based on word frequency. It's a word -> index dictionary (every word gets a
# unique integer value). Lower integer means more frequent word.
tokenizer.fit_on_texts(training_tweets)
word_index = tokenizer.word_index
VOCAB_SIZE = len(word_index)
print(f"Vocabulary size: {VOCAB_SIZE}")

# It basically takes each word in the text and replaces it with its corresponding
# integer value from the word_index dictionary
training_sequences = tokenizer.texts_to_sequences(training_tweets)  # By default, all punctuation is removed
training_padded = pad_sequences(sequences=training_sequences, maxlen=MAX_LENGTH, truncating='post')
testing_sequences = tokenizer.texts_to_sequences(testing_tweets)
testing_padded = pad_sequences(testing_sequences, maxlen=MAX_LENGTH, truncating='post')
print("Tokenizing and Padding completed---------")

# Encoding the labels
print("Encoding labels. . .")
encoder = LabelEncoder()
encoder.fit(training_labels)  # training data labels

training_labels = encoder.transform(training_labels)
testing_labels = encoder.transform(testing_labels)
training_labels = training_labels.reshape(-1, 1)
testing_labels = testing_labels.reshape(-1, 1)
print("Label encoding complete----------")


# Performing manual (pre-trained) Glove encoding (word embedding)
print("Performing GloVe word embeddings. . .")
embeddings_index = {}
with open('glove.6B.300d.txt', encoding='utf8') as glove_file:
    for line in glove_file:
        values = line.split()
        word = value = values[0]
        coefs = np.array(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print(f"Found {len(embeddings_index)} word vectors")
embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
for word, index in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
print("GloVe word embeddings complete--------")


# ---------------MODEL LEARNING, VALIDATION AND EVALUATION----------------------------
#------- LSTM
# Initializing the model
print("Setting up LSTM model. . .")
lstm_model = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH, weights=[embedding_matrix], trainable=False),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1, activation='sigmoid') # Since we made positive sentiment '1' then this will be fine, otherwise softmax would be more appropriate
])
lstm_model.summary()
print("LSTM Model setup complete------")

print("Compiling LSTM model. . .")
lstm_model.compile(loss='binary_crossentropy', optimizer=adam_v2.Adam(learning_rate=LEARNING_RATE),
                  metrics=['accuracy', 'mean_squared_error', f1_m, precision_m, recall_m])
ReduceLROn_Plateau = ReduceLROnPlateau(factor=0.1, min_lr=0.01, monitor='val_loss', verbose=1)
Model_Checkpoint = ModelCheckpoint(filepath='LSTM checkpoints\model-{epoch:03d}-{val_loss}-{accuracy:03f}-{val_accuracy:03f}.h5', verbose=1, monitor='val_loss', mode='min', save_best_only=True)
print("LSTM Compilation complete--------")

print("Fitting LSTM model. . .")
history = lstm_model.fit(x=training_padded, y=training_labels, epochs=EPOCHS, validation_split=0.1,
                        batch_size=BATCH_SIZE, callbacks=[ReduceLROn_Plateau, Model_Checkpoint])
print("LSTM Fitting complete---------")

plot_graphs(history, 'accuracy', 'LSTM Accuracy')
plot_graphs(history, 'loss', 'LSTM Loss')
plot_graphs(history, 'mean_squared_error', 'LSTM MSE')
plot_graphs(history, 'f1_m', 'LSTM F-1 Score')
plot_graphs(history, 'precision_m', 'LSTM Precision')
plot_graphs(history, 'recall_m', 'LSTM Recall')

print("Evaluating LSTM model. . .")
start_time = time.time()
scores = lstm_model.evaluate(x=testing_padded, y=testing_labels)
end_time = time.time()
print("LSTM Evaluation complete---------")
print(f"LSTM Evaluation Time taken: {end_time - start_time}")
print("---------LSTM EVALUATION STATS---------")
print(f"Loss: {scores[0]}\nAccuracy: {scores[1]}\nMean Squared Error: {scores[2]}\nF1-Score: {scores[3]}\nPrecision: {scores[4]}\nRecall: {scores[5]}")

print("Making LSTM predictions")
start_time = time.time()
predictions = lstm_model.predict(x=testing_padded, batch_size=BATCH_SIZE)
end_time = time.time()
print("LSTM Predictions complete------")
print(f"LSTM Prediction Time taken: {end_time - start_time}")

rounded_predictions = []
for prediction in predictions:
    if prediction[0] > 0.5:
        rounded_predictions.append(1)
    else:
        rounded_predictions.append(0)

confuse_matrix = confusion_matrix(y_true=testing_labels, y_pred=rounded_predictions)
disp = ConfusionMatrixDisplay(confuse_matrix)
disp.plot()
plt.title("LSTM Confusion Matrix")
plt.show()
print(classification_report(y_true=testing_labels, y_pred=rounded_predictions))

# save model and architecture to single file
lstm_model.save("LSTM_Last.h5")
print("Saved final (last epoch) LSTM model to disk")


#------- GRU
# Initializing the model
print("Setting up GRU model. . .")
gru_model = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH, weights=[embedding_matrix], trainable=False),
    Dropout(0.2),
    GRU(64),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
gru_model.summary()
print("GRU Model setup complete------")

print("Compiling GRU model. . .")
gru_model.compile(loss='binary_crossentropy', optimizer=adam_v2.Adam(learning_rate=LEARNING_RATE),
                  metrics=['accuracy', 'mean_squared_error', f1_m, precision_m, recall_m])
ReduceLROn_Plateau = ReduceLROnPlateau(factor=0.1, min_lr=0.01, monitor='val_loss', verbose=1)
Model_Checkpoint = ModelCheckpoint(filepath='GRU checkpoints\model-{epoch:03d}-{val_loss}-{accuracy:03f}-{val_accuracy:03f}.h5', verbose=1, monitor='val_loss', mode='min', save_best_only=True)
print("GRU Compilation complete--------")

print("Fitting GRU model. . .")
history = gru_model.fit(x=training_padded, y=training_labels, epochs=EPOCHS, validation_split=0.1,
                        batch_size=BATCH_SIZE, callbacks=[ReduceLROn_Plateau, Model_Checkpoint])
print("GRU Fitting complete---------")

plot_graphs(history, 'accuracy', 'GRU Accuracy')
plot_graphs(history, 'loss', 'GRU Loss')
plot_graphs(history, 'mean_squared_error', 'GRU MSE')
plot_graphs(history, 'f1_m', 'GRU F-1 Score')
plot_graphs(history, 'precision_m', 'GRU Precision')
plot_graphs(history, 'recall_m', 'GRU Recall')

print("Evaluating GRU model. . .")
start_time = time.time()
scores = gru_model.evaluate(x=testing_padded, y=testing_labels)
end_time = time.time()
print("GRU Evaluation complete---------")
print(f"GRU Evaluation Time taken: {end_time - start_time}")
print("---------GRU EVALUATION STATS---------")
print(f"Loss: {scores[0]}\nAccuracy: {scores[1]}\nMean Squared Error: {scores[2]}\nF1-Score: {scores[3]}\nPrecision: {scores[4]}\nRecall: {scores[5]}")

print("Making GRU predictions")
start_time = time.time()
predictions = gru_model.predict(x=testing_padded, batch_size=BATCH_SIZE)
end_time = time.time()
print("GRU Predictions complete------")
print(f"GRU Prediction Time taken: {end_time - start_time}")

rounded_predictions = []
for prediction in predictions:
    if prediction[0] > 0.5:
        rounded_predictions.append(1)
    else:
        rounded_predictions.append(0)

confuse_matrix = confusion_matrix(y_true=testing_labels, y_pred=rounded_predictions)
disp = ConfusionMatrixDisplay(confuse_matrix)
disp.plot()
plt.title("GRU Confusion Matrix")
plt.show()
print(classification_report(y_true=testing_labels, y_pred=rounded_predictions))

# save model and architecture to single file
gru_model.save("GRU_LAST.h5")
print("Saved final (last epoch) GRU model to disk")