########################################################################################
########################################################################################
###############         Sentiment prediction and             ###########################
###############       Sentiment Index construction           ###########################
###############                                              ###########################
########################################################################################
########################################################################################

# from __future__ import division
# import pickle
import pandas as pd
import numpy as np
import os
import re
import nltk
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K


def clean_text(textList):
    article_list = []
    expression = "[^a-zA-Z ]"
    for article in textList:
        sentenceList = []
        for sentence in article.split(','):
            if (len(sentence) > 40) :
                sentence = re.sub(expression, '', sentence)
                sentence = sentence.lower()  # Converts everything in news_text to lowercase
                sentenceList.append(sentence)
        article_list.append(sentenceList)
    return(article_list)

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall



path = "/Users/cathychen/OneDrive/NAS Backup/NASDAQ/NewsTopic/NewsTopic/Data"
os.chdir(path)

#  load NASDAQ overnight news articles

texts = pd.read_csv("overnight_2_utf8.csv")
articles = texts.article
# articles = texts.article[900:1000]
article_list = articles.tolist()
# len(article_list)
# 70431
article_list = clean_text(article_list)

###Calculate sentiment for each sentence in an article:
#(Note: Calculation might take a while, suppress output for larger dataset)

#Load one of the trained models
max_fatures = 5500
tokenizer = Tokenizer(num_words=max_fatures, split=' ',filters='!"#&()*+,-./:;<=>?@[\]^_`{"}~\t\n',lower = True)
filepath_a ='Naive.hdf5'
model = load_model(filepath_a,custom_objects={"precision": precision, "recall":recall})

article_sent = []
article_sent_profile = []
for i, article in enumerate(article_list):
    sentence_sent = []
    sentence_profile = []
    for j in range(len(article)):
        # print(sentence, end='\n')
        twt = [article[j]]
        tokenizer.fit_on_texts(twt)
        X_pred = tokenizer.texts_to_sequences(twt)
        #padding the sentence to have exactly the same shape as `embedding_2` input
        X_pred = pad_sequences(X_pred, maxlen=60, dtype='int32', value=0)   #MAXLEN from before is 81/56 automize!
        sentiment = model.predict(X_pred, batch_size=100, verbose=2)[0]
        sentence_profile.append(sentiment)
        if(np.argmax(sentiment) == 1):
            # print('neutral')
            sentence_sent.append(np.argmax(sentiment))
        elif(np.argmax(sentiment) == 0):
            sentence_sent.append(np.argmax(sentiment))
            # print('negative')
        elif(np.argmax(sentiment) == 2):
            sentence_sent.append(np.argmax(sentiment))
            # print('positive')
    print(i)
    article_sent.append(sentence_sent)
    article_sent_profile.append(sentence_profile)


###Calculate sentiment for each article based on the prediction above and the notation introduced:
article_sent_Doc = []
article_sent_Doc_B = []

for article in range(len(article_list)):#len(article_list):
    count = 0
    Pos = 0
    Neg = 0
    for sentence in range(len(article_sent[article])):
        count += 1
        if(article_sent[article][sentence]==0):
            Neg += 1
        elif(article_sent[article][sentence]==2):
            Pos += 1
    print(Pos,Neg)
    PF = Pos/count
    NF = Neg/count
    B_a = np.log(1+PF) - np.log(1+NF)
    #Transform to [-1,1] scale for better interpretation
    B = np.log(2)*B_a
    article_sent_Doc.append(B_a)
    article_sent_Doc_B.append(B)
    # article_sent_full.insert(article, B)

Sent_list = zip(article_sent_Doc, article_sent_Doc_B)
Sent_df = pd.DataFrame(list(Sent_list), columns=['sent', 'log2Sent'])
Sent_df['sent'].sort_values()
Sent_df['sent'].mean()
Sent_df['sent'].quantile(q=[0.25, 0.5, 0.75])


# article_sent
# [[1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1],[1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1],...]

# article_sent_profile
# [[array([0.12835893, 0.84216994, 0.02947113], dtype=float32), array([0.02182116, 0.9281721 , 0.05000673],

article_sent_Doc = []
article_sent_Doc_B = []

for i, article in enumerate(article_list):
    count = 0
    Pos = 0
    Neg = 0
    sent = 0
    for j in range(len(article)):
        count += 1
        if(article_sent[i][j]==0):
            Neg += 1
        elif(article_sent[i][j]==2):
            Pos += 1
        sent+= (article_sent_profile[i][j][2]-article_sent_profile[i][j][0])
    # print(Pos,Neg)
    PF = Pos/count
    NF = Neg/count
    B_a = np.log(1+PF) - np.log(1+NF)
    #Transform to [-1,1] scale for better interpretation
    # B = np.log(2)*B_a
    sent = sent/count
    article_sent_Doc.append(B_a)
    article_sent_Doc_B.append(sent)
    # article_sent_full.insert(article, B)

print(article_sent_Doc)
print(article_sent_Doc_B)

DataInput = zip(articles, article_sent_Doc_B, article_sent_Doc)
df_sent = pd.DataFrame([articles, article_sent_Doc_B, article_sent_Doc], columns=['article', 'sent_class', 'sent_binary'])
df_sent = pd.DataFrame(list(DataInput), columns=['article', 'sent_class', 'sent_binary'])
df_sent.to_csv("sentiment.csv")

