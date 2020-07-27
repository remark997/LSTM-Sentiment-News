import json
from MongoDbProvider import DatabaseProvider
import datetime
import pandas as pd
import numpy as np
from numpy import mean
import statistics
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




mdb = DatabaseProvider()
queryList = mdb.ReadFromQueryParamsFile()

#Load one of the trained models
max_fatures = 5500
tokenizer = Tokenizer(num_words=max_fatures, split=' ',filters='!"#&()*+,-./:;<=>?@[\]^_`{"}~\t\n',lower = True)
filepath_a ='Naive.hdf5'
model = load_model(filepath_a,custom_objects={"precision": precision, "recall":recall})

#result = mdb.ReadNewsFromDatabase('3M Co', datetime.datetime(1990,3,1), datetime.datetime(2000,1,1))

def LSTM_Pred(Data):
    article_list = clean_text(Data)
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
        # print(i)
        if sentence_sent != []:
            article_sent.append(sentence_sent)
            article_sent_profile.append(sentence_profile)
    article_sent_Doc = []
    article_sent_Doc_B = []
    for article in range(len(article_sent)):  # len(article_sent):
        count = 0
        Pos = 0
        Neg = 0
        for sentence in range(len(article_sent[article])):
            count += 1
            if (article_sent[article][sentence] == 0):
                Neg += 1
            elif (article_sent[article][sentence] == 2):
                Pos += 1
        # print(Pos, Neg)
        PF = Pos / count
        NF = Neg / count
        B_a1 = np.log(1 + PF) - np.log(1 + NF)
        # Transform to [-1,1] scale for better interpretation
        B1 = np.log(2) * B_a1
        article_sent_Doc.append(B_a1)
        article_sent_Doc_B.append(B1)
    B_a = mean(article_sent_Doc)
    B   = mean(article_sent_Doc_B)
    B_std = np.std(article_sent_Doc_B)
    ###  sentiment is measured by the difference between pos and neg
    article_sent_Doc_B = []
    for i, article in enumerate(article_sent):
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
        # B_a = np.log(1+PF) - np.log(1+NF)
         #Transform to [-1,1] scale for better interpretation
        # B = np.log(2)*B_a
        sent = sent/count
        # article_sent_Doc.append(B_a)
        article_sent_Doc_B.append(sent)
    B_PN = mean(article_sent_Doc_B)
    B_PN_std = np.std(article_sent_Doc_B)
    return([B,B_a,B_PN, B_std,B_PN_std])



list_B_a = []
list_B = []
list_B_PN = []
list_B_std = []
list_B_PN_std = []
FnameList = []
DateList = []
List_of_query = []
i=1

for queryIndex in range(len(queryList)):
    print(queryList[queryIndex]["companyName"], datetime.datetime.strptime(queryList[queryIndex]["sDt"], "%Y-%m-%dT00:00:00"), datetime.datetime.strptime(queryList[queryIndex]["eDt"],"%Y-%m-%dT00:00:00"))
    Data = mdb.ReadNewsFromDatabase(queryList[queryIndex]["companyName"], datetime.datetime.strptime(queryList[queryIndex]["sDt"], "%Y-%m-%dT%H:%M:%S"), datetime.datetime.strptime(queryList[queryIndex]["eDt"],"%Y-%m-%dT%H:%M:%S"))
    if Data != []:
        # print(queryIndex)
        rst = LSTM_Pred(Data)
        list_B.append(rst[0])
        list_B_a.append(rst[1])
        list_B_PN.append(rst[2])
        list_B_std.append(rst[3])
        list_B_PN_std.append(rst[4])
        DateList.append(datetime.datetime.strptime(queryList[queryIndex]["eDt"],"%Y-%m-%dT%H:%M:%S"))
        FnameList.append(queryList[queryIndex]["companyName"])
        List_of_query.append(queryIndex)
        print(i)
        i += 1




queryIndex = 7
Data = mdb.ReadNewsFromDatabase(queryList[queryIndex]["companyName"],
                                  datetime.datetime.strptime(queryList[queryIndex]["sDt"], "%Y-%m-%dT%H:%M:%S"),
                                  datetime.datetime.strptime(queryList[queryIndex]["eDt"], "%Y-%m-%dT%H:%M:%S"))


