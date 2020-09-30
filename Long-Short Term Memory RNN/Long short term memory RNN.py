# -*- coding: utf-8 -*-
"""

@author: Brandon Langley
bplangl
CPSC 4820
Project 4

"""

#import pandas as pd
import os.path
import keras
from keras.layers import Dense
from keras.layers.embeddings import Embedding
import numpy as np



def main():
    from keras.datasets import imdb
    numWords=15000
    wordPerReview=350
    (xTrain, yTrain), (xTest, yTest) = imdb.load_data(path="imdb.npz",num_words=numWords, skip_top=0, 
                                                      
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)
    '''
    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=numWords)
    data = np.concatenate((training_data, testing_data), axis=0)
    targets = np.concatenate((training_targets, testing_targets), axis=0)
    index = imdb.get_word_index()
    reverse_index = dict([(value, key) for (key, value) in index.items()]) 
    decoded = " ".join( [reverse_index.get(i - 3, "________") for i in data[545]] )
    print(data[545])

    print(decoded)
    '''
    
    
    #print (xTrain[545])
    #print (yTrain[545])


    xTrain=keras.preprocessing.sequence.pad_sequences(xTrain,maxlen=wordPerReview)
    xTest=keras.preprocessing.sequence.pad_sequences(xTest,maxlen=wordPerReview)

   # yTrain=keras.preprocessing.sequence.pad_sequences(yTrain,maxlen=wordPerReview)
   
    model=keras.models.Sequential()
    model.add(Embedding(numWords, 32, input_length=wordPerReview))
    model.add(keras.layers.LSTM(25,dropout=.1, recurrent_dropout=.1))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    results=model.fit(xTrain,yTrain,epochs=2)
    
    scores = model.evaluate(xTest, yTest, verbose=1)
    print (results)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    
    
    
if __name__== "__main__":
    main()
