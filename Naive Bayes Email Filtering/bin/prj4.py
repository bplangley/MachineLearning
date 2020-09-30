# -*- coding: utf-8 -*-
"""

@author: Brandon Langley
bplangl
CPSC 4820
Project 4

"""

#import pandas as pd
import os.path



def main():
    spam =0
    ham=0
    counted=dict()    
    fName="GEASTrain.txt"
    fName=input("Please Enter the file name of the Training Data: ")
    while os.path.exists(fName) is not True:
        print("\n\nInvalid File Name")
        fName=input("Please Enter the file name of the Training Data: ")
    f=open(fName,"r")
    
    fstop="StopWords.txt"
    fstop=input("Please Enter the file name of the Stop Words: ")
    while os.path.exists(fstop) is not True:
        print("\n\nInvalid File Name")
        fstop=input("Please Enter the file name of the Stop Words: ")
    fs=open(fstop,"r")
    
    stopWords = fs.read().splitlines()
    #print(stopWords)
    line=f.readline()
    while line!="":
        isSpam=int(line[:1])
        if isSpam==1:
            spam=spam+1
        else :
            ham=ham+1
        line=cleanText(line[1:])
        words=line.split()
        words=set(words)
        counted=countedWords(words,isSpam,counted,stopWords)
        line =f.readline()
    
    pl=percentList(.1,counted,spam,ham)
    total=spam+ham
    
    
    testSpam=0
    testHam=0
    tp=0
    tn=0
    fp=0
    fn=0
    fTest="GEASTest.txt"
    fTest=input("Please Enter the file name of the Test Data: ")
    while os.path.exists(fTest) is not True:
        print("\n\nInvalid File Name")
        fTest=input("Please Enter the file name of the Test Data: ")
    ft=open(fTest,"r")
    sl=ft.readline()
    while sl !="":
        isSpam=int(sl[:1])
        if isSpam==1:
           testSpam=testSpam+1
        else :
            testHam=testHam+1
        sl=cleanText(sl)
        sl=sl.split()
        sl=set(sl)
        spamProb=probability(sl,pl,1)
        #print("spam: ",spamProb)
        hamProb=probability(sl,pl,0)
        #print("ham:  ",hamProb)
        prediction=(spamProb*(spam/total))/(spamProb*(spam/total) + hamProb*(ham/total))
        #print("prediction",prediction,"\n\n")
        
        if prediction >= .5:
           prediction=1
        else:
            prediction=0
        if prediction==isSpam and prediction==1:
            tp+=1
        elif prediction==isSpam and prediction==0:
            tn+=1
        elif prediction!=isSpam and prediction==1:
            fp+=1
        elif prediction!=spam and prediction==0:
            fn+=1
            
        sl=ft.readline()
        
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    f1=2*((1)/((1/precision)+(1/recall)))
    
    print("\n\nSpam in test file: ",testSpam)
    print("Ham in test file:  ",testHam,"\n\n")
    print("True Positive: \t",tp)
    print("True Negative: \t",tn)
    print("False Positive:\t",fp)
    print("False Negative:\t",fn,"\n\n")
    print("Accuracy:   %.5f"%(accuracy))
    print("Precision:  %.5f"%(precision))
    print("Recall:     %.5f"%(recall))
    print("F1:         %.5f"%(f1))
    
    
    fs.close()
    f.close()

def probability(sl,pl,i):
    i=i
    prob=1
    for word in pl:
        if word in sl:
            prob=prob*pl[word][i]
        else:
            prob=prob*(1-pl[word][i])
        #print(prob)
    return prob
            
def cleanText(line):
    line=line.lower()
    line=line.strip()
    for letters in line:
        if letters in """[]!.,"-!_@;':#$%^&*()+/?""":
            line=line.replace(letters," ")
    return line


def countedWords(text, is_spam, counted, stopWords):
    for word in text: 
        if word in counted and word not in stopWords: 
            if is_spam==1:
                counted[word][1]=counted[word][1]+1
            else:
                counted[word][0]=counted[word][0]+1
        elif word not in stopWords:
            if is_spam==1:
                counted[word]=[0,1]
            else:
                counted[word]=[1,0]
    return counted


def percentList(k, theCount,spams,hams):
    for key in theCount:
        theCount[key][0]=(theCount[key][0] + k)/(2 * k + hams)
        theCount[key][1]=(theCount[key][1] + k)/(2 * k + spams)
    return theCount

    



if __name__== "__main__":
    main()
