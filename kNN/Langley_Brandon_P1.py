# -*- coding: utf-8 -*-
"""

@author: Brandon Langley
bplangl
CPSC 4820 
Project 1

"""
import math
import operator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os.path

def main():
    #prompt user for the trianing data check for file valididty and then open and import the training data 
    fileName="FF34[123490].txt"
    fileName=input("Please Enter the file name of the Training Data: ")
    while os.path.exists(fileName) is not True:
        print("\n\nInvalid File Name")
        fileName=input("Please Enter the file name of the Training Data: ")
    f=open(fileName,"r")
    
    #open data file and verify succesful import
    size=int(f.readline())
    data=pd.read_csv(f,sep='\t', names=['Body Length', 'Dorsal Length', 'Type'])
    data=data.reindex(np.random.permutation(data.index))
    data=data.to_numpy()
    if(data.shape[0]!=size):
        print("read error")


   #define the ideal k valueto be 7     
    kVal=3
    #read body and dorsal length from user and make a prediction using kNN
    bodyLength=-1
    dorsalLength=-1
    bodyLength=float(input("Enter Body Length: "))
    dorsalLength = float(input("Enter Dorsal Fin Length value: ") )
    #continue making predictions until user enters 0 for both dorsal and body length 
    while bodyLength!=0 and dorsalLength!=0:
        test=kNN(data,[bodyLength,dorsalLength],kVal)
        prediction=classify(test)
        #print("Based on the ",kVal," nearest neighbors, this is most likely TigerFish", prediction)
        print("TigerFish",prediction)
        bodyLength=float(input("Enter Body Length: "))
        dorsalLength = float(input("Enter Dorsal Fin Length value: "))
                
    f.close()



#accept a list of nearest neighbors and based upon the known types from the trainig set 
#make a prediction about what the most likely type of the test subject is
def classify(test):
    tigerfish0=0
    tigerfish1=0
    for i in range(len(test)):
        if test[i][2]==0:
           tigerfish0=tigerfish0+1
        elif test[i][2]==1:
           tigerfish1=tigerfish1+1
    if tigerfish0>tigerfish1:
        return 0
    else:
        return 1

#take an array of training vlues in trainingset, 
#a subject value for which to find the nearest neighbors in the training Set 
# and k- the number of neighbors to return 
def kNN(trainingSet, value,k):
    testValue=value
    distances=[]
    for x in range(trainingSet.shape[0]):
        dist=eDist(testValue, [trainingSet[x,0],trainingSet[x,1]])
        distances.append(([trainingSet[x,0],trainingSet[x,1], trainingSet[x,2]], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors=[]
    for y in range(k):
        neighbors.append(distances[y][0])
    return neighbors

#calculate the distance between two 2-deminsional data nodes
def eDist(node,q):
    distance=math.sqrt((node[0]-q[0])**2+(node[1]-q[1])**2)
    return distance


if __name__== "__main__":
    main()
