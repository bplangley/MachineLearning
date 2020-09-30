w# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 18:47:12 2019

@author: Brandon

Code to manipulte the input and calculate the weights   
"""

import math
import operator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os.path

def main():

    fileName="FF34[123490].txt"
    while os.path.exists(fileName) is not True:
        print("\n\nInvalid File Name")
        fileName=input("Please Enter the file name of the Training Data: ")
    f=open(fileName,"r")

    #open data file and verify succesful import
    size=int(f.readline())
    data=pd.read_csv(f,sep='\t', names=['Study Time', 'oz of Beer', 'GPA'])
    ######################################################################################################################################
    ##turn Randimization back on
    ######################################################################################################################################
    data=data.reindex(np.random.permutation(data.index))
    data=data.to_numpy()
    if(data.shape[0]!=size):
        print("read error")

    dfs = np.split(data, 10)
    testSet=np.concatenate([dfs[0],dfs[1], dfs[2]])
    #frames =
    trainingSet = np.concatenate([dfs[3], dfs[4], dfs[5],dfs[6],dfs[7], dfs[8],dfs[9]])
    #print(trainingSet)
    print(len(trainingSet))

    #print(x1)
    weights=np.zeros(3)
    #weights=[.5,.5,.5]
    #weights=[ 0.08929862, 0.04041465, -0.31615312]
    #weights=[ 0.64652193,  0.03680625, -0.32296804]
    weights=[1.27705108,0.02545808,-0.2986095]
    #alpha=0.00000001
    alpha=0.000001 
    
    iterations=70000
    x0=np.ones([len(trainingSet),1])
    #print(x0)
    x=trainingSet[:,0:2]
    #x=np.array(x)
    y=np.vstack(trainingSet[:,2])
    #y=data[:,2]
    #print(x)

    #print(y)
    x=np.hstack((x0,x))
    #print(x)
    #print(y)

   # for row in trainingSet:
    #    yhat=hypothesisFunction(row,weights)

        #print("Expected=%.3f, Predicted=%.3f" % (row[2], yhat))

    #regression(alpha,iterations)
    #print(cost2(x,y,weights))
    #gradientDescent(x,y,weights,alpha,iterations)
    #gradient_descent(x,y,weights,alpha,iterations)
     
    
        
    x0=np.ones([len(testSet),1])
    #print(x0)
    x=testSet[:,0:2]
    #x=np.array(x)
    y=np.vstack(testSet[:,2])
    #y=data[:,2]
    #print(x)

#    print(y)
    x=np.hstack((x0,x))
    print("\n\n\nTESTSET\n\n\n")
    T0=0
    F0=0
    T1=0
    F1=0
    yPredicted=[]
    for t in range(len(x)):
       yPredicted.append(hypothesisFunction(x[t],weights))
       error=yPredicted-y[t]
       #print("predicted: %.3f"% (yPredicted[t]),".............actual: %.3f"%y[t],"..........error: %.3f...."%error,x[t])
    
    for t in range(len(x)):
        #yPredicted=hypothesisFunction(x[t],weights)
        #error=yPredicted-y[t]
        if(yPredicted[t]>=.5):
            prediction=1
        else:
            prediction=0
            #print("predicted: %.3f"% (yPredicted),".............actual: %.3f"%y[t],"..........error: %.3f...."%error,x[t])
            #if yPredicted>=.5:
            #   print("predicted: 1",".............actual: %.3f"%y[t],"..........error: %.3f...."%error,x[t])
            #else:
            #   print("predicted: 0",".............actual: %.3f"%y[t],"..........error: %.3f...."%error,x[t])
        if prediction==y[t] and prediction==0:
             T0=T0+1
             #print("T0: ",T0)
        elif prediction==y[t] and prediction==1:
             T1=T1+1
             #print("T1: ",T1)
        elif prediction!=y[t] and prediction==0:
             F0=F0+1
             #print("F0: ",F0)
        elif prediction!=y[t] and prediction==1:
             F1=F1+1
             #print("F1: ",F1)
        
    acuraccy=((T0+T1)/(T1+T0+F1+F0))
    print("acuraccy ",acuraccy)
    print("j ",cost_new(yPredicted, y))
    print("T0: ",T0)
    print("T1: ",T1)
    print("F0: ",F0)
    print("F1: ",F1)
    #print (i,": ",costArray[i],".........acuraccy", acuraccy)
   # print("cost on testset",cost2(x,y,weights))


def hypothesisFunction(row, weight):
    #print(row)
    x1=row[1]
    x2=row[2]
    y = weight[0] + weight[1] * x1 + weight[2] * x2 #+ weight[3] * x1**2 + weight[4] * x2**2
    
    hyp=1/(1+math.e**-y)
    #print(y)
    if hyp>=.5:
        prediction=1
    else:   
        prediction=0
    #print(np.dot(row,weight))
    #print(hyp)
    return hyp

#def calcWeight():
    
def cost_new(prediction, y):
    #print(prediction)
    
    #j=-y*math.log(prediction)-(1-y)*math.log(1-prediction)
    j=0
    for i in range(len(y)):
        if prediction[i]!=1.0 and prediction[i]!=0.0:
            #print(prediction[i])
            j+=-y[i]*math.log(prediction[i])-(1-y[i])*math.log(1-prediction[i])
        #print(j)
    j=j/len(y)
    
    return j
    


def gradientDescent(x,y,weights,alpha,iterations):
    temp=np.zeros((3,1))
    plt.figure(figsize=(10,10))
    costArray=[]
    

    for i in range(iterations):
        yPrediction=[]
        temp0=0
        temp1=0
        temp2=0
        temp3=0
        temp4=0
        j=0
        #print("*************************************", i, weights)
        for m in range(len(x)):
            
            hypo=hypothesisFunction(x[m],weights)
            yPrediction.append(hypo)
            #j+=cost_new(hypo,y[m])
            temp0+=(hypo-y[m])*x[m][0]
            temp1+=(hypo-y[m])*x[m][1]
            temp2+=(hypo-y[m])*x[m][2]
            #temp3+=(hypo-y[m])*x[m][1]**2
            #temp4+=(hypo-y[m])*x[m][2]**2
        #costArray.append(j/len(x))
        costArray.append(cost_new(yPrediction,y))

        
        weights[0]=weights[0]-alpha*temp0
        weights[1]=weights[1]-alpha*temp1
        weights[2]=weights[2]-alpha*temp2
        #weights[3]=weights[3]-alpha*temp1
        #weights[4]=weights[4]-alpha*temp2
           
        
       
        if i%1000==0:
          #  print(i,": ",costArray[i])
            T0=0
            F0=0
            T1=0
            F1=0
            for t in range(len(x)):
                yPredicted=hypothesisFunction(x[t],weights)
                error=yPredicted-y[t]
                if(yPredicted>=.5):
                    prediction=1
                else:
                    prediction=0
                #print("predicted: %.3f"% (yPredicted),".............actual: %.3f"%y[t],"..........error: %.3f...."%error,x[t])
                #if yPredicted>=.5:
                 #   print("predicted: 1",".............actual: %.3f"%y[t],"..........error: %.3f...."%error,x[t])
                #else:
                #   print("predicted: 0",".............actual: %.3f"%y[t],"..........error: %.3f...."%error,x[t])
                if prediction==y[t] and prediction==0:
                    T0=T0+1
                    #print("T0: ",T0)
                elif prediction==y[t] and prediction==1:
                    T1=T1+1
                    #print("T1: ",T1)
                elif prediction!=y[t] and prediction==0:
                    F0=F0+1
                    #print("F0: ",F0)
                elif prediction!=y[t] and prediction==1:
                    F1=F1+1
                    #print("F1: ",F1)
                
            acuraccy=((T0+T1)/(T1+T0+F1+F0))
            print (i,": ",costArray[i],".........acuraccy", acuraccy)

    print("\n\n\nfinal")
   # print(c3)        
    print(weights)

    
    plt.xlabel('Iterations')
    plt.ylabel('J')
    plt.title('Cost J vs Number of Iterations')
    plt.xlim=(0, iterations)
    plt.ylim=(.5, .8)
    #print(costArray)
    plt.scatter(range(iterations),costArray,color='red',marker='v')
    plt.show()
    plt.figure(figsize=(10,10))
    
    plt.xlabel('Iterations')
    plt.ylabel('J')
    plt.title('Cost J vs Number of Iterations')
    plt.scatter(range(int(iterations*.97),iterations),costArray[int(iterations*.97)::],color='red',marker='v')

    
   
"""
"""

if __name__== "__main__":
    main()
