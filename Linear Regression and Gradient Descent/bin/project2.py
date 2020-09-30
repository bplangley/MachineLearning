# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 18:47:12 2019

@author: Brandon
"""

import math
import operator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os.path

def main():

    fileName="GPAData.txt"
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
    #weights=[ 1.95040464e-05, -1.93726369e-02,  2.73014849e-03] #15,000
    #weights=[ 1.90433173e-05, -3.59803153e-02,  2.60388725e-03]
    #weights=[2.19845811e-05, 5.20852296e-02, 2.59243661e-03]
    #weights=[-0.0400411, 0.00360268, -0.00239843]
    #weights=[-0.15236346, 0.00341829, -0.00135547]
    #weights=[-0.17287292, 0.00354258, -0.00151283]

    alpha=0.00001
    iterations=50000
    x0=np.ones([len(trainingSet),1])
    #print(x0)
    x=trainingSet[:,0:2]
    #x=np.array(x)
    y=np.vstack(trainingSet[:,2])
    #y=data[:,2]
    #print(x)

#    print(y)
    x=np.hstack((x0,x))
    #print(x)
    #print(y)

   # for row in trainingSet:
    #    yhat=hypothesisFunction(row,weights)

        #print("Expected=%.3f, Predicted=%.3f" % (row[2], yhat))

    regression(alpha,iterations)
    print(cost2(x,y,weights))
    #gradientDescent(x,y,weights,alpha,iterations)
    """  
     for t in range(len(x)):
        yPredicted=hypothesisFunction(x[t],weights)
        error=yPredicted-y[t]
        print("predicted: %.3f"% (yPredicted),".............actual: %.3f"%y[t],"..........error: %.3f...."%error,x[t])
        """  
    x0=np.ones([len(testSet),1])
    #print(x0)
    x=testSet[:,0:2]
    #x=np.array(x)
    y=np.vstack(testSet[:,2])
    #y=data[:,2]
    #print(x)

#    print(y)
    x=np.hstack((x0,x))
    print("cost on testset",cost2(x,y,weights))


def hypothesisFunction(row, weight):
    #print(row)
    x1=row[1]
    x2=row[2]
    y = weight[0] + weight[1] * x1 + weight[2] * x2
    
    hyp=1/(1+math.e**y)
    #print(y)
    #print(np.dot(row,weight))
    return hyp

#def calcWeight():

def cost2(x,y,weights):
    sumError=0
    for i in range(len(x)):
        yp=hypothesisFunction(x[i],weights)
        error=yp-y[i]
        sumError+=error
    sumError=sumError**2
    cost=sumError/(2*len(x))
    return cost 

def cost(prediction, y):
    #print(x,y)

    """
    errorSum=0
    h_w=[]
    x0=0
    x1=0
    x2=0
    #return np.sum(np.power(((x @ weight.T) - y), 2)) / (2 * len(x))"""
    for i in range(len(x)):
        yp=hypothesisFunction(x[i],weight)
        error=yp-y[i]
        error=error**2
        errorSum+=error
        x0+=weight.T @ x[i]*x[0]
        x1+=weight.T @ x[i]*x[1]
        x2+=weight.T @ x[i]*x[2]
        #print(h_w)


    h_w.append([x0,x1,x2])
    #j=errorSum/(2*len(x))
    return h_w


def regression(alpha, iterations):
    weights=np.zeros([3,1])

#def gradientHelper():
    
def gradientDescent(x,y,weights,alpha,iterations):
    temp=np.zeros((3,1))
    plt.figure(figsize=(10,10))
    costArray=[]

    for i in range(iterations):
        for w in range(len(weights)):
            sumError=0
            for m in range(len(x)):
                error=(hypothesisFunction(x[m],weights)-y[m])*x[m,w]
               # print(error, weights )
                weights[w]=weights[w]-error*alpha/(len(x))
                
                
            
        c2=cost2(x,y,weights)   
        costArray.append(c2)        
        #print(c2)
        if i%500==0:
            print(i,": ",c2)
    print("\n\n\nfinal")
    print(c2)        
    print(weights)

    
    plt.xlabel('Iterations')
    plt.ylabel('J')
    plt.title('Cost J vs Number of Iterations')
    plt.scatter(range(iterations),costArray,color='red',marker='v')
    plt.show()
    plt.figure(figsize=(10,10))
    plt.xlabel('Iterations')
    plt.ylabel('J')
    plt.title('Cost J vs Number of Iterations')
    #for p in range(int(iterations*.75),iterations):
    plt.scatter(range(int(iterations*.97),iterations),costArray[int(iterations*.97)::],color='red',marker='v')

    """ for p in range(iterations):
        if p>100:
            if p%100==0:
                plt.scatter(p,costArray[p],color='red',marker='v')
        else:
            plt.scatter(p,costArray[p],color='red',marker='v')
    plt.show()
    plt.figure(figsize=(10,10))
    plt.xlabel('Iterations')
    plt.ylabel('J')
    plt.title('Cost J vs Number of Iterations')
    for p in range(int(iterations*.75),iterations):
        plt.scatter(p,costArray[p],color='red',marker='v')
        """
    """
        
            
            sumError=sumError**2/(2*len(x))    
            temp[w]=alpha/(len(x))*sumError
            """
            #print(sumError)
            #print(cost2(x,y,weights))
            #weights=temp
      
    """for t in range(len(x)):
        yPredicted=hypothesisFunction(x[t],weights)
        error=yPredicted-y[t]
        print("predicted: %.3f"% (yPredicted),".............actual: %.3f"%y[t],"..........error: %.3f...."%error,x[t])
                
   """
        
        
    


def gradientDescent1(x, y, weights, alpha, iterations):
    print (weights)
    #print(x)
    #print(y)
    plt.figure(figsize=(20,20))
    costt=[]

    for i in range(0,iterations):
        j=0
        ypredicted=[]
        sumError=0
        #h_w=weights.T @ x
        """h=cost(x,y,weights)
        weights = weights - (alpha/len(x)) * np.sum((x @ weights.T - y) * x, axis=0)
        print(h)

        print(x.shape)
        print(y.shape)

        print(weights.shape)
        #weights=weights - (alpha/len(x)) * np.sum((x @ weights - y) * x, axis=0)
        j=cost(x,y,weights)
        weights[0]=weights[0]-(1/len(x))*alpha*j
        weights[1]=weights[1]-(1/len(x))*alpha*j*x[1]
        weights[2]=weights[2]-(1/len(x))*alpha*j*x[2]
        #print(j)
        """

        for row in x:
            yPredicted=hypothesisFunction(row,weights)
            error=yPredicted-y[j]
            sumError+=error**2

            #if (i==iterations-1):
               #print("predicted: %.3f"% (yPredicted),".............actual: %.3f"%y[j],"..........error: %.3f...."%error,row)
            #print(weights)
            weights[0]=weights[0]-(1/len(x))*alpha*error
            weights[1]=weights[1]-(1/len(x))*alpha*error*x[:1].mean()
            weights[2]=weights[2]-(1/len(x))*alpha*error*x[:,2].mean()
            j+=1



        costt.append(sumError)
        #print(sumError,".....",cost(x,y,weights))
        print(sumError)
        if(i>000 and i%50==0):
            plt.scatter(i,sumError,color='red',marker='v')


            #temp0=weights[0]-(alpha/len(x))*np.sum(hypothesisFunction(row,weights)-y[row])
            #temp1=weights[1]-(alpha/len(x))*np.sum(hypothesisFunction(row,weights)-y[row])
    #print(error)
    #print (cost)
    #for m in range(0,iterations):
        #print(m)
     #   if m>000:
      #      plt.scatter(m,cost[m],color='red',marker='v')
    print(weights)

    #costs=[]


       # weights=weights-(alpha/len(x))*np.sum((x @ weights.T-y)*x, axis=0)
        #temp=cost(x, y, weights)

    #return temp;


if __name__== "__main__":
    main()
