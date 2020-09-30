# -*- coding: utf-8 -*-
"""

@author: Brandon Langley
bplangl
CPSC 4820
Project 3

"""
import math

def main():
    bodyLength=-1
    dorsalLength=-1
    
    #define weights as discovered in the report/project3.py 
    weights=[1.27705108,0.02545808,-0.2986095]

    bodyLength=float(input("Enter Body Length: "))
    dorsalLength = float(input("Enter Dorsal Fin Length: ") )
    data=[1,bodyLength, dorsalLength]
    #continue making predictions until user enters 0 for both dorsal and body length 
    while bodyLength!=0 or dorsalLength!=0:
        prediction=hypothesisFunction(data,weights)
        if prediction>=.5:
            prediction=1
        else:
            prediction=0
        print("TigerFish",prediction)
        bodyLength=float(input("Enter Body Length: "))
        dorsalLength = float(input("Enter Dorsal Fin Length value: "))
        data=[1,bodyLength, dorsalLength]


def hypothesisFunction(row, weight):
    x1=row[1]
    x2=row[2]
    y = weight[0] + weight[1] * x1 + weight[2] * x2 
    hyp=1/(1+math.e**-y)
   
    return hyp

if __name__== "__main__":
    main()