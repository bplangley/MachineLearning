# -*- coding: utf-8 -*-
"""

@author: Brandon Langley
bplangl
CPSC 4820
Project 1

"""


def main():

    #define weights as discovered in the report
    weights=[-0.17287292, 0.00354258, -0.00151283]

    studyTime=float(input("Please enter time spent studying pre week in minutes: "))
    ozBeer = float(input("Please enter the volume of beer consumed per week in ounces: ") )
    data =[1, studyTime, ozBeer]

    while studyTime!=0 or ozBeer!=0:
        gpa=hypothesisFunction(data, weights)
        print("Predicted GPA: %.2f" %(gpa))
        studyTime=float(input("Please enter time spent studying pre week in minutes: "))
        ozBeer = float(input("Please enter the volume of beer drank per week in ounces: ") )

        data =[1, studyTime, ozBeer]


def hypothesisFunction(row, weight):
    x1=row[1]
    x2=row[2]
    y = weight[0] + weight[1] * x1 + weight[2] * x2
    return y


if __name__== "__main__":
    main()
