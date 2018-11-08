#!/usr/bin/env python3

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

fileFolderAndName = '20181103/20181103_114940_497_Auton.txt'

try:
    firstArg = sys.argv[1]
    fileFolderAndName = firstArg
except:
    print('No input given, using default file...')

with open('../data/' + fileFolderAndName, newline='') as csvfile:
    tsvreader = csv.reader(csvfile, delimiter='\t', quotechar='|')

    columnHeaderList = next(tsvreader)
    colCount = len(columnHeaderList)
    rowCount = sum(1 for row in tsvreader)
    print('\n\nNumber of Columns: %i  Number of Rows: %i\n\n' % (colCount, rowCount))
    for i in range(0, colCount):
        print('Column %i: %s' % (i, columnHeaderList[i]))

with open('../data/' + fileFolderAndName, newline='') as csvfile:
    tsvreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
    dataMatrix = np.zeros((rowCount, colCount))
    for i in range(1, rowCount):
        row = next(tsvreader)
        for j in range(0,colCount):
            try:
                dataMatrix[i,j] = float(row[j])
            except:
                dataMatrix[i,j] = None

def  plotPairOfCoordsByInput():
    xVar = int(input("\nEnter number of Column for horizontal axis: "))
    yVar = int(input("Enter number of Column for vertical axis: "))
    plotPairOfColumns(xVar, yVar)

def plotVarVnumCycles(colNum):
    fig = plt.figure()
    plt.xlabel("Number of Cycles")
    plt.ylabel(columnHeaderList[colNum])
    plt.plot(range(0, len(rowCount)), dataMatrix[:,ycolNum])
    # plt.savefig(('../renders/' + columnHeaderList[xVar] + ' -VS- ' + columnHeaderList[yVar] +'.png'))
    plt.show()

def plotPairOfColumns(xCol, yCol):
    fig = plt.figure()
    plt.xlabel(columnHeaderList[xCol])
    plt.ylabel(columnHeaderList[yCol])
    plt.plot(dataMatrix[:,xCol], dataMatrix[:,yCol])
    # plt.savefig(('../renders/' + columnHeaderList[xVar] + ' -VS- ' + columnHeaderList[yVar] +'.png'))
    plt.show()


def plotKallmanXY():
    plotPairOfColumns(9,10)

def plotOldMeasuredXY():
    plotPairOfColumns(4,5)

def plotKallmanTheta():
    plotVarVnumCycles(11)

def plotOldMeasuredTheta():
    plotVarVnumCycles(6)

def plotXVariance():
    plotVarVnumCycles(14)

def plotYVariance():
    plotVarVnumCycles(20)

def plotThetaVariance():
    plotVarVnumCycles(26)

def fullFilterAnalysisPlot():
    plotKallmanXY()
    plotOldMeasuredXY()
    plotKallmanTheta()
    plotOldMeasuredTheta()
    plotXVariance()
    plotYVariance()
    plotThetaVariance()


fullFilterAnalysisPlot()







