#!/usr/bin/env python3

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

fileFolderAndName = '20181103/20181103_114940_497_Auton.txt'

# Check to see if an argument was given for the file to use
try:
    firstArg = sys.argv[1]
    fileFolderAndName = firstArg
except:
    print('No input given, using default file...')

# Open of the data file and get the number of columns and rows
with open('../data/' + fileFolderAndName, newline='') as csvfile:
    tsvreader = csv.reader(csvfile, delimiter='\t', quotechar='|')

    columnHeaderList = next(tsvreader)
    colCount = len(columnHeaderList)
    rowCount = sum(1 for row in tsvreader)
    print('\n\nNumber of Columns: %i  Number of Rows: %i\n\n' % (colCount, rowCount))
    for i in range(0, colCount):
        print('Column %i: %s' % (i, columnHeaderList[i]))

# Open the file again because csv.reader is an itterator so that the file
# doesn't go into memory. Then put all the data in a numpy array
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

numVars = int(input("\nEnter number of lines you want to plot: "))
xVar = int(input("\nEnter number of Column for horizontal axis: "))

# Loop to put the desired Vertical variables in an array
yVars = np.arange(numVars)
for i in range(0, yVars.size):
    yVars[i] = int(input(("\nEnter number of Column for vertical axis " + str(i) + ": ")))
fig = plt.figure()
plt.xlabel(columnHeaderList[xVar])

# Itterate through all the vertical variables to plot
for i in range(0, yVars.size):
    plt.plot(dataMatrix[:,xVar], dataMatrix[:,yVars[i]], label=columnHeaderList[yVars[i]])
plt.legend()

plt.savefig(('../renders/' + columnHeaderList[xVar] + ' -VS- ' + columnHeaderList[yVars[0]] +'.png'))
plt.show()