#!/usr/bin/env python3

import csv
import numpy as np
import matplotlib as plt

fileFolderAndName = '20181101/20181101_202450_540_Auton.tsv'

with open('../data/' + fileFolderAndName, newline='') as csvfile:
    tsvreader = csv.reader(csvfile, delimiter='\t', quotechar='|')

    columnHeaderList = next(tsvreader)
    colCount = len(columnHeaderList)
    rowCount = sum(1 for row in tsvreader)
    print('\n\nNumber of Columns: %i  Number of Rows: %i\n\n' % (colCount, rowCount))


