import sys
import numpy

colNum = int(sys.argv[1])
infile = sys.argv[2]
outfile = "ans1.txt"
table = []

# File input
data = numpy.loadtxt(infile)

# Extract specific column
col = data[:, colNum]

# Sort the column
sortCol = numpy.sort(col)

# File output
with open(outfile, 'w') as outputFile:
	outputFile.write(','.join(map(str, sortCol)))