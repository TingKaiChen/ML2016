import sys

colNum = int(sys.argv[1])
infile = sys.argv[2]
outfile = "ans1.txt"
table = []

# File input
with open(infile, 'r') as data:
	rows = data.readlines()
	for row in rows:
		numlist = map(float, row.strip().split(' '))
		table.append(numlist)

# Extract specific column
col = []
for nums in table:
	col.append(nums[colNum])

# Sort the column
col.sort()

# File output
with open(outfile, 'w') as outputFile:
	outputFile.write(','.join(map(str, col)))