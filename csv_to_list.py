import csv

with open('x1.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

newData1 = []
newData2 = []
for i in range(len(data)):
    newData1.append(float(data[i][0]))

print(newData1)