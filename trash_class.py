import pandas as pd
import csv

rows = []
stress=''
time=''
data=[]

with open(".csv", 'r', encoding="utf-8_sig") as file:
    reader = csv.reader(file)
    for row in reader:
        rows.append(row)

for i in range(len(rows)-1):
    if (float(rows[i+1][1])-float(rows[i][1]))>0:
        data.append([rows[i][0],'1'])
    else:
        data.append([rows[i][0],'0'])

with open('mori_classed.csv', 'w', newline='') as file:
    writer = csv.writer(file) 
    writer.writerows(data) 