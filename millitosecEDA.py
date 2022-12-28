#EDA_ST unit was millisecond changes to second gets the average
import pandas as pd
import csv

rows = []
with open("gaoE.csv", 'r', encoding="utf-8_sig") as file:
    reader = csv.reader(file)
    for row in reader:
        rows.append(row)

data=[]

num=0
sST=0
sEDA=0
key=rows[0][0]
for i in range(len(rows)):
    cEDA=float(rows[i][1])
    cST=float(rows[i][2])
    if key!=rows[i][0]:
        EDA=sEDA/num
        ST=sST/num
        data.append([key,EDA,ST])
        num=0
        sEDA=0
        sST=0
        key=rows[i][0]
    if key==rows[i][0]:
        num+=1
        sST+=cST
        sEDA+=cEDA

EDA=sEDA/(num)
ST=sST/(num)
data.append([key,EDA,ST])

with open('gaoE.csv', 'w', newline='') as file:
    writer = csv.writer(file) 
    writer.writerows(data) 

