import pandas as pd
import csv

rows = []
gradEDA=False
s1=0
s2=0
s3=0
s4=0
score=0
data=[]

def write(data):
    with open('gaoE.csv', 'w', newline='') as file:
        writer = csv.writer(file) 
        writer.writerows(data) 

def rule1(i):
    list=[2,3,4,5]
    for num in list:
        if (i+num)<len(rows):
            if (float(rows[i+num][1])-float(rows[i+num-1][1]))>0:
                return 1
    list=[6,7,8]
    for num in list:
        if (i+num)<len(rows):
            if (float(rows[i+num][1])-float(rows[i+num-1][1]))>0:
                return 0.5
    return 0

def rule2(i):
    if i==0:
        return 0
    list=[2,3,4,5,6]
    if (float(rows[i][1])-float(rows[i-1][1]))>0:
        for num in list:
            if float(rows[i][2])-float(rows[i-1][2])<0:
                return 0.5
            else:
                return 1
    else:
        return 0
        

with open("gaoE.csv", 'r', encoding="utf-8_sig") as file:
    reader = csv.reader(file)
    for row in reader:
        rows.append(row)

for i in range(len(rows)):

    s1=rule1(i)
    s2=rule2(i)
    sum=(s1+s2)/2
    if sum<=0.5:
        data.append([rows[i][0],'1'])
    else:
        data.append([rows[i][0],'0'])
    #print(rows[i][0],stress)

#data=[time,stress]

write(data)