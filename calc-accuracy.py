# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 15:41:43 2020

@author: hasbe
"""

word = "ağ"
test_dir = "E:/CSE/Thesis/Thesis/Code/test/"
predFile = open(test_dir + word + '-pred.txt', "r", encoding="utf-8")

y = []
y_pred = []
count = 1
for line in predFile:
    line = line.strip()
    y_pred.append(int(line))
for i in range(100):
    if i < 50:
        y.append(2)
    else:
        y.append(1)

true = 0
false = 0
for i in range(len(y_pred)):
    if y_pred[i] == y[i]:
        true+=1
    else:
        false+=1
print("--------------")
print("Correct sense prediction: " + str(true))
print("Wrong sense prediction: " + str(false))
accuracy = true / (true + false)
print("Accuracy: " + str(accuracy * 100) + "%")
print("--------------")

predFile.close()