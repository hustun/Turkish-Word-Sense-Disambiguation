# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:02:58 2020

@author: hasbe
"""

removeList = []

corpus_dir = "E:/CSE/Thesis/Thesis/Code/corpus/"

removeFile = open("remove", "r", encoding="utf-8")
mixedFile = open(corpus_dir + 'posts12-mixed-all.txt', "r", encoding="utf-8")
mixedRemovedFile = open(corpus_dir + 'posts12-mixed-all-removed.txt', "w", encoding="utf-8")

for line in removeFile:
    line = line.strip()
    removeList.append(line)
    
removeFile.close()

for line in mixedFile:
    words = line.split()
    dif = [word for word in words if word not in removeList]
    newLine = " ".join(dif)
    mixedRemovedFile.write(newLine + "\n")
