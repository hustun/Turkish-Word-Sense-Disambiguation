# -*- coding: utf-8 -*-
"""
Created on Sat May 30 03:06:58 2020

@author: hasbe
"""


import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import gensim
from gensim import utils

def tokenize_tr(content,token_min_len=2,token_max_len=50,lower=True):
    if lower:
        lowerMap = {ord(u'A'): u'a',ord(u'A'): u'a',ord(u'B'): u'b',ord(u'C'): u'c',ord(u'Ç'): u'ç',ord(u'D'): u'd',ord(u'E'): u'e',ord(u'F'): u'f',ord(u'G'): u'g',ord(u'Ğ'): u'ğ',ord(u'H'): u'h',ord(u'I'): u'ı',ord(u'İ'): u'i',ord(u'J'): u'j',ord(u'K'): u'k',ord(u'L'): u'l',ord(u'M'): u'm',ord(u'N'): u'n',ord(u'O'): u'o',ord(u'Ö'): u'ö',ord(u'P'): u'p',ord(u'R'): u'r',ord(u'S'): u's',ord(u'Ş'): u'ş',ord(u'T'): u't',ord(u'U'): u'u',ord(u'Ü'): u'ü',ord(u'V'): u'v',ord(u'Y'): u'y',ord(u'Z'): u'z'}
        content = content.translate(lowerMap)

    return [
	utils.to_unicode(token) for token in utils.tokenize(content, lower=False, errors='ignore')
	if token_min_len <= len(token) <= token_max_len and not token.startswith('_')
	]

# model_dir = "E:/CSE/Thesis/Thesis/Code/models/"
# model_source = model_dir + "model-mixed" #change model here
# model = gensim.models.KeyedVectors.load_word2vec_format(model_source, binary=True)

# words = []

# for word in model.vocab.keys():
#     words.append(word)

wordsFile = open('unique-words.txt', "r", encoding="utf-8")
wikipediaFile = open('wiki-corpus-40k.txt', "a", encoding="utf-8")


# for word in words:
#     wordsFile.write(word + "\n")
    
# wordsFile.close()
    
import requests
from bs4 import BeautifulSoup

c=0
for line in wordsFile:
    c+=1
    if c <= 85836:
        continue
    print(c)
    word = line.strip()
    URL = "https://tr.wikipedia.org/wiki/" + word
    
    r = requests.get(url = URL) 
      
    response = r.text  
    
    soup = BeautifulSoup(response, 'html.parser')
    
    article = soup.find_all("div", class_="mw-parser-output")
    
    article = BeautifulSoup(str(article), 'html.parser')
    
    art = ""
    for text in article.find_all('p'):
        art = art + text.text
        # print(text.text)
        # print(tokenize_tr(text.text))
    # print(art)
    if art == "" or art == " ":
        continue
    tokenized_article = tokenize_tr(art)
    string_article = " ".join(tokenized_article)
    wikipediaFile.write(string_article + "\n")