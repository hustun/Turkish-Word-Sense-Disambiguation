# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:51:25 2020

@author: hasbe
"""


from gensim.test.utils import datapath
from gensim import utils
from gensim.models.word2vec import LineSentence
import multiprocessing

from gensim.models import Word2Vec

import gensim.models

model = Word2Vec(LineSentence("E:/CSE/Thesis/Thesis/Code/corpus/stemmed-12-alternative.txt"), size=400, window=5, min_count=5, workers=multiprocessing.cpu_count())
    
from gensim.models import Word2Vec, KeyedVectors   
model.wv.save_word2vec_format('models/model-stemmed-12-alternative', binary=True)