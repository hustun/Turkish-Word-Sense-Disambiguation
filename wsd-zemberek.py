# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:57:17 2020

@author: hasbe
"""

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import gensim
from gensim import utils
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM, java
from typing import List

import math

resultsFile = open("results-file.txt", "w", encoding="utf-8")

def tokenize_tr(content,token_min_len=2,token_max_len=50,lower=True):
	if lower:
		lowerMap = {ord(u'A'): u'a',ord(u'A'): u'a',ord(u'B'): u'b',ord(u'C'): u'c',ord(u'Ç'): u'ç',ord(u'D'): u'd',ord(u'E'): u'e',ord(u'F'): u'f',ord(u'G'): u'g',ord(u'Ğ'): u'ğ',ord(u'H'): u'h',ord(u'I'): u'ı',ord(u'İ'): u'i',ord(u'J'): u'j',ord(u'K'): u'k',ord(u'L'): u'l',ord(u'M'): u'm',ord(u'N'): u'n',ord(u'O'): u'o',ord(u'Ö'): u'ö',ord(u'P'): u'p',ord(u'R'): u'r',ord(u'S'): u's',ord(u'Ş'): u'ş',ord(u'T'): u't',ord(u'U'): u'u',ord(u'Ü'): u'ü',ord(u'V'): u'v',ord(u'Y'): u'y',ord(u'Z'): u'z'}
		content = content.translate(lowerMap)
	return [
	utils.to_unicode(token) for token in utils.tokenize(content, lower=False, errors='ignore')
	if token_min_len <= len(token) <= token_max_len and not token.startswith('_')
	]

def get_mean_vector(word2vec_model, words):
    # remove out-of-vocabulary words
    words = [word for word in words if word in word2vec_model.wv]
    if len(words) >= 1:
        return np.mean(word2vec_model[words], axis=0)
    else:
        return np.zeros((1, word2vec_model.vector_size))
    
def get_mean_vector_tf_idf(word2vec_model, words, line, idf):
    # remove out-of-vocabulary words
    words = [word for word in words if word in word2vec_model.wv]
    if len(words) >= 1:
        vectors = []
        for word in words:
            wordVec = word2vec_model[word]
            tf_idf = calculate_tf_idf(word, line, idf)
            word_vec_tf_idf = np.dot(wordVec, tf_idf)
            vectors.append(word_vec_tf_idf)
        # print([vector[0] for vector in vectors])
        # print("---------")
        # print(np.mean(vectors, axis=0))
        return np.mean(vectors, axis=0)
    else:
        return np.zeros((1, word2vec_model.vector_size))

def calculate_tf_idf(word, line, idf):
    count = line.count(word)
    length = len(line)
    tf = count / length
    tf_idf = idf[word] * tf
    # print(word)
    # print(tf_idf)
    # print("----")
    return tf_idf
    
def contains_word(s, w):
    return f' {w} ' in f' {s} '

def remove_elements_from_list(the_list, elements):
   return [element for element in the_list if element not in elements]

common = []

def tsne_plot(model, perp, additionalWords):
    "Create TSNE model and plot it"
    labels = []
    tokens = []
    words = []
    words = ['yemek', "ye", 'iç', 'yiyecek', 'içecek', 'içki', 'haziran', 'temmuz', 'ağustos', 'eylül', 'ekim', 'kasım'
            , 'aralık', 'ocak', 'şubat', 'mart', 'nisan', 'mayıs', 'pazartesi', 'salı', 'çarşamba', 'perşembe', 'cuma'
            , 'saat', 'gün', 'dakika', 'ay', 'sene', 'yıl', 'yüz', 'çehre', 'surat', 'sima', 'burun', 'kulak', 'ağız'
            , 'dil', 'çene', 'göz', 'saç', 'alın', 'kafa', 'beyin', 'deniz', 'hava', 'su', 'gökyüzü', 'uçak', 'gemi'
            , 'iki', 'üç', 'dört', 'beş', 'on', 'yirmi', 'otuz', 'kırk', 'elli', 'bin', 'milyon', 'üniversite', 'kış', 'ilkbahar'
            , 'lise', 'okul', 'profesör', 'makale', 'gazete', 'televizyon', 'radyo', 'kadın', 'erkek', 'kral', 'kraliçe'
            , 'gezegen', 'uydu', 'dünya', 'mars', 'jüpiter', 'venüs', 'uzay', 'mevsim', 'güneş', 'astroloji', 'satürn'
            , 'fil', 'karınca', 'kedi', 'köpek', 'çikolata', 'şeker', 'zürafa', 'çilek', 'ananas', 'kiraz', "hafta"
            , 'vişne', 'türkiye', 'italya', 'ispanya', 'amerika', 'pizza', 'makarna', 'hamburger', 'kebap', 'lokum'
            , 'portakal', 'balık', 'kuş', 'balina', 'ilkokul', 'kitap', 'futbol', 'basketbol', 'golf', 'sonbahar', 'bahar'
            , 'hentbol', 'voleybol', 'cümle', 'kelime', 'yazar', 'yapım', 'yönetmen', 'doktor', 'avukat', 'mühendis'
            , 'veteriner', 'insan', 'hayvan', 'bitki', 'biyoloji', 'hücre', 'öğretmen', 'öğrenci', 'aslan', 'kaplan'
            , 'fransa', 'ingiltere', 'türkçe', 'ingilizce', 'para', 'ücret', 'fiyat', 'değer', 'zam', 'enflasyon', 'yaz'
            , 'merkür', 'uranüs', 'neptün', 'plüton', 'yörünge', 'tutul', 'tut', 'plüton', 'medcezir', 'krater', 'kulaç', 'yürü'
            , 'okyanus', 'havuz', 'lunapark', 'otel', 'plaj', 'kum', 'sahil', 'antalya', 'muğla', 'çeşme'
            , 'market', 'müşteri', 'ürün', 'sat', 'alışveriş', 'ticari', 'semt', 'eczane', 'tezgah', 'satıcı']
    
    for word in words:
        tokens.append(model[word])
        labels.append(word)
    for word in additionalWords.keys():
        tokens.append(additionalWords[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=perp, n_components=2, init="pca", n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(24, 24)) 
    for i in range(len(x)):
        if labels[i].startswith('center') and len(labels[i]) > 6:
            plt.scatter(x[i],y[i], s = 100, c = 'gray')
        elif labels[i].startswith(word) and len(labels[i]) > len(word):
            plt.scatter(x[i],y[i], s = 200, c = 'red')
        elif labels[i].startswith('ccenter') and len(labels[i]) > 7:
            plt.scatter(x[i],y[i], s = 500, c = 'blue')
        elif labels[i].startswith('sense') and len(labels[i]) > 5:
            plt.scatter(x[i],y[i], s = 300, c = 'green')
        else:
            plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

word = "pazar" #change word here
wordCount = 0
lineCount = 0
frequency = 0
windowSize = 3
clusterNumber = 2
wordVectors = {}
additionalWords = {}
corpus_dir = "D:/Engineering Project/Thesis/Thesis/Code/corpus/"
linesFile = open(f'{word}.txt', "w", encoding="utf-8")
# zemberekFile = open(f'{word}-zemberek.txt', "r", encoding="utf-8")
# wikiFile = open("E:/CSE/Thesis/2020feb/Thesis/trwiki-20200101-pages-articles.xml/trwiki-20200101-pages-articles.xml", "r", encoding="utf-8")
stemmedFile = open(corpus_dir + 'stemmed.txt', "r", encoding="utf-8")
stemmed12File = open(corpus_dir + 'stemmed-12.txt', "r", encoding="utf-8")
mixedAllRemovedFile = open(corpus_dir + 'posts3-mixed-all-removed.txt', "r", encoding="utf-8")
corpusFile = open(corpus_dir + 'posts12-mixed-all-removed.txt', "r", encoding="utf-8")
workingFile = stemmedFile #change corpus here

model_dir = "D:/Engineering Project/Thesis/Thesis/Code/models/"
model_source = model_dir + "model-stemmed" #change model here
model = gensim.models.KeyedVectors.load_word2vec_format(model_source, binary=True)
word_vectors = model.wv

df = {}
idf_dic = {}

for key in model.vocab.keys():
    df[key] = 0

lines = 0
for line in workingFile:
    seen = set()
    lines+=1
    words = line.split()
    for targetWord in words:
        if targetWord in df and targetWord not in seen:
            df[targetWord]+=1
        seen.add(targetWord)
        # while targetWord in words:
        #     words.remove(targetWord)
print(df[word])
for key in df.keys():
    idf = lines / (df[key] + 1)
    log_idf = math.log10(idf)
    idf_dic[key] = log_idf

newLines = []
workingFile.seek(0)
for line in workingFile:
    lineCount+=1
    line = line.strip()
    if line == "" or line == " ":
        continue
    words = line.split()
    if word in words:
        frequency+=1
        
    while word in words:
        wordCount+=1
        currentWord = word + str(wordCount)
        firstIndex = words.index(word) - windowSize if words.index(word) - windowSize > 0 else 0
        lastIndex = words.index(word) + windowSize + 1 if words.index(word)  + windowSize < len(words) else len(words)
        
        
        contextWords = words[firstIndex:lastIndex]
        # contextWords.remove(word)
        wordVector = get_mean_vector(model, contextWords)
        # wordVector = get_mean_vector_tf_idf(model, contextWords, line.split(), idf_dic)
        
        wordVectors[currentWord] = wordVector
        words.remove(word)
        
print(lineCount)
X = list(wordVectors.values())
kmeans = KMeans(n_clusters=clusterNumber, n_init = 2)
pred_y = kmeans.fit_predict(X)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=clusterNumber)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
centers = kmeans.cluster_centers_
c = 1
print("--------")
for center in kmeans.cluster_centers_:
    print("Cluster " + str(c) + " related words:\n")
    a = model.similar_by_vector(center, topn=20)
    for res in a:
        print(res)
    resultsFile.write(str(model.similar_by_vector(center, topn=50)) + "\n")
    additionalWords['center' + str(c)] = center
    c+=1
print("--------")
print(wordCount)
print(len(wordVectors))

####################### TDK SENSE ######################

startJVM(
    getDefaultJVMPath(),
    '-ea',
    f'-Djava.class.path=D:/Engineering Project/Thesis/Thesis/Code/zemberek-full.jar',
    convertStrings=False
)

TurkishMorphology: JClass = JClass('zemberek.morphology.TurkishMorphology')
WordAnalysis: JClass = JClass('zemberek.morphology.analysis.WordAnalysis')

TurkishTokenizer: JClass = JClass('zemberek.tokenization.TurkishTokenizer')
Token: JClass = JClass('zemberek.tokenization.Token')

morphology: TurkishMorphology = TurkishMorphology.createWithDefaults()

tokenizer: TurkishTokenizer = TurkishTokenizer.builder().ignoreTypes(
        Token.Type.Punctuation,
        Token.Type.NewLine,
        Token.Type.SpaceTab,
        Token.Type.URL
    ).build()

# importing the requests library 
import requests 
  
# api-endpoint 
URL = "https://sozluk.gov.tr/gts"
  
# location given here 
  
# defining a params dict for the parameters to be sent to the API 
PARAMS = {'ara':word} 
  
# sending get request and saving the response as response object 
r = requests.get(url = URL, params = PARAMS) 
  
# extracting data in json format 
data = r.json() 

senseList = []
senseVectors = {}
for i in range(len(data)):
    for j in range(len(data[i]['anlamlarListe'])):
        sense = data[i]['anlamlarListe'][j]['anlam']
        senseList.append(sense)
        print(sense)
        print("-----------------")
    print("+++++++++++++++++")
    print("+++++++++++++++++")


removeList = []

removeFile = open("remove", "r", encoding="utf-8")

for line in removeFile:
    line = line.strip()
    removeList.append(line)

removeFile.close()

c = 0
for sense in senseList:
    c+=1
    senseTok = tokenize_tr(sense)
    senseTok = [word for word in senseTok if word not in removeList]
    senseTok = " ".join(senseTok)
    analysis: java.util.ArrayList = (
    morphology.analyzeAndDisambiguate(senseTok).bestAnalysis()
    )
    pos: List[str] = []
    for i, analysis in enumerate(analysis, start=1):
        pos.append(
            f'{str(analysis.getLemmas()[0])}'
            )
    senseTok = remove_elements_from_list(pos, ["UNK", "lt", "gt"])

    senseVector = get_mean_vector(model, senseTok)
    additionalWords['sense' + str(c)] = senseVector
    senseVectors[sense] = senseVector

from scipy.spatial import distance
c = 0
dictionary = {}
for center in kmeans.cluster_centers_:
    c+=1
    k = 0
    maxx = 0
    best_sense = "Best sense not found"
    for key in senseVectors.keys():
        k+=1
        vec1 = center
        vec2 = senseVectors[key]
        if np.sum(vec1*vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)) > maxx:
            maxx = np.sum(vec1*vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
            best_sense = key

        print("----- " + str(c) + " -------- " + str(key) + " ---------")
        print(np.sum(vec1*vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
        resultsFile.write("----- " + str(c) + " -------- " + str(key) + " ---------" + "\n")
        resultsFile.write(str(np.sum(vec1*vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))) + "\n")
        print(distance.euclidean(vec1, vec2))
    dictionary[c] = best_sense

    
for key in dictionary.keys():
    print("----------------------")
    print("Best sense selected for center " + str(key) + ": \n" + dictionary[key])
    resultsFile.write("Best sense selected for center " + str(key) + ": \n" + dictionary[key] + "\n")

print("----------------------")

resultsFile.close()

####################### TDK SENSE ######################
    
# startJVM(
#     getDefaultJVMPath(),
#     '-ea',
#     f'-Djava.class.path=E:/CSE/Thesis/Thesis/Code/zemberek-full.jar',
#     convertStrings=False
# )

# TurkishMorphology: JClass = JClass('zemberek.morphology.TurkishMorphology')
# WordAnalysis: JClass = JClass('zemberek.morphology.analysis.WordAnalysis')

# TurkishTokenizer: JClass = JClass('zemberek.tokenization.TurkishTokenizer')
# Token: JClass = JClass('zemberek.tokenization.Token')

# morphology: TurkishMorphology = TurkishMorphology.createWithDefaults()

# tokenizer: TurkishTokenizer = TurkishTokenizer.builder().ignoreTypes(
#         Token.Type.Punctuation,
#         Token.Type.NewLine,
#         Token.Type.SpaceTab,
#         Token.Type.URL
#     ).build()
 
# ###################### TEST ########################

# removeList = []

# corpus_dir = "E:/CSE/Thesis/Thesis/Code/corpus/"

# removeFile = open("remove", "r", encoding="utf-8")

# for line in removeFile:
#     line = line.strip()
#     removeList.append(line)
    
# removeFile.close()

# sentences = []

# # sentence1 = "mevsimlerden yazdı"
# # sentence2 = "orhan 1948 yılında son kitabını yazdı"
# # sentence3 = "hasan adını tahtaya yazdı"
# # sentence4 = "bugüne kadarki en güzel şiirini yazdı"
# # sentence5 = "1918 den beri en sıcak yazdı"
# # sentence6 = "yaz aylarını çok severdi"
# # sentence7 = "aklındakileri yazmayı çok severdi"

# # sentences = ['pazar akşamları derin düşüncelere dalıyordu',
# #               'Henüz kapkaranlıktı dışarısı ve derin bir sessizlik içindeydi ev.',
# #               'Derin konulara girmeden önce bilgilerini gözden geçir.',
# #               'Su çok derin olduğu için boğulma tehlikesi yaşadık',
# #               'Derin sularda yüzmeden çok iyi yüzme bilmen gerektiğini bilmen gerekirdi.',
# #               'Genç kız onun kırık dişli ağzının içindeki derin karanlığa bakıyor.',
# #               'çukur çok derin gözüküyordu',
# #               'Öğle saatlerinde derin bir uykuya daldı',
# #               'Bu tarifi yapabilmek için derin bir kaba ihtiyacımız var.',
# #     ]


# test_dir = "E:/CSE/Thesis/Thesis/Code/test/"
# ayFile = open(test_dir + word + '-test', "r", encoding="utf-8")
# predFile = open(test_dir + word + '-pred.txt', "w", encoding="utf-8")
# for example in ayFile:
#     sentences.append(example)



# stemmedSentences = []
# for sentence in sentences:   
#     tokenizedLine = []
#     for i, token in enumerate(tokenizer.tokenizeToStrings(JString(sentence))):
#         tokenizedLine.append(str(token))
#     tokenizedLine = " ".join(tokenizedLine)
#     if tokenizedLine == "" or tokenizedLine == " ":
#         continue
#     analysis: java.util.ArrayList = (
#     morphology.analyzeAndDisambiguate(tokenizedLine).bestAnalysis()
#     )
#     pos: List[str] = []
#     for i, analysis in enumerate(analysis, start=1):
#         pos.append(
#             f'{str(analysis.getLemmas()[0])}'
#             )
#     pos = remove_elements_from_list(pos, ["UNK", "lt", "gt"])
#     analyzedLine = " ".join(pos)
#     stemmedSentences.append(analyzedLine)

# testVectors  = []
# for sentence in stemmedSentences:
#     sentence = " ".join(tokenize_tr(sentence))
#     words = sentence.split()
#     words = [word for word in words if word not in removeList]
#     firstIndex = words.index(word) - windowSize if words.index(word) - windowSize > 0 else 0
#     lastIndex = words.index(word) + windowSize + 1 if words.index(word)  + windowSize < len(words) else len(words)
#     contextWords = words[firstIndex:lastIndex]
#     wordVector = get_mean_vector(model, contextWords)
#     testVectors.append(wordVector)

# from scipy.spatial import distance
# sentence = 0
# max_sentence = 0
# for vector in testVectors:
#     sentence+=1
#     sense = 0
#     max_sense = 0
#     maxx = 0
#     for center in kmeans.cluster_centers_:
#         sense+=1
#         vec1 = center
#         vec2 = vector
#         if np.sum(vec1*vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)) > maxx:
#             maxx = np.sum(vec1*vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
#             max_sense = sense
#         print("----- " + str(sentence) + " -------- " + str(sense) + " ---------")
#         print(np.sum(vec1*vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
#         print(distance.euclidean(vec1, vec2))
#         resultsFile.write("----- " + str(sentence) + " -------- " + str(sense) + " ---------" + "\n")
#         resultsFile.write(str(np.sum(vec1*vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))) + "\n")
#         resultsFile.write(str(distance.euclidean(vec1, vec2)) + "\n")
#     resultsFile.write("Sentence " + str(sentence) + " prediction is:  " + str(max_sense)+ "\n")
#     predFile.write(str(max_sense) + "\n")
# predFile.close()
# resultsFile.close()
    
# ####################### TEST ########################
    
# ####################### GRAPHS ########################

# additionalWords = {}
# tsne_plot(model, 40, additionalWords)
# tsne_plot(model, 30, additionalWords)
# tsne_plot(model, 20, additionalWords)
# tsne_plot(model, 10, additionalWords)

# tsne_plot(model, 45, additionalWords)
# tsne_plot(model, 35, additionalWords)
# tsne_plot(model, 25, additionalWords)
# tsne_plot(model, 15, additionalWords)
# tsne_plot(model, 50, additionalWords)

# tsne_plot(model, 5, additionalWords)
# tsne_plot(model, 55, additionalWords)
# tsne_plot(model, 60, additionalWords)
# tsne_plot(model, 65, additionalWords)
# tsne_plot(model, 70, additionalWords)

# tsne_plot(model, 75, additionalWords)
# tsne_plot(model, 80, additionalWords)
# tsne_plot(model, 85, additionalWords)
# tsne_plot(model, 90, additionalWords)
# tsne_plot(model, 95, additionalWords)

# tsne_plot(model, 100, additionalWords)
# tsne_plot(model, 1, additionalWords)
# tsne_plot(model, 2, additionalWords)
# tsne_plot(model, 3, additionalWords)
# tsne_plot(model, 4, additionalWords)

# ####################### GRAPHS ########################

# inspect = []
# inspect.append(additionalWords["center1"])
# inspect.append(additionalWords["center2"])
# inspect.append(model["ay"])
# inspect.append(model["haziran"])
# inspect.append(model["yirmi"])

# shutdownJVM()


