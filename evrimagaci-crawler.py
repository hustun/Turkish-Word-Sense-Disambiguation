# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:34:00 2020

@author: hasbe
"""

from gensim import utils
def tokenize_tr(content,token_min_len=2,token_max_len=50,lower=True):
	if lower:
		lowerMap = {ord(u'A'): u'a',ord(u'A'): u'a',ord(u'B'): u'b',ord(u'C'): u'c',ord(u'Ç'): u'ç',ord(u'D'): u'd',ord(u'E'): u'e',ord(u'F'): u'f',ord(u'G'): u'g',ord(u'Ğ'): u'ğ',ord(u'H'): u'h',ord(u'I'): u'ı',ord(u'İ'): u'i',ord(u'J'): u'j',ord(u'K'): u'k',ord(u'L'): u'l',ord(u'M'): u'm',ord(u'N'): u'n',ord(u'O'): u'o',ord(u'Ö'): u'ö',ord(u'P'): u'p',ord(u'R'): u'r',ord(u'S'): u's',ord(u'Ş'): u'ş',ord(u'T'): u't',ord(u'U'): u'u',ord(u'Ü'): u'ü',ord(u'V'): u'v',ord(u'Y'): u'y',ord(u'Z'): u'z'}
		content = content.translate(lowerMap)
	return [
	utils.to_unicode(token) for token in utils.tokenize(content, lower=False, errors='ignore')
	if token_min_len <= len(token) <= token_max_len and not token.startswith('_')
	]

# importing the requests library 
import requests 
  
# api-endpoint 
URL = "https://evrimagaci.org/sitemaps"
  
# location given here 
# keyword = "yüz"
  
# defining a params dict for the parameters to be sent to the API 
# PARAMS = {'ara':keyword} 
  
# sending get request and saving the response as response object 
r = requests.get(url = URL) 
  
# extracting data in json format 
# data = r.json() 

links = []

response = r.text

target1 = "<loc>"
target2 = "</loc>"

while target1 in response:
    startIndex = response.find(target1) + len(target1)
    endIndex = response.find(target2)
    
    # print(response[startIndex:endIndex])
    if "/contents/" in response[startIndex:endIndex]:
        links.append(response[startIndex:endIndex])
    response = response.replace(target1, "", 1)
    response = response.replace(target2, "", 1)

i=0
corpus_dir = "E:/CSE/Thesis/Thesis/Code/corpus/"
postsFile = open(corpus_dir + 'posts12.txt', "a", encoding="utf-8")
for link in links:
    print(f'Link {i}')
    print(link)
    if "hurriyetdailynews" in link:
        continue
    i+=1
    URL = link
    r = requests.get(url = URL) 
    
    response = r.text
    
    posts = []
    
    while target1 in response:
        startIndex = response.find(target1) + len(target1)
        endIndex = response.find(target2)
            
        # print(response[startIndex:endIndex])
        posts.append(response[startIndex:endIndex])
        response = response.replace(target1, "", 1)
        response = response.replace(target2, "", 1)
        
    c = 0
    corpus = []
    tokenizedCorpus = []
    print(len(posts))
    for post in posts:
        c+=1
        if c % 100 == 0:
            print(f'Post {c}')
        URL = post
        r = requests.get(url = URL) 
        
        if r.status_code != 200:
            continue
        response = r.text
        # print(response)
        
        import re
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response, 'html.parser')
        
        # print("--------")
        # print("--------")
        str1 =  ""
        if "/yerel-haberler/" in URL:
            header = soup.find_all("div", class_=re.compile("news-detail-spot"))
            article = soup.find_all("div", class_=re.compile("news-detail-text"))
            
            header = BeautifulSoup(str(header), 'html.parser')
            article = BeautifulSoup(str(article), 'html.parser')
            
            for header in header.find_all('h2'):
                str1 = str1 + " " + header.text
                # print(header.text)
        elif "/yazarlar/" in URL:
            header = soup.find_all("h1", class_=re.compile("article-title"))
            desc = soup.find_all("div", class_=re.compile("news-description"))
            article = soup.find_all("div", class_=re.compile("news-text"))
            
            header = BeautifulSoup(str(header), 'html.parser')
            desc = BeautifulSoup(str(desc), 'html.parser')
            article = BeautifulSoup(str(article), 'html.parser')
            
            str1 = header.get_text() + " " + desc.get_text()
            
            str1 = str1.replace('[', '')
            str1 = str1.replace(']', '')
        else:
            # header = soup.find_all("h2", class_=re.compile("rhd-article-spot"))
            article = soup.find_all("div", class_="content")
            
            # header = BeautifulSoup(str(header), 'html.parser')
            article = BeautifulSoup(str(article), 'html.parser')
            
            # str1 = header.get_text()
            # str1 = str1.replace('[', '')
            # str1 = str1.replace(']', '')
            
        # print("--------")
        # print(post)
        # print("--------")
        str2 = ""
        if (article.get_text().strip("[").strip("]").strip().startswith("Terim")):
            l = [child.strip() for child in article.find('div').children if "<" not in str(child) and len(child) > 1 ]
            s = " ".join(l)
            if s == "":
                for text in article.find_all('p'):
                            s = s + " " + text.text
            s = s.strip()
            str2 = str2 + s
        else:
            for text in article.find_all('p'):
                str2 = str2 + text.text
                # print(text.text)
                # print(tokenize_tr(text.text))
        strPost = str2
        if(strPost == "" or strPost == " "):
            print(URL)
        corpus.append(strPost)
        tokenizedCorpus.append(tokenize_tr(strPost))
        
        postsFile.write(" ".join(tokenize_tr(strPost)) + "\n")

postsFile.close()
# for i in range(len(data)):
#     for j in range(len(data[i]['anlamlarListe'])):
#         print(data[i]['anlamlarListe'][j]['anlam'])
#         print("-----------------")
#     print("+++++++++++++++++")
#     print("+++++++++++++++++")