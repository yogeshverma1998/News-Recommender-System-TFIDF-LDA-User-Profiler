''' News Recommender System'''

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import random
import sys
import time

import os
import math
import time
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px

# Below libraries are for text processing using NLTK
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize


# Below libraries are for feature representation using sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Below libraries are for similarity matrices using sklearn
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans

#Topic Modelling libraries
from IPython.core.display import display
import gensim
from gensim.parsing.preprocessing import preprocess_string
pd.set_option('display.max_colwidth', 200)
# %matplotlib inline

!pip install pyLDAvis

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


news_corpus = pd.read_csv('news_new.csv')

news_corpus = news_corpus.drop(news_corpus.columns[0],axis=1)

news_corpus = news_corpus.dropna()  #Removing Nan objects

news_corpus = news_corpus.reset_index()
news_corpus = news_corpus.drop(news_corpus.columns[0],axis=1)

news_corpus_arr = shuffle(news_corpus.to_numpy())

news_corpus = pd.DataFrame(news_corpus_arr,columns=["Id","title","content"])

news_articles_temp = news_corpus.copy()

"""## DATA PREPROCESSING

Stop-Words Removal
"""

stop_words = set(stopwords.words('english'))  #Removing Stopwords

for i in range(len(news_articles_temp["title"])):  #Removing StopWords from title
    string = ""
    for word in news_articles_temp["title"][i].split():
        word = ("".join(e for e in word if e.isalnum()))
        word = word.lower()
        if not word in stop_words:
          string += word + " "  
    news_articles_temp.at[i,"title"] = string.strip()

for i in range(len(news_articles_temp["content"])):  #Removing StopWords from content
    string = ""
    for word in news_articles_temp["content"][i].split():
        word = ("".join(e for e in word if e.isalnum()))
        word = word.lower()
        if not word in stop_words:
          string += word + " "  
    news_articles_temp.at[i,"content"] = string.strip()

"""Lemmatization"""

lemmatizer = WordNetLemmatizer()

#Title Lemmatization
for i in range(len(news_articles_temp["title"])):
    string = ""
    for w in word_tokenize(news_articles_temp["title"][i]):
        string += lemmatizer.lemmatize(w,pos = "v") + " "
    news_articles_temp.at[i, "title"] = string.strip()

#Content Lemmatization
for i in range(len(news_articles_temp["content"])):
    string = ""
    for w in word_tokenize(news_articles_temp["content"][i]):
        string += lemmatizer.lemmatize(w,pos = "v") + " "
    news_articles_temp.at[i, "content"] = string.strip()

"""### TF-IDF method + Euclidean Similarity of articles recommendation

TF-IDF method to represent document in a d-dimensional vector and then recommend the articles based on content similarity betweeen the previous article andthe corpus using Eucledian distance as a metric.
"""

"""Article Recommending based on Title only"""
tfidf_headline_vectorizer = TfidfVectorizer(min_df = 0)
tfidf_headline_features = tfidf_headline_vectorizer.fit_transform(news_articles_temp["title"])
def tfidf_title_model(row_index, num_similar_items):
    l_indice = []
    couple_dist = pairwise_distances(tfidf_headline_features,tfidf_headline_features[row_index])
    indices = np.argsort(couple_dist.ravel())[0:num_similar_items]
    for i in indices:
        l_indice.append(i)
    #return df.iloc[1:,1]
    return l_indice

"""Article Recommending based on Content only"""
tfidf_headline_vectorizer = TfidfVectorizer(min_df = 0)
tfidf_headline_features = tfidf_headline_vectorizer.fit_transform(news_articles_temp["content"])
def tfidf_content_model(row_index, num_similar_items):
    l_indice = []
    couple_dist = pairwise_distances(tfidf_headline_features,tfidf_headline_features[row_index])
    indices = np.argsort(couple_dist.ravel())[0:num_similar_items]
    for i in indices:
        l_indice.append(i)   
    #return df.iloc[1:,1]
    return l_indice

"""# LDA Topic Modelling

LDA topic modelling to get Topic distribution in each document. Prespecified number of topics = 15. Based on similarity between topic distribution recommend articles
"""

vect =TfidfVectorizer(stop_words=stop_words,max_features=1000) #convert to d dimensional vector using TFIDF
vect_text=vect.fit_transform(news_articles_temp['content'])
from sklearn.decomposition import LatentDirichletAllocation
lda_model=LatentDirichletAllocation(n_components=15,learning_method='online',random_state=42,max_iter=1) #topics = 15
lda_top=lda_model.fit_transform(vect_text)

def lda_recommend(row_index,num_items):
  ar = lda_top[row_index]
  ar = ar.reshape(ar.shape[0],-1)
  ar = ar.transpose()
  ar.shape
  couple_dist = pairwise_distances(lda_top,ar)
  l_indice = []
  indices = np.argsort(couple_dist.ravel())[0:num_items]
  for i in indices:
      l_indice.append(i)

  return l_indice

"""# Cold Start Problem

Used the generated LDA topic distribution for each document to cluster the documents in 10 clusters and randomly selecting news article from each cluster to Recommend 10 news article to user
"""

def cold_start_problem():
  print("Welcome to our News Recommender System!. We are gald that you came here. Here are the current latest news:\n")
  print("Before Going to Start, Do you want Content+ Headline(1) or only Headline(0) for News, Type the integer(given in bracket) next to choice below:")
  g = input("Choice : ")
  l1 = []
  l2 = []
  l3 = []
  l4 = [] 
  l5 = []
  l6 = []
  l7 = []
  l8 = []
  l9 = []
  l10 = []
  l11 = []
  from sklearn.cluster import KMeans
  csp = KMeans(n_clusters=11, verbose = 0)  
  csp.fit(lda_top)
  for i in range(len(lda_top)):
    ar = lda_top[i]
    ar = ar.reshape(ar.shape[0],-1)
    ar = ar.transpose()
    cluster = csp.predict(ar)[0]
    if cluster == 1: l1.append(i)
    if cluster == 2: l2.append(i)
    if cluster == 3: l3.append(i)
    if cluster == 4: l4.append(i)
    if cluster == 5: l5.append(i)
    if cluster == 6: l6.append(i)
    if cluster == 7: l7.append(i)
    if cluster == 8: l8.append(i)
    if cluster == 9: l9.append(i)
    if cluster == 10: l10.append(i)
  article1 = random.sample(l1, k = 1)
  article2 = random.sample(l2, k = 1)
  article3 = random.sample(l3, k = 1)
  article4 = random.sample(l4, k = 1)
  article5 = random.sample(l5, k = 1)
  article6 = random.sample(l6, k = 1)
  article7 = random.sample(l7, k = 1)
  article8 = random.sample(l8, k = 1)
  article9 = random.sample(l9, k = 1)
  article10 = random.sample(l10, k = 1)
  Articles = article1 + article2 + article3 + article4 + article5 + article6 + article7 + article8 + article9 + article10

  for i in range(len(Articles)):
    news_id = Articles[i]
    if int(g) == 1:
       print(i,"\t",news_corpus['title'][news_id],"\n",news_corpus['content'][news_id])
    elif int(g) == 0:
       print(i,"\t",news_corpus['title'][news_id])

  return Articles,g

"""# User Profiler and Feedback loop

Deployment of News Recommender System by making the feedback loop i.e selecting the articles and recommending the similar articles based on selection by user and printing of the clickstream data and user profile in text file
"""

def final_recommend(Given_Articles,g):
  l_id = []
  content_indice = []
  print("\nThank You for Using our News Recommender, If you are interested in some article kindly type number of articles and the id besides it below(integer) or If you are finished type exit\n")
  number = input("Number of articles Interested : ")
  if number == 'exit': 
    pass
  else:
     for i in range(int(number)):
         ID = input("Article id  :"  )
         l_id.append(int(ID))
 

     for i in range(len(l_id)):
         article_ID = Given_Articles[l_id[i]]
         tf_indice = tfidf_content_model(article_ID,10)
         lda_indice = lda_recommend(article_ID,10)
         tf_indice = random.sample(tf_indice, k = 5)
         lda_indice = random.sample(lda_indice, k = 5)
         for i in range(len(tf_indice)):
           content_indice.append(tf_indice[i])
           content_indice.append(lda_indice[i])

     reco_indice = random.sample(content_indice, k = 10)
     print("Recommended Articles are:")
     for i in range(len(reco_indice)):
         news_id = reco_indice[i]

         if int(g) == 1:
            print(i,"\t",news_corpus['title'][news_id],"\n",news_corpus['content'][news_id])
         elif int(g) == 0:
            print(i,"\t",news_corpus['title'][news_id]) 
     print("Do you want any more Recommendations based on above News : type yes or no")
     choice =  input("Type yes or no : ") 
     if choice == 'no': pass
     else:
          Given_Articles = reco_indice
          g = g
          final_recommend(Given_Articles,g)
  return l_id

def user_profiler():
  start = time.time()
  Given_Articles,g = cold_start_problem()
  l_id = final_recommend(Given_Articles,g)
  try:
    f = open('user_data.txt')
    f.close()
    status = 1
  except FileNotFoundError:
    status = 0
  
  if status == 0:
      f = open("user_data.txt","w+")
      f.write("Article_ID")
      f.write("\t")
      f.write("User")
      f.write("\t")
      f.write("Time_elapsed")
      f.write("\n")
      for i in range(len(l_id)):
        f.write(str( Given_Articles[l_id[i]]))
        f.write("\t")
      f.write("\t")
      f.write("User_1")
      f.write("\t")
      end = time.time()
      f.write(str(end-start))
      f.write("\n")
  
  if status==1:
      fname = "user_data.txt"
      count = 0
      with open(fname, 'r') as f:
            for line in f:
                count += 1
      s = "User" + str(count)
      f = open("user_data.txt","a")
      for i in range(len(l_id)):
        f.write(str( Given_Articles[l_id[i]]))
        f.write("\t")
      f.write("\t")
      f.write(s)
      f.write("\t")
      end = time.time()
      f.write(str(end-start))
      f.write("\n")
  f.close()

user_profiler()

