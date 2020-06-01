''' News Article Scraping'''
import requests
from bs4 import BeautifulSoup
import pandas as pd
import random



data = pd.read_csv('links.csv')
lis = list(data.links)

lis_sport = []
for i in range(len(lis)):
  if lis[i][25:30] == 'sport':
    lis_sport.append(lis[i])

lis_features = []
for i in range(len(lis)):
  if lis[i][25:33] == 'features':
    lis_features.append(lis[i])

lis_news = []
for i in range(len(lis)):
  if lis[i][25:29] == 'news':
    lis_news.append(lis[i])

lis_inter = []
for i in range(len(lis)):
  if lis[i][30:35] == 'inter':
    lis_inter.append(lis[i])

sport_list = random.sample(lis_sport, k = 1)
features_list = random.sample(lis_features, k = 1)
news_list = random.sample(lis_news, k = 1)
inter_list = random.sample(lis_inter, k = 1)

total_list = sport_list + features_list + news_list + inter_list

a1 = []
a2 = []
for i in total_list:
    r3 = requests.get(i)
    cp3 = r3.content
    soup3 = BeautifulSoup(cp3, 'html5lib')
    cpn3 = soup3.find('h1')
    a1.append(cpn3.get_text()[1:len(cpn3.get_text())-1])
    cpn3 = soup3.find_all('p')
    t = 0
    for j in cpn3:
        if j.find(class_ = True):
            break
        else:
            t = t + 1 
    str1 = ""
    for j in range(0, t-2):
        str1 = str1 + cpn3[j].get_text()
    a2.append(str1)

d = {'id': list(range(0,len(total_list))), 'title': a1, 'content': a2}
df = pd.DataFrame(data = d)
df.to_csv('news_new.csv')



