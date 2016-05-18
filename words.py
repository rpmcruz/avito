# -*- coding: utf-8 -*-

from tictoc import tic, toc
import pickle
import pandas as pd
from wordcloud import WordCloud
from stop_words import get_stop_words
stopwords = get_stop_words('ru')

from googleapiclient.discovery import build
key = 'AIzaSyB2y8w1jnsrrjXB1mG5DcdcHw_Zch8mmqs'


def translate(text):
    service = build('translate', 'v2', developerKey=key)
    tr = service.translations().list(
        source='ru', target='en', q=text).execute()
    return tr['translations'][0]['translatedText']


print 'load csv...'
tic()
Xinfo = pd.read_csv('data/ItemInfo_train.csv', index_col=0,
                    usecols=[0, 1, 2, 3], encoding='utf-8')
toc()
Xpair = pd.read_csv('data/ItemPairs_train.csv', usecols=[0, 1, 2])
with open('idxmap.pickle', 'rb') as f:
    idxmap = pickle.load(f)

dpairs = (
    Xpair[Xpair['isDuplicate'] == 0].as_matrix(['itemID_1', 'itemID_2']),
    Xpair[Xpair['isDuplicate'] == 1].as_matrix(['itemID_1', 'itemID_2']),
)

from categorias import categorias
import matplotlib.pyplot as plt
plt.ioff()

for category in Xinfo['categoryID'].unique():
    tic()
    for dup in (0, 1):
        pairs = dpairs[dup].flatten()
        items = Xinfo.iloc[idxmap[pairs]]
        items = items[items['categoryID'] == category]
        try:
            if len(items):
                words = [t.split() for t in items['description']]
                words = [word for sublist in words for word in sublist]
                words = [word for word in words if word not in stopwords]
                #words = translate(words)
                text = u'\n'.join(words)
                wordcloud = WordCloud().generate(text)

                plt.subplot(1, 2, dup+1)
                plt.imshow(wordcloud)
                if dup == 0:
                    plt.title(categorias[category])
        except Exception as ex:
            print 'Wordcloud Error - ignoring:', ex
    plt.axis('off')
    plt.show()
    toc()
