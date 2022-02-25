from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import numpy as np
import pymysql
import pandas as pd
from collections import Counter
from model_train import tokenizing, word_counter, token_filtering

# 불용어 리스트 불러오기
stop_wards_file = open('./말뭉치/불용어.txt', encoding='utf-8')
stop_wards = [ ward.strip() for ward in stop_wards_file ]
stop_wards_file.close()


con = pymysql.connect(host='localhost', user='root', password='6169yong', database='emo_communicate')
cur = con.cursor()
sql = 'select * from conv;'
cur.execute(sql)

columns = ['id', 'emotion', 'text']
emotion = {'분노':0, '슬픔':1, '불안':2, '상처':3, '당황':4, '기쁨':5}
df = pd.DataFrame(cur.fetchall(), columns=columns)
df['tokens'] = df['text'].apply(lambda x: tokenizing(x))


wc = word_counter(df['tokens'])
vocab_words = { word: i + 1 for i, word in enumerate(wc['word'].iloc[:9999]) }
df['tokens'] = df['tokens'].apply(lambda x: token_filtering(x, vocab_words))

maxlen = int(np.mean([len(text) for text in df['text']]) + 50)
text = '더이상 날 아무도 좋아하지 않아 이렇게 살 수는 없어.'
tokens = tokenizing(text)
tokens = pad_sequences([token_filtering(tokens, vocab_words)], maxlen=maxlen)
loaded_model = load_model('GRU_model.h5')

score = loaded_model.predict(tokens)
print(score)