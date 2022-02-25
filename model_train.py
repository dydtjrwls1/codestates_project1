from inspect import CO_ASYNC_GENERATOR
import re
import numpy as np
import pymysql
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model



# 불용어 리스트 불러오기
def load_stop_wards():
    stop_wards_file = open('./말뭉치/불용어.txt', encoding='utf-8')
    stop_wards = [ ward.strip() for ward in stop_wards_file ]
    stop_wards_file.close()
    return stop_wards

# 토큰화 진행 함수
def tokenizing(text):
    tokens = text.split(' ')
    re_tokens = [re.sub(r'[^가-힣]', '', token) for token in tokens]
    re_tokens = [ token for token in re_tokens if token ]
    stop_wards = load_stop_wards()
    removed_tokens = [ token for token in re_tokens if token not in stop_wards ]
    return removed_tokens

def load_dataframe():
    con = pymysql.connect(host='localhost', user='root', password='6169yong', database='emo_communicate')
    cur = con.cursor()
    sql = 'select * from conv;'
    cur.execute(sql)

    columns = ['id', 'emotion', 'text']
    df = pd.DataFrame(cur.fetchall(), columns=columns)
    
    return df

def word_counter(docs):
    word_counts = Counter()
    word_in_docs = Counter()
    total_docs = len(docs)

    for doc in docs:
        word_counts.update(doc)
        word_in_docs.update(set(doc))

    wc = pd.DataFrame(zip(word_counts.keys(), word_counts.values()), columns=['word', 'count'])
    wc['rank'] = wc['count'].rank(method='first', ascending=False)
    total = wc['count'].sum()

    wc['percent'] = wc['count'].apply(lambda x: x / total)
    wc = wc.sort_values(by='rank')
    wc['cul_percent'] = wc['percent'].cumsum()

    temp2 = zip(word_in_docs.keys(), word_in_docs.values())
    ac = pd.DataFrame(temp2, columns=['word', 'word_in_docs'])
    wc = ac.merge(wc, on='word')
    
    wc['word_in_docs_percent'] = wc['word_in_docs'].apply(lambda x: x / total_docs)

    return wc.sort_values(by='rank')

# 상위 10000개 단어만 추출하는 함수
def token_filtering(docs, wc, num=10000):
    vocab_words = { word: i + 1 for i, word in enumerate(wc['word'].iloc[:9999]) }
    filtered_tokens = [ [vocab_words[token] if token in vocab_words else 0 for token in doc] for doc in docs ]
    return pd.Series(filtered_tokens)

if __name__ == '__main__':
    # db 연결
    df = load_dataframe()
    emotion = {'분노':0, '슬픔':1, '불안':2, '상처':3, '당황':4, '기쁨':5}
    df['tokens'] = df['text'].apply(lambda x: tokenizing(x))
    df['labels'] = df['emotion'].apply(lambda x: emotion[x])
    print(df['text'].head(10))

    vocab_size = 20000
    embedding_dim = 100
    hidden_units = 128

    wc = word_counter(df['tokens'])
    df['tokens'] = token_filtering(df['tokens'], wc, vocab_size)

    max_len = int(np.mean([len(text) for text in df['text']]) + 50)
    

    X_train, X_test, y_train, y_test = train_test_split(df['tokens'], df['labels'], test_size=.2, random_state=34)

    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(GRU(hidden_units))
    model.add(Dense(6, activation='softmax'))

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
    mc = ModelCheckpoint('GRU_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=64, validation_split=0.2)

    loaded_model = load_model('GRU_model.h5')
    print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))