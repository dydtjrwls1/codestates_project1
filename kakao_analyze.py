import collections
import re
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model_train import load_dataframe, tokenizing, word_counter, token_filtering, load_dataframe

def file_open(name):
    file_name = [f_name for f_name in os.listdir('./카톡대화') if name in f_name][0]
    file = open('./카톡대화/' + file_name, encoding='utf-8')
    return file

def sep_talks(file):
    '''
    카카오톡 내보내기 텍스트 파일내용에서 사용자 별 대화내용을 딕셔너리로 반환.
    return : dict(사용자이름1 : [대화내용], 사용자이름2 : [대화내용])
    '''
    logs = {}
    p = re.compile(r'\[([가-힣]{3})\] \[오[후|전] [0-9]+:[0-9]+\] (.+)')
    temp_name = ''
    temp_talk = ''
    for log in kakao_file:
        m = p.search(log)
        if not m:
            continue
        name, talk = m.group(1, 2)
        
        if temp_name != name:
            logs[temp_name] = logs.get(temp_name, []) + [temp_talk]
            temp_name = name
            temp_talk = talk
        else:
            if not talk:
                continue
            temp_talk = temp_talk + ' ' + talk
        
    return logs

df = load_dataframe()
df['tokens'] = df['text'].apply(lambda x: tokenizing(x))
wc = word_counter(df['tokens'])

maxlen = int(np.mean([len(text) for text in df['text']]))
loaded_model = load_model('GRU_model.h5')

kakao_file = file_open('박은비')
logs = sep_talks(kakao_file)
del logs['']

scores = {}
for name, talks in logs.items():
    logs[name] = [tokenizing(talk) for talk in talks]
    logs[name] = [talk for talk in logs[name] if talk] # 결측지 제거
    logs[name] = token_filtering(logs[name], wc, 10000)
    logs[name] = pad_sequences(logs[name], maxlen=maxlen)
    scores[name] = loaded_model.predict(logs[name])

emotion = {'분노':0, '슬픔':1, '불안':2, '상처':3, '당황':4, '기쁨':5}
emotion = { value: key for key, value in emotion.items() }
emotions = { name: [] for name in scores.keys()}
for name, score in scores.items():
    emo_lst = [emotion[np.argmax(emo)] for emo in score]
    emotions[name] = emo_lst

kakao_file = file_open('박은비')
logs = sep_talks(kakao_file)
for talk, emo in zip(logs['박은비'], emotions['박은비']):
    print('대화 :', talk)
    print('감정 :', emo)
    print('-----------------------------------')

'''
emotion = {'분노':0, '슬픔':1, '불안':2, '상처':3, '당황':4, '기쁨':5}
emotion = { value: key for key, value in emotion.items() }
emotions = { name: {} for name in scores.keys()}
for name, score in scores.items():
    idx_lst = [np.argmax(emo) for emo in score]
    for idx in idx_lst:
        logs[name]
        emotions[name][emotion[idx]] = emotions[name].get(emotion[idx], 0) + 1
'''
'''
plt.rcParams['font.family'] = 'Malgun Gothic'
fig, ax = plt.subplots(ncols=2)
i = 0
for name, emo in emotions.items():
    sns.barplot(list(emo.values()), list(emo.keys()), ax=ax[i])
    ax[i].set_title(name)
    i += 1
plt.show()
'''
