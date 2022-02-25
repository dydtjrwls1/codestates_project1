import json
import pymysql
import os 

emotion_dict = {'1': '분노', '2': '슬픔', '3': '불안', '4': '상처', '5': '당황', '6': '기쁨'}
con = pymysql.connect(host='localhost', user='root', password='6169yong', database='emo_communicate')
cur = con.cursor()

id = 0
file_lst = list(os.listdir('./말뭉치'))
for file in file_lst:    
    with open('./말뭉치/'+file, encoding='utf-8') as f:
        json_data = json.load(f)

    for data in json_data:
        emo = emotion_dict[data['profile']['emotion']['type'][1]]
        conv = data['talk']['content']['HS01']
        sql = 'INSERT INTO conv (id, emo, conversation) VALUES (%s, %s, %s)'
        cur.execute(sql, (id, emo, conv))
        id += 1




con.commit()