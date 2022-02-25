import pymysql

con = pymysql.connect(host='localhost', user='root', password='6169yong', database='emo_communicate')
cur = con.cursor()

cur.execute('DROP TABLE IF EXISTS conv;')
sql = '''CREATE TABLE conv (
    id INT not null primary key,
    emo CHAR(2) not null,
    conversation TEXT
)
'''
cur.execute(sql)
con.commit()