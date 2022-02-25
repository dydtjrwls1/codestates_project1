from model_train import load_dataframe, word_counter, tokenizing, token_filtering
import seaborn as sns
import matplotlib.pyplot as plt

df = load_dataframe()
emotion = {'분노':0, '슬픔':1, '불안':2, '상처':3, '당황':4, '기쁨':5}
df['tokens'] = df['text'].apply(lambda x: tokenizing(x))
df['labels'] = df['emotion'].apply(lambda x: emotion[x])
wc = word_counter(df['tokens'])

plt.rcParams['font.family'] = 'Malgun Gothic'
sns.barplot(wc['word'][:10], wc['count'][:10])
plt.show()
