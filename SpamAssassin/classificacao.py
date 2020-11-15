import glob
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer


paths = ["./dados/20021010_easy_ham/easy_ham/*",
         "./dados/20021010_hard_ham/hard_ham/*",
         "./dados/20021010_spam/spam/*",
         "./dados/20030228_easy_ham/easy_ham/*",
         "./dados/20030228_easy_ham_2/easy_ham_2/*",
         "./dados/20030228_hard_ham/hard_ham/*",
         "./dados/20030228_spam/spam/*",
         "./dados/20030228_spam_2/spam_2/*",
         "./dados/20050311_spam_2/spam_2/*"]

list_of_files_spam = []
list_of_files_ham  = []

for path in paths:
    if 'spam' in path:
        list_of_files_spam += glob.glob(path)
    else:
        list_of_files_ham += glob.glob(path)

df = []
for file in list_of_files_spam:
    conteudo = open(file, 'r', encoding = 'cp437')
    df.append([conteudo.read(), 1])
    conteudo.close()
for file in list_of_files_ham:
    conteudo = open(file, 'r', encoding = 'cp437')
    df.append([conteudo.read(), 0])
    conteudo.close()    

import pandas as pd
df = pd.DataFrame(df)
df.columns = ['Text', 'Label']
vectorizer = TfidfVectorizer(norm="l1", stop_words="english",max_df=0.95, min_df=2)

X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Label'], test_size=0.3, random_state=0)

vectorizer.fit(X_train)
vectorizer.fit(X_test)

tfidf_train = vectorizer.transform(X_train).toarray()
tfidf_test = vectorizer.transform(X_test).toarray()

gnb = GaussianNB()
gnb.fit(tfidf_train, y_train)
y_true, y_pred = y_test, gnb.predict(tfidf_test)
print(classification_report(y_true, y_pred))

