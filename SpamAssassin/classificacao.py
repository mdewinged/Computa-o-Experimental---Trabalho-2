"""
Índice
- Bibliotecas
- Leitura e preparação dos dados
- Função principal do experimento
- Módulos de pré-processamento
- Experimentos
"""

### Bibliotecas
import glob
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus   import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer 

# Necessários apenas na 1º run
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
#

sw = stopwords.words('english')
###

### Leitura e preparação dos dados
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
###

### Função principal do experimento
# Executa o experimento com o dataframe df, mostra o classification_report e a quantidade de acertos e erros
def experimento(df):
    y_true = []
    y_pred = []
    
    # Números aleatórios devem ficar anotados para permitir a replicação do experimento
    for random in [69341]:
        vectorizer = TfidfVectorizer(norm="l1", max_df=0.95, min_df=2)

        X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Label'], test_size=0.3, random_state = random)
        
        vectorizer.fit(X_train)
        vectorizer.fit(X_test)
         
        tfidf_train = vectorizer.transform(X_train).toarray()
        tfidf_test = vectorizer.transform(X_test).toarray()
        
        gnb = GaussianNB()
        gnb.fit(tfidf_train, y_train)
        y_true += list(y_test)
        y_pred += list(gnb.predict(tfidf_test))
        
    print(classification_report(y_true, y_pred) + '\n')
    errados = 0
    for i in range(0, len(y_true)):
        if y_true[i] != y_pred[i]:
            errados += 1
    print('Corretos: %d\nErrados: %d\n' %(len(y_true) - errados, errados))
###

### Módulos de pré-processamento
def remocao_de_stopwords(df):
    new_df = []
    for i in range(0, len(df)):
        tokens = word_tokenize(df['Text'][i])
        s = [word for word in tokens if word not in sw]
        new_df.append([' '.join(s), df['Label'][i]])
    new_df = pd.DataFrame(new_df)
    new_df.columns = ['Text', 'Label']
    return pd.DataFrame(new_df)

def stemming(df):
    new_df = []
    lancaster = LancasterStemmer()
    for i in range(0, len(df)):
        tokens = word_tokenize(df['Text'][i])
        s = [lancaster.stem(word) for word in tokens]
        new_df.append([' '.join(s), df['Label'][i]])
    new_df = pd.DataFrame(new_df)
    new_df.columns = ['Text', 'Label']
    return pd.DataFrame(new_df)

def lemmatization(df):
    new_df = []
    lemmatizer = WordNetLemmatizer() 
    for i in range(0, len(df)):
        tokens = word_tokenize(df['Text'][i])
        s = [lemmatizer.lemmatize(word) for word in tokens]
        new_df.append([' '.join(s), df['Label'][i]])
    new_df = pd.DataFrame(new_df)
    new_df.columns = ['Text', 'Label']
    return pd.DataFrame(new_df)
###

### Experimentos
print("Teste sem pré-processamento")
experimento(df)

print("Teste com remoção de stopwords")
new_df = remocao_de_stopwords(df)
experimento(new_df)

print("Teste com stemming")
new_df = stemming(df)
experimento(new_df)

print("Teste com lemmatization")
new_df = lemmatization(df)
experimento(new_df)

print("Teste com remoção de stopwords e lemmatization")
new_df = remocao_de_stopwords(df)
new_df = lemmatization(new_df)
experimento(new_df)
###