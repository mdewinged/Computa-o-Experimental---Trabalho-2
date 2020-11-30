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
import os

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer

import time

# Necessários apenas na 1º run
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
#

sw = stopwords.words('english')


### Leitura e preparação dos dados
def getEmailContent(fileContent):
    if 'Subject' in fileContent:
        return fileContent[fileContent.find('Subject'):]
    else:
        return None


DATASET_DIRECTORY = "./dados/**"

list_of_files_spam = []
list_of_files_ham = []

for path in glob.glob(DATASET_DIRECTORY, recursive=True):
    if 'spam' in path:
        list_of_files_spam += glob.glob(path)
    else:
        list_of_files_ham += glob.glob(path)

df = []
for label, files in {1: list_of_files_spam, 0: list_of_files_ham}.items():
    for file in files:

        if os.path.isfile(file):
            conteudo = open(file, 'r', encoding='cp437')
            emailContent = getEmailContent(conteudo.read())

            # ignorar arquivos que não contenham o campo 'subject'
            if emailContent is None:
                continue

            df.append([emailContent, label])
            conteudo.close()

import pandas as pd

df = pd.DataFrame(df)
df.columns = ['Text', 'Label']


def dumb1():
    pass


def dumb2():
    pass


def measure_execution_time(func, func_param):
    timer = time.thread_time_ns()
    # result = func(func_param)
    result = func()
    timer = time.thread_time_ns() - timer
    return timer, result


# Função que executa as funções de pré-processamento na ordem que são passadas e executa o treinamento.
# A cada pré-processamento é extraído:
# - O tempo de execução
# - Consumo de memória
# - Ciclos da CPU
def experimentoCompleto(df, msg, prepros_list):
    results = {}
    result = {}

    new_df = []

    print(msg)

    # Aplicação do pré-processamento.
    for prepros in prepros_list:
        # new_df = prepros(new_df)
        timer, result = measure_execution_time(prepros, None)

        print("Aplicar {} demorou {} ns".format(prepros.__name__, timer))


###

### Função principal do experimento
# Executa o experimento com o dataframe df, mostra o classification_report e a quantidade de acertos e erros
def experimento(df):
    y_true = []
    y_pred = []

    # Números aleatórios devem ficar anotados para permitir a replicação do experimento
    for random in [69341]:
        vectorizer = TfidfVectorizer(norm="l1", max_df=0.95, min_df=2)

        X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Label'], test_size=0.3, random_state=random)

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
    print('Corretos: %d\nErrados: %d\n' % (len(y_true) - errados, errados))


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

experimentoCompleto(df, 'Experimento 1', [dumb1, dumb2])

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
