"""
Índice
- Bibliotecas
- Leitura e preparação dos dados
- Função principal do experimento
- Módulos de pré-processamento
- Experimentos
"""

### Bibliotecas
import datetime
import glob
import json
import os
from itertools import permutations

import pandas as pd
import statistics

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

# ---------- VARIÁVEIS DE CONFIGURAÇÃO ----------
DATASET_DIRECTORY = "./dados/**"

# -> Gerará combinações diferentes de embaralhamento de treino e predição.
RANDOM_STATES = range(1, 3)

# Uma rodada é constituído de: aplicação de pré-processamento e treino pra um random_state.
# Isso visa diminuir os ruídos causados nas métricas de CPU/MEM devido o computador estar executando outros programas.
NUM_ROUNDS = 10


### Leitura e preparação dos dados
def get_email_content(fileContent):
    if 'Subject' in fileContent:
        return fileContent[fileContent.find('Subject'):]
    else:
        return None


def create_df(directory):
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
                emailContent = get_email_content(conteudo.read())

                # ignorar arquivos que não contenham o campo 'subject'
                if emailContent is None:
                    continue

                df.append([emailContent, label])
                conteudo.close()
    return df


def dumb1(func_param):
    pass


def dumb2(func_param):
    pass


def measure_execution_time(func, func_param):
    timer = time.thread_time_ns()
    result = func(func_param)
    timer = time.thread_time_ns() - timer
    return timer, result


# Função que executa as funções de pré-processamento na ordem que são passadas e executa o treinamento.
# A cada pré-processamento é extraído:
# - O tempo de execução
# - Consumo de memória
# - Ciclos da CPU


### Função principal do experimento
# Executa o experimento com o dataframe df, mostra o classification_report e a quantidade de acertos e erros

# Etapas:

# Experimento vai receber a lista de preprocessamento pra aplicar. Quem vai criar a combinação é a main!

# 1 - Carregar arquivos.
# 2 - aplicar pré-processamento
# 3 - aplicar classificador
def experimento(df, prepros_list=[]):
    tempo_total = time.thread_time_ns()

    rodadas = []

    result = {}
    new_df = []

    y_true = []
    y_pred = []

    prepros_name = list(map(lambda x: x.__name__, prepros_list))
    print(
        "--> Será aplicado os seguintes pré-processadores: " + ', '.join(prepros_name if prepros_name else ['Nenhum']))

    new_df = df
    # Aplicação do pré-processamento.
    for prepros in prepros_list:
        # new_df = prepros(new_df)
        timer, new_df = measure_execution_time(prepros, new_df)
        print("{} demorou {} ns".format(prepros.__name__, timer))
        result[prepros.__name__] = timer

    for random in RANDOM_STATES:
        round = {}
        timer = time.thread_time_ns()

        vectorizer = TfidfVectorizer(norm="l1", max_df=0.95, min_df=2)

        X_train, X_test, Y_train, Y_test = train_test_split(df['Text'], df['Label'], test_size=0.3, random_state=random)

        vectorizer.fit(X_train)
        vectorizer.fit(X_test)

        tfidf_train = vectorizer.transform(X_train).toarray()
        tfidf_test = vectorizer.transform(X_test).toarray()

        gnb = GaussianNB()
        gnb.fit(tfidf_train, Y_train)
        timer = time.thread_time_ns() - timer

        # round["X_train"] = X_train
        # round["Y_train"] = Y_train
        # round["X_test"] = X_test
        # round["Y_test"] = Y_test

        round["Tempo"] = timer
        result["Rodada_{}".format(random)] = round
        rodadas.append(timer)

        print("Rodada_{} demorou {} ns ({} s)".format(random, timer, timer / pow(10, 9)))

    tempo_total = time.thread_time_ns() - tempo_total
    result["Tempo Total"] = tempo_total
    result["Tempo medio rodadas"] = statistics.mean(rodadas)
    result["Desvio padrao das rodadas"] = statistics.stdev(rodadas)
    print("--> Experimento tempo total {} ns ({} s). Média Rodadas {} ns ({} s) \u00B1 {} ns ({} s)".format(tempo_total,
                                                                                                            tempo_total / pow(
                                                                                                                10, 9),
                                                                                                            statistics.mean(
                                                                                                                rodadas),
                                                                                                            statistics.mean(
                                                                                                                rodadas) / pow(
                                                                                                                10, 9),
                                                                                                            statistics.stdev(
                                                                                                                rodadas),
                                                                                                            statistics.stdev(
                                                                                                                rodadas) / pow(
                                                                                                                10, 9)
                                                                                                            ))
    return result

    #
    #
    # y_true += list(Y_test)
    # y_pred += list(gnb.predict(tfidf_test))
    #
    # print(classification_report(y_true, y_pred) + '\n')
    # errados = 0
    # for i in range(0, len(y_true)):
    #     if y_true[i] != y_pred[i]:
    #         errados += 1
    # print('Corretos: %d\nErrados: %d\n' % (len(y_true) - errados, errados))


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


### MAIN
if __name__ == "__main__":
    report = {}

    horario_inicio = datetime.date.today().strftime("%d-%m-%Y %H-%M-%S")
    report['horario (inicio)'] = horario_inicio
    report['random states'] = list(RANDOM_STATES)
    exp_report = {}

    df = create_df(DATASET_DIRECTORY)
    df = pd.DataFrame(df)
    df.columns = ['Text', 'Label']

    pre_processadores = [dumb1, dumb2]

    # exp_report["experimento 1"] = {
    #     "pre-processadores": ', '.join(pre_processadores if pre_processadores else ['Nenhum'])}
    exp_report["resultado"] = experimento(df, pre_processadores)
    report[0] = exp_report

    report['horario (fim)'] = datetime.date.today().strftime("%d-%m-%Y %H-%M-%S")

    # Geração de log.
    parsed = json.dumps(report)
    with open('resultado-{}.json'.format(horario_inicio), 'w') as file:
        file.write(parsed)

    ### Experimento
    # experimento(df, [remocao_de_stopwords, stemming, lemmatization])
    ###
