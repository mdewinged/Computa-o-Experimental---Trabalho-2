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
import statistics
import time

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Necessários apenas na 1º run
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
#

# Apenas pra diminuir magic numbers no código...
HAM  = 0
SPAM = 1

CUSTOS = {
    'remocao_de_stopwords': 47,     # esses foram os tempos médios que cada um levou na minha máquina.
    'stemming': 58,                 # tempo em segundos!
    'lemmatization': 44
}
STAT_ROD = 8                        # custo de uma rodada


sw = stopwords.words('english')

# ---------- VARIÁVEIS DE CONFIGURAÇÃO ----------
DATASET_DIRECTORY = "./dados/**"

# -> Gerará combinações diferentes de embaralhamento de treino e predição.
RANDOM_STATES = range(1, 3)

# Uma rodada é constituído de: aplicação de pré-processamento e treino pra um random_state.
# Isso visa diminuir os ruídos causados nas métricas de CPU/MEM devido o computador estar executando outros programas.
# TODO: Ainda não é utilizada no código...
NUM_ROUNDS = 10
# --------------------------------


### Leitura e preparação dos dados
def get_email_content(file_content):
    if 'Subject' in file_content:
        return file_content[file_content.find('Subject'):]
    else:
        return None


def create_df():
    list_of_files_spam = []
    list_of_files_ham = []

    for path in glob.glob(DATASET_DIRECTORY, recursive=True):
        if 'spam' in path:
            list_of_files_spam += glob.glob(path)
        else:
            list_of_files_ham += glob.glob(path)

    for label, files in {SPAM: list_of_files_spam, HAM: list_of_files_ham}.items():
        for file in files:

            if os.path.isfile(file):
                conteudo = open(file, 'r', encoding='cp437')
                email_content = get_email_content(conteudo.read())

                # ignorar arquivos que não contenham o campo 'subject'
                if email_content is None:
                    continue

                df.append([email_content, label])
                conteudo.close()
    return df


def measure_execution_time(func, func_param):
    timer = time.thread_time_ns()
    result = func(func_param)
    timer = time.thread_time_ns() - timer
    return timer, result

### Função principal do experimento
# Executa o experimento com o dataframe df, mostra o classification_report, quantidade de acertos e erros e retorna um relatório.
# Experimento vai receber a lista com a ordem de pré-processadores a serem aplicados.
def experimento(df, prepros_list=[]):
    tempo_total = time.thread_time_ns()

    rodadas = []

    result = {}

    stat_acertos = []
    stat_erros = []

    prepros_name = list(map(lambda x: x.__name__ if x is not None else 'Nenhum', prepros_list))
    print("--> Será aplicado os seguintes pré-processadores: "
          + ', '.join(prepros_name if prepros_name is not None else ['Nenhum']))

    new_df = df

    # Aplicação do pré-processamento.
    for prepros in prepros_list:
        if prepros is not None:
            timer, new_df = measure_execution_time(prepros, new_df)
            print("{} demorou {} ns ({} s)".format(prepros.__name__, timer, timer / pow(10, 9)))
            result[prepros.__name__] = timer

    for random in RANDOM_STATES:
        y_true = []
        y_pred = []

        rodada = {}
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

        y_true += list(Y_test)
        y_pred += list(gnb.predict(tfidf_test))

        errados = 0
        for i in range(0, len(y_true)):
            if y_true[i] != y_pred[i]:
                errados += 1

        stat_acertos.append(len(y_true) - errados)
        stat_erros.append(errados)

        rodada["Tempo"] = timer
        classificador_resultado = classification_report(y_true, y_pred, output_dict=True)
        rodada["Resultado"] = {'spam': classificador_resultado[str(SPAM)], 'ham': classificador_resultado[str(HAM)]}

        rodada["Corretos"] = len(y_true) - errados
        rodada["Errados"] = errados
        result["Rodada_{}".format(random)] = rodada
        rodadas.append(timer)

        print("-> Rodada_{} demorou {} ns ({} s)".format(random, timer, timer / pow(10, 9)))
        print(classification_report(y_true, y_pred))
        print('-> Corretos: %d\tErrados: %d' % (len(y_true) - errados, errados))

    tempo_total = time.thread_time_ns() - tempo_total

    # relatório
    result["Tempo Total"] = tempo_total
    result["Tempo medio rodadas"] = statistics.mean(rodadas)
    result["Desvio padrao das rodadas"] = statistics.stdev(rodadas)

    result["media acertos"] = statistics.mean(stat_acertos)
    result["desvio acertos"] = statistics.stdev(stat_acertos)

    result["media errados"] = statistics.mean(stat_erros)
    result["desvio errados"] = statistics.stdev(stat_erros)

    print("-> Média acertos: {} \u00B1 {} Média erros: {} \u00B1 {}\n".format(statistics.mean(stat_acertos),
                                                                              statistics.stdev(stat_acertos),
                                                                              statistics.mean(stat_erros),
                                                                              statistics.stdev(stat_erros)))

    print(
        "--> Experimento tempo total {} ns ({} s). Média Rodadas {} ns ({} s) \u00B1 {} ns ({} s)\n".format(tempo_total,
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


def calcular_custo(prepros_list):
    custo = 0
    for pre_pros in prepros_list:
        custo = custo + STAT_ROD * len(list(RANDOM_STATES))
        for f in pre_pros:
            custo = custo + CUSTOS.get(f.__name__)
    return custo


### MAIN
if __name__ == "__main__":
    # Quais pré-processadores e a ordem que devem ser aplicados pra cada experimento.

    # combinacoes_prepros = list(combinations([None, remocao_de_stopwords, stemming, lemmatization], 3))
    combinacoes_prepros = [[], [remocao_de_stopwords], [stemming], [lemmatization],
                           [remocao_de_stopwords, lemmatization]]

    num_experimentos = 0
    report = {}
    exp_report = {}

    horario_inicio = datetime.datetime.today().strftime("%d-%m-%Y %H-%M-%S")

    report['horario (inicio)'] = horario_inicio
    report['random states'] = list(RANDOM_STATES)

    df = create_df()
    df = pd.DataFrame(df)
    df.columns = ['Text', 'Label']

    # Aviso
    total_tempo = calcular_custo(combinacoes_prepros)
    print(
        "!!! ATENÇÃO !!!:\nCom a configuração atual levará um tempo médio de {} segundos ou {} minutos para concluir tudo!\n".format(
            total_tempo, total_tempo / 60))

    for pre_process_list in combinacoes_prepros:
        print(10 * '#' + ' Experimento ' + str(num_experimentos) + ' ' + 10 * '#')
        exp_report['resultado'] = experimento(df, pre_process_list)
        report[num_experimentos] = exp_report
        num_experimentos = num_experimentos + 1

    report['horario (fim)'] = datetime.datetime.today().strftime("%d-%m-%Y %H-%M-%S")
    # Geração de relatório.
    parsed = json.dumps(report, indent=4)
    with open('resultado-{}.json'.format(horario_inicio), 'w') as file:
        file.write(parsed)
