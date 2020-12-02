"""
Índice
- Bibliotecas
- Leitura e preparação dos dados
- Função principal do experimento
- Módulos de pré-processamento
- Experimentos
"""

import datetime
import json
import os
import statistics
import time
from itertools import permutations

import nltk
import pandas as pd
from memory_profiler import memory_usage

from preprocessing import remocao_de_stopwords, lemmatization, stemming
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from utils import create_df, calcular_custo, measure_execution_function, SPAM, HAM

# Necessários apenas na 1º run
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
#

# ---------- VARIÁVEIS DE CONFIGURAÇÃO ----------
DATASET_DIRECTORY = "./dados/**"

# -> Gerará combinações diferentes de embaralhamento de treino e predição.
RANDOM_STATES = range(1, 3)
# --------------------------------

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

    new_df = df # verificar se é diferente.

    # Aplicação do pré-processamento.
    for prepros in prepros_list:
        if prepros is not None:
            func_result, new_df = measure_execution_function(prepros, new_df)
            print("{} demorou {} ns ({} s). [{} bytes ({} KiB)"
                  .format(prepros.__name__,
                          func_result['tempo'],
                          func_result['tempo'] / pow(10, 9),
                          func_result['memoria_media'],
                          func_result['memoria_media'] / (1024)))
            result[prepros.__name__] = func_result

    # Treinamento
    for random in RANDOM_STATES:
        y_true = []
        y_pred = []
        rodada = {}

        timer = time.thread_time_ns()

        vectorizer = TfidfVectorizer(norm="l1", max_df=0.95, min_df=2)
        x_train, x_test, y_train, y_test = train_test_split(df['Text'], df['Label'], test_size=0.3, random_state=random)

        vectorizer.fit(x_train)
        vectorizer.fit(x_test)

        tfidf_train = vectorizer.transform(x_train).toarray()
        tfidf_test = vectorizer.transform(x_test).toarray()

        gnb = GaussianNB()
        gnb.fit(tfidf_train, y_train)
        timer = time.thread_time_ns() - timer

        y_true += list(y_test)
        y_pred += list(gnb.predict(tfidf_test))

        errados = 0
        for i in range(0, len(y_true)):
            if y_true[i] != y_pred[i]:
                errados += 1

        # Feito outro treinamento pra verificar o consumo de memória.
        memory = memory_usage((train_test_split, (df['Text'], df['Label'],), {'test_size': 0.3, 'random_state': random},), interval=1)
        memory = statistics.mean(memory)
        rodada["memoria_media"] = memory

        # Parte de coleta de métricas.
        stat_acertos.append(len(y_true) - errados)
        stat_erros.append(errados)

        rodada["tempo"] = timer
        classificador_resultado = classification_report(y_true, y_pred, output_dict=True)
        rodada["resultado"] = {'spam': classificador_resultado[str(SPAM)], 'ham': classificador_resultado[str(HAM)]}

        rodada["corretos"] = len(y_true) - errados
        rodada["errados"] = errados
        result["rodada_{}".format(random)] = rodada
        rodadas.append(timer)

        print("-> Rodada_{} demorou {} ns ({} s)".format(random, timer, timer / pow(10, 9)))
        print(classification_report(y_true, y_pred))
        print('-> Corretos: %d\tErrados: %d' % (len(y_true) - errados, errados))
        print('-> Memória: {} bytes ou {} KiB'.format(memory, memory/1024))

    tempo_total = time.thread_time_ns() - tempo_total

    # relatório
    result["tempo_total"]               = tempo_total
    result["tempo_medio_rodadas"]       = statistics.mean(rodadas)
    result["Desvio padrao das rodadas"] = statistics.stdev(rodadas)

    result["media_acertos"]     = statistics.mean(stat_acertos)
    result["desvio_acertos"]    = statistics.stdev(stat_acertos)

    result["media_errados"]     = statistics.mean(stat_erros)
    result["desvio_errados"]    = statistics.stdev(stat_erros)

    print("-> Média acertos: {} \u00B1 {} Média erros: {} \u00B1 {}\n".format(statistics.mean(stat_acertos),
                                                                              statistics.stdev(stat_acertos),
                                                                              statistics.mean(stat_erros),
                                                                              statistics.stdev(stat_erros)))

    print(
        "--> Experimento tempo total {} ns ({} s). Média Rodadas {} ns ({} s) \u00B1 {} ns ({} s)\n"
            .format(tempo_total,
                    tempo_total / pow(10, 9),
                    statistics.mean(rodadas),
                    statistics.mean(rodadas) / pow(10, 9),
                    statistics.stdev(rodadas),
                    statistics.stdev(rodadas) / pow(10, 9)))

    return result

### MAIN
if __name__ == "__main__":
    # Quais pré-processadores e a ordem que devem ser aplicados pra cada experimento.
    combinacoes_prepros = list(permutations([remocao_de_stopwords, stemming, lemmatization]))
    # Caso em que nenhum pré-processador é aplicado.
    combinacoes_prepros.insert(0, [])

    # Se for pra testar, limitar o número de combinações pra ser mais rápido!
    #combinacoes_prepros = combinacoes_prepros[:3]

    num_experimentos = 0
    report = {}

    horario_inicio = datetime.datetime.today().strftime("%d-%m-%Y %H-%M-%S")

    report['horario(inicio)'] = horario_inicio
    report['random_states'] = list(RANDOM_STATES)

    df = create_df(DATASET_DIRECTORY)
    df = pd.DataFrame(df)
    df.columns = ['Text', 'Label']

    # ---- Aviso ----
    total_tempo = calcular_custo(combinacoes_prepros, RANDOM_STATES)
    print(
        "!!! ATENÇÃO !!!:\nCom a configuração atual levará um tempo médio de {} segundos ou {} minutos para concluir tudo!\n".format(
            total_tempo, total_tempo / 60))
    # --------

    # Parte principal que é feita os experimentos
    for pre_process_list in combinacoes_prepros:
        exp_report = {}
        print(10 * '#' + ' Experimento [{}/{}]'.format(num_experimentos, len(combinacoes_prepros) - 1) + ' ' + 10 * '#')
        exp_report['resultado'] = experimento(df, pre_process_list)
        report[num_experimentos] = exp_report
        num_experimentos = num_experimentos + 1

    report['horario (fim)'] = datetime.datetime.today().strftime("%d-%m-%Y %H-%M-%S")

    # Geração de relatório.
    if not os.path.isdir('./resultados'):
        os.mkdir("./resultados")

    parsed = json.dumps(report, indent=4)
    with open('./resultados/resultado-{}.json'.format(horario_inicio), 'w') as file:
        file.write(parsed)
