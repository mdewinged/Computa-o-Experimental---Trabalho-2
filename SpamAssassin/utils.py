import glob
import os
import statistics
import time

# Apenas pra diminuir magic numbers no código...
from memory_profiler import memory_usage

HAM = 0
SPAM = 1

CUSTOS = {
    'remocao_de_stopwords': 47,     # esses foram os tempos médios que cada um levou na minha máquina.
    'stemming':             60,     # tempo em segundos!
    'lemmatization':        47,
}
STAT_ROD = 8                        # custo de uma rodada


def get_email_content(file_content):
    if 'Subject' in file_content:
        return file_content[file_content.find('Subject'):]
    else:
        return None


def create_df(directory_path):
    stat_total_arquivos = 0
    stat_arquivos_carregados = 0

    list_of_files_spam = []
    list_of_files_ham = []
    df = []

    for path in glob.glob(directory_path, recursive=True):
        if 'spam' in path:
            list_of_files_spam += glob.glob(path)
        else:
            list_of_files_ham += glob.glob(path)

    for label, files in {SPAM: list_of_files_spam, HAM: list_of_files_ham}.items():
        for file in files:

            if os.path.isfile(file):
                conteudo = open(file, 'r', encoding='cp437')
                email_content = get_email_content(conteudo.read())
                stat_total_arquivos = stat_total_arquivos + 1

                # ignorar arquivos que não contenham o campo 'subject'
                if email_content is None:
                    continue

                stat_arquivos_carregados = stat_arquivos_carregados + 1
                df.append([email_content, label])
                conteudo.close()

    print("Encontrado {} arquivos. Foram carregados {}.".format(stat_total_arquivos, stat_arquivos_carregados))

    return df

# Apenas mensura o tempo de execução de uma função.
# É utilizado thread_time_ns pois é desconsiderado o tempo de switch da CPU.
def measure_execution_function(func, func_param):
    timer = time.thread_time_ns()
    result = func(func_param)
    timer = time.thread_time_ns() - timer
    # A função é infelizmente chamada de novo pra dessa vez calcular seu consumo de memória
    memory = memory_usage((func, (func_param,)), interval=1)
    return {'tempo':timer, 'memoria_media': statistics.mean(memory)}, result

# Apenas pra dar uma ideia de quanto tempo irá demorar a experimentação.
def calcular_custo(prepros_list, random_states):
    custo = 0
    for pre_pros in prepros_list:
        custo = custo + 2*STAT_ROD * len(list(random_states))
        for f in pre_pros:
            # O custo da função é o dobro pois ela é chamada duas vezes.
            custo = custo + 2*CUSTOS.get(f.__name__)
    return custo
