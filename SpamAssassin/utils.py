# Apenas pra diminuir magic numbers no código...
import glob
import os
import time

HAM = 0
SPAM = 1

CUSTOS = {
    'remocao_de_stopwords': 47,  # esses foram os tempos médios que cada um levou na minha máquina.
    'stemming': 58,  # tempo em segundos!
    'lemmatization': 45
}
STAT_ROD = 8  # custo de uma rodada


def get_email_content(file_content):
    if 'Subject' in file_content:
        return file_content[file_content.find('Subject'):]
    else:
        return None


def create_df(directory_path):
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


def calcular_custo(prepros_list, random_states):
    custo = 0
    for pre_pros in prepros_list:
        custo = custo + STAT_ROD * len(list(random_states))
        for f in pre_pros:
            custo = custo + CUSTOS.get(f.__name__)
    return custo
