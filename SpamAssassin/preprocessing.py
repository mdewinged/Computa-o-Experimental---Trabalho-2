### Módulos de pré-processamento
from nltk import word_tokenize, LancasterStemmer, WordNetLemmatizer


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
