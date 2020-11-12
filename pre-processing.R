# [0] Includes
    library(tm )            # install.packages("tm" )
    library(fpc)            # install.packages("fpc")
    library(qdap)           # install.packages("qdap")
    library(rJava  )        # install.packages("rJava"  )
    library(ngram  )        # install.packages("ngram"  )
    library(dplyr  )        # install.packages("dplyr"  )
    library(cluster)        # install.packages("cluster")
    library(stringr)
    library(openNLP)        # install.packages("openNLP") - Verificar se a versao do Java esta atualizada
    library(ggplot2)        # install.packages("ggplot2")
    library(tidytext)       # install.packages("tidytext")
    library(wordcloud   )   # install.packages("WordCloud"   )
    library(topicmodels )   # install.packages("topicmodels" )
    library(SnowballC)
    
# [1] Pre-processamento de texto
    # Brief: Separa em palavras segundo um caractere
    # Param:
    #   text = Texto a ser tokenizado
    #   sep = Caractere utilizado para separar palavras
    # Retorno: Texto tokenizado
    tokenization = function (text, sep = " ")
    {
        text <- strsplit(text, sep)
        return(text)
    }
    
    
    # Brief: Realiza um pré-processamento de texto de lowercase, stopwords, numeros, pontuacoes de um vetor de caracteres, stemming e tokenization
    # Param:
    #   text = Texto a ser pré-processado
    #   rm_upper = Booleano indicando se devem ser removidas letras maiúsculas
    #   rm_punctuation = Booleano que indica se as pontuaçoes devem ser removidas
    #   rm_numbers = Booleano que indica se os números devem ser removios
    #   rm_stopwords = Booleano que indica se as stopwords devem ser removidas
    #   stem = Booleano que indica se realizará stemming
    #   rm_otherwords = Lista de palavras que devem ser removidas
    # Retorno: Texto pré-processado
    text_preprocessing = function (text, rm_upper = FALSE, unique1 = FALSE,  o = FALSE, rm_numbers = FALSE, rm_stopwords= FALSE, stem = FALSE, rm_otherwords = NULL, tokenization = TRUE, rm_punctuation = FALSE)
    {
        if(rm_numbers) text = removeNumbers(text)
        text = rm_stopwords(text, unlist = TRUE, unique = unique1, stopwords = c(stopwords("en"), rm_otherwords), ignore.case = rm_upper, strip = rm_punctuation)
        if(stem      ) for (i in 1:length(text)) text[i] = SnowballC::wordStem(text[i])
        if(!tokenization){
            text = concatenate(text)[[1]]  
            text = stripWhitespace(text)
        } 
        return(text)
    }
    