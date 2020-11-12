# [0] Includes
    setwd ("D:/Unb/Semestre 5.1/Computação Experimental/Trabalhos/Trabalho 2/Código/ceas08-1")
    library(tm)
    library(sys)
    library(pracma)
    library(gofastR)
    library(naivebayes)           # install.packages("naivebayes") https://cran.r-project.org/web/packages/naivebayes/naivebayes.pdf
    library(data.table)
    source("../pre-processing.R")
    source("../dataset.R")
    
# [1] Functions
    generate_table = function(vector, rm_stopwords = FALSE, rm_upper = FALSE, stem = FALSE, rm_punctuation = FALSE){
        corpus = VCorpus(VectorSource(vector)) 
        dtm    = DocumentTermMatrix(corpus, control = list(tolower = rm_upper, removeNumbers = FALSE, removePunctuation = rm_punctuation, stemming = stem))
        if(rm_stopwords) dtm = remove_stopwords(dtm, stopwords = stopwords("english"))
        freqWords = findFreqTerms(dtm,5)
        freq = dtm[,freqWords]

        convert_counts <- function(x) {
            x <- ifelse(x > 0, "Yes", "No")
        }
        
        return(apply(freq, MARGIN = 2, convert_counts))
    }
    
    train_bayes = function (data, dtm){
        return(naive_bayes(dtm, data$label))
    }
    
    # Brief: Realiza a análise dado uma configuração específica
    # Param:    
    #   index = Conjunto de labels e paths dos arquivos de email do dataset de spam
    #   model = Trained Naive Bayes model
    #   size = quantidade de emails a serem processados
    # Retorno: Texto tokenizado
    total_time = 0
    processamento = function(data, model, dtm){
        start_time = Sys.time()
        result = tolower(predict(model, dtm))
        total_time <<- Sys.time() - start_time
        return (result)
    }
    #http://www.dbenson.co.uk/Rparts/subpages/spamR/
        
    
# [x] Execution 
    rm_punctuation = c(FALSE, TRUE)
    rm_stopwords   = c(FALSE, TRUE)
    rm_upper = c(FALSE, TRUE)
    stemming = c(FALSE, TRUE)
    
    predicted_labels = list()
    total_time_list  = list()
    labels = list()
    for (rm_stpw in rm_stopwords){
        for (rm_pct in rm_punctuation){
            for(rm_up in rm_upper){
                for(stem in stemming){
                    dtm = generate_table(t_dataset$email, rm_stopwords = TRUE, rm_upper = TRUE, stem = TRUE, rm_punctuation = TRUE)
                    model = train_bayes(t_dataset, dtm)
                    dtm = generate_table(p_dataset$email, rm_stopwords = TRUE, rm_upper = TRUE, stem = TRUE, rm_punctuation = TRUE)
                    experimento = processamento(p_dataset, model, dtm)
                    predicted_labels = append(predicted_labels, list(experimento))
                    total_time_list  = append(total_time_list, list(total_time))
                    name = ""
                    if(rm_stpw == FALSE) name = name + "_sw"
                    if(rm_pct  == FALSE) name = name + "_pct"
                    if(rm_up   == FALSE) name = name + "_upper"
                    if(stem    == TRUE) name = name + "_stem"
                    labels = append(labels, name)
                }
            }
        }
    }
        
    saveRDS(predicted_labels, "predictedlabels.rds")
    saveRDS(total_time_list, "total_time_list.rds")
    saveRDS(labels, "labels.rds")
    
    