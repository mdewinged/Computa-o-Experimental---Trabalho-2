# [0] Includes
    setwd ("D:/Unb/Semestre 5.1/Computação Experimental/Trabalhos/Trabalho 2/Código/ceas08-1")
    library(naivebayes)           # install.packages("naivebayes")
    source("../pre-processing.R")
    
# [1] Functions
    clean_email = function(email){
        # Loop for finding the subject and remove the scope of the email
        aux = 0
        for (i in 1:length(email)){
            if (str_detect(email[i], "x-ceas-tracking:")){
                aux[length(aux)+1] = i
                email = email[-aux]
                break
            }
            else if(str_detect(email[i], "Subject: ")){
                str = tokenization(email[i], sep = ' ')[[1]]
                email[i] = concatenate(str[-1])
            }
            else {
                aux[length(aux)+1] = i
            }
        }
        return (email)
    }
    
    
    clean_dataset = function(paths){
        for (path in paths){
            email = try(scan(path, what = "character", sep = "\n"))
            email = clean_email(email)
            write(email, path)
            print(path)
        }
        print("Cleaning done")
    }
    
    
    load_to_mem = function(indexes, max = 10000){
        data = data.frame("index" = character(0), "email" = character(0),"label" = character(0))
        if (max > length(indexes)){
            max = length(indexes)
        }
        for (i in 1:max){
            aux = tokenization(indexes[i])[[1]]
            if(file.exists(aux[2])){
                email = scan(aux[2], what = "character", sep = "\n")
                data = rbind(data, list(aux[2], concatenate(email)[[1]], aux[1]))    
            }
        }
        names(data) = c("index", "email", "label")
        return(data)
    }
    

# [x] Execution 
    # [x.1] Cleaning dataset
    #full_immediate = scan("full-immediate/index", what = "character", sep = "\n")
    #full_pretrain_feedback = scan("trainment/index", what = "character", sep = "\n")
    
    #for (i in 1:length(full_immediate)){ full_immediate[i] = tokenization(full_immediate[i])[[1]][-1] }
    #for (i in 1:length(trainment)){ trainment[i] = tokenization(trainment[i])[[1]][-1] }
    
    #clean_dataset(full_immediate)
    #clean_dataset(trainment)
    
    
    # [x.2] Load dataset
    full_immediate = scan("full-immediate/index", what = "character", sep = "\n")
    trainment = scan("trainment/index", what = "character", sep = "\n")
    
    p_dataset = load_to_mem(full_immediate, max = 1500)
    t_dataset = load_to_mem(trainment, max = 3333)
    
    