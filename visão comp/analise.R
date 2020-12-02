# [0] Includes
    setwd ("D:/Unb/Semestre 5.1/Visão Computacional/Trabalhos/Trabalho 2/Dados")    
    library(qdap)           # install.packages("qdap")
    library(ggplot2)        # install.packages("ggplot2")
    library(ngram  )        # install.packages("ngram"  )
    library(stringr)      
    library(dplyr  )        # install.packages("dplyr"  )
    library(tm)
    library(wordcloud)
    source("files.R")
    

# [1] Functions
    `%nin%`= Negate(`%in%`)
    
    # Brief: Calcula o intervalo de confiança  de um vetor
    # Param:
    #     x    = vetor de valores a ser analisado
    #     conf = p-valor 
    ic.m <- function(x, conf = 0.95){
      n <- length(x)
      media <- mean(x)
      variancia <- var(x)
      quantis <- qt(c((1-conf)/2, 1 - (1-conf)/2), df = n-1)
      ic <- media + quantis * sqrt(variancia/n)
      return(ic)
    }
    
    # Brief: Realiza o teste ANOVA para cada parametro
    # Param:
    #     y = contem o vetor com todos os dados das cartas
    #     x = contem o label para cada carta segundo algum criterio de divisao (por sexo, idade, etc.)
    anova_all_param = function (y, x)
    {
      modelos = summary(aov(y ~ x))   
      return(modelos)
    }
    
    # Brief: Verifica se ha diferenças significativas entre as medias de todos os parametros
    # Param:
    #     x1 = vetor da primeira classe
    #     x2 = vetor da segunda classe
    t_test_all_param = function (x1, x2)
    {
      classes = names(cartas)
      classes = removeWords(classes, c('id', 'text', 'idade', 'sexo', 'metodo', 'meio', 'registro', 'horario', 'id_pessoa', 'dia_semana', 'endereco_fato', 'tipo_local', 'tipo_local_agrupado', 'naturalidade', 'endereco_residencial', 'end_fato_residencial_mesmo', 'cor_pele', 'altura', 'peso', 'IMC', 'IMC_agrupado', 'motivo', 'alcool', 'droga', "carta_direcionada_a", 'instrucoes_especificas', 'afetividade', 'dificuldades_mencionadas'))
      classes = classes[classes != '']
      for (i in 1:length(classes))
      {
          print(classes[i])
          print(t.test(x1[classes[i]], x2[classes[i]]))
      }
    }
    
    # Brief: Verifica se ha diferenças significativas entre as medias de todos os parametros
    # Param:
    #     x1 = vetor da primeira classe
    #     x2 = vetor da segunda classe
    read_data = function()
    {
      data = list()

      for (k in kernel){
        for (n in num_filters){
          name = paste(k, "_", n, sep ="")
          data[name] = list(openfile(name, tipo = ".json")) 
        }
      }
      return (data)
    }
    
# [2] Execution
    kernel = c(2,4,8,16,32)
    num_filters = c(2,8,32,64,128)
    
    data = read_data()
    
    resumo = list()
    for (i in c(1:length(data))){
      len = length(data[[i]][['val_accuracy']])
      resumo[[names(data[i])]] = c(data[[i]][['val_accuracy']][len], data[[i]][['val_mean_absolute_error']][len])
    }
    
    
    # Análise de regressão múltipla
    regr = data.frame(row.names = c('kz','nf','pr'))
    for (i in c(1:length(kernel))){
      for (j in c(1:length(num_filters))){
        print((j+5*(i-1)))
        regr = rbind(regr,data.frame("kz" = kernel[i], "nf" = num_filters[j], "pr" = data[[j+5*(i-1)]][['val_accuracy']][100]))
      }
    }
  
    
    # Verifica qual que é a relevância de cada variável na precisão
    model = lm(regr$pr~kz+nf, regr)
    summary(model)
    plot(model)
    
    # Verifica qual que é a influencia entre cada variável na precisão
    model = lm(regr$pr~kz+nf+kz:nf, regr)
    summary(model)
    plot(model)
    
    # Analise de variância por número de filtros e tamanho do filtro
    y = integer(0)
    x = integer(0)
    file_name = names(data)
    for (i in c(1:length(data))){
      aux =  data[[i]][['val_accuracy']]
      y = c(y, aux)
      x = c(x, rep(file_name[i], length(aux)))
    }
    
    anova_all_param(y,x)
  
    TukeyHSD(aov(y~x))    
    # Analise de variância por número de filtros e tamanho do filtro
    
    
    # Análise de variância por Divisão por tamanho do filtro
    x = integer(0)
    file_name = names(data)
    for (i in c(1:length(data))){
      if (i %% 5 == 0)
        x = c(x, rep(toString(kernel[i/5]), 5*length(aux)))
    }
    
    anova_all_param(y,x)
    TukeyHSD(aov(y~x))    
    # Análise de variância por Divisão por tamanho do filtro
    
    
    # Análise de variância por Divisão por número de filtros
    x = integer(0)
    for (i in c(1:length(data))){
      if (i %% 5 == 0)
        x = c(x, rep('128', length(aux)))
      if (i %% 5 == 1)
        x = c(x, rep('2', length(aux)))
      if (i %% 5 == 2)
        x = c(x, rep('8', length(aux)))
      if (i %% 5 == 3)
        x = c(x, rep('32', length(aux)))
      if (i %% 5 == 4)
        x = c(x, rep('64', length(aux)))
    }
    
    anova_all_param(y,x)
    TukeyHSD(aov(y~x))    
    # Análise de variância por Divisãopor número de filtros
    
    
    # Analise de ROC por kernel_size
    roc = openfile("roc_result", tipo = '.json')
    x_roc = integer(0)
    y_roc = integer(0)
    for (i in c(1:length(roc[[1]]))){
        y_roc = c(y_roc, as.numeric(roc[[2]][i]))
        if (i%%5 == 0){
          x_roc = c(x_roc, rep(toString(i/5), 5))
        }
    }
    
    anova_all_param(y_roc,x_roc)
    TukeyHSD(aov(y_roc~x_roc))
    
    
    # Analise de ROC por num_filter
    x_roc = integer(0)
    for (i in c(1:length(data))){
      if (i %% 5 == 0)
        x_roc = c(x_roc, '128')
      if (i %% 5 == 1)
        x_roc = c(x_roc, '2')
      if (i %% 5 == 2)
        x_roc = c(x_roc, '8')
      if (i %% 5 == 3)
        x_roc = c(x_roc, '32')
      if (i %% 5 == 4)
        x_roc = c(x_roc, '64')
    }
    
    anova_all_param(y_roc,x_roc)
    TukeyHSD(aov(y_roc~x_roc))    
    
    
    
      