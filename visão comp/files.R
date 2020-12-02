# [0] Includes
  library(ngram  )        # install.packages("ngram"  )
  library(jsonlite)         # install.packages("jsonlite")


# [1] Load/Save data
  # Brief: Dado um path, abre um arquivo do tipo .csv ou .json
  # Param:
  #   path = Caminho do arquivo a ser aberto
  #   separador = separador das células do arquivo tipo .csv (esse parâmetro serve apenas para o .csv)
  #   tipo = O formato do arquivo a ser lido da memória
  openfile = function (path, separador = ';', tipo = ".csv")  
  {
    if(tipo == ".csv")
    {
      text <- as.data.frame(read.csv(concatenate(path, tipo, rm.space = TRUE), header = TRUE, sep = separador, encoding = "latin1", stringsAsFactors = F))
    }
    else if (tipo == ".json")
    {
      text <- as.data.frame(fromJSON(concatenate(path, tipo, rm.space = TRUE)))
    }
    return(text)
  }


  # Brief: Salva na memória a estrutura de dados em .csv e .json
  # Param:
  #   datagrame = estrutura da dados e ser armazenada
  #   path = Destino do arquivo a ser salvo
  write_json_csv_txt = function (dataframe, path)
  {
    jsonlite::write_json(dataframe, concatenate(path, ".json", rm.space = TRUE))
    write.csv2(dataframe, concatenate(path, ".csv", rm.space = TRUE), sep = ';')
  }