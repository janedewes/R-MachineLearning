


#### Prevendo a Inadinplência de clientes com Machine Learning #####


# OBS: O resultado desse modelo pode ser publicado no PowerBI (online) ou Shiny.  

# def dir
setwd("")
getwd()


# Instalar os pacotes
install.packages("Amelia") # fç para tratar valores ausentes
install.packages("caret") # Modelos de ML e pré-processar os dados 
install.packages("ggplot2") # gráficos
install.packages("dplyr") # manipulação de dados 
install.packages("reshape") # modificar o formato dos dados 
install.packages("randomForest") # ML
install.packages("e1071") # ML

# Carregar os pacotes
library(Amelia)
library(ggplot2)
library(caret)
library(dplyr)
library(reshape)
library(randomForest)
library(e1071)



# Carregar os dados 
# Fonte: archive.ics.uci.edu/ml/datasets
dados_clientes <- read.csv("dataset.csv")


# Verificar dados:  (Fazer uma análise das variáveis)
head(dados_clientes)
View(dados_clientes)
str(dados_clientes)
summary(dados_clientes)
dim(dados_clientes)


################################################################################
################# Análise Exploratória, Limpeza e Transformação ################
################################################################################

# Remover a primeira col ID
dados_clientes$ID <- NULL
dim(dados_clientes)
View(dados_clientes)


# Renomenado a col classe (var target)
colnames(dados_clientes) #nomes das cols
colnames(dados_clientes)[24] <- "inadimplente"
colnames(dados_clientes)
View(dados_clientes)




# --------------------------- Important-----------------------------------------

# Verificar valores ausentes e remover do dataset
sapply(dados_clientes, function(x) sum(is.na(x)))
?missmap
#Verificar se existe valores na de maneira gráfica
missmap(dados_clientes, main = "Valores Missing Observados")
dados_clientes <- na.omit(dados_clientes)

#  -----------------------------------------------------------------------------


# Convertendo os atributos genero, escolaridade, estado civil e idade para fatores (CATEGORIAS)
colnames(dados_clientes)
colnames(dados_clientes)[2] <- "Genero"
colnames(dados_clientes)[3] <- "Escolaridade"
colnames(dados_clientes)[4] <- "Estado_Civil"
colnames(dados_clientes)[5] <- "Idade"
colnames(dados_clientes)

View(dados_clientes)


# Genero - subst 1, 2 por Masculino e Feminino ---------------------------------
View(dados_clientes$Genero)
str(dados_clientes$Genero)
summary(dados_clientes$Genero)

################################################################################
?cut # ############ fç é converter var numerica para factor!!! #################
# cut = converter os TIPOS e os VALORES das variaveis ##########################
################################################################################

dados_clientes$Genero <- cut(dados_clientes$Genero, 
                             c(0,1,2),
                             labels = c("Masculino", 
                                        "Feminino"))

View(dados_clientes$Genero)
str(dados_clientes$Genero)
summary(dados_clientes$Genero)




# Escolaridade - 1, 2,3, 4 trocar ----------------------------------------------
str(dados_clientes$Escolaridade)
summary(dados_clientes$Escolaridade)

dados_clientes$Escolaridade <- cut(dados_clientes$Escolaridade, 
                                   c(0,1,2,3,4),
                                   labels = c("Pós Graduado", 
                                              "Graduado", 
                                              "Ensino Médio", 
                                              "Outros"))
View(dados_clientes$Escolaridade)
str(dados_clientes$Escolaridade)
summary(dados_clientes$Escolaridade) # agora temos valores ausentes!




# Estado Cívil - 0, 1, 2, 3 ----------------------------------------------------
str(dados_clientes$Estado_Civil)
summary(dados_clientes$Estado_Civil)

dados_clientes$Estado_Civil <- cut(dados_clientes$Estado_Civil, 
                                   c(-1,0,1,2,3), 
                                   labels = c("Desconhecido", 
                                              "Casado", 
                                              "Solteiro", 
                                              "Outro"))
View(dados_clientes$Estado_Civil)
str(dados_clientes$Estado_Civil)
summary(dados_clientes$Estado_Civil)



# Idade - com faixa etária -----------------------------------------------------
str(dados_clientes$Idade)
summary(dados_clientes$Idade)
hist(dados_clientes$Idade)

dados_clientes$Idade <- cut(dados_clientes$Idade, 
                            c(0, 30,50,100), 
                            labels = c("Jovem", 
                                       "Adulto", 
                                       "Idoso"))

View(dados_clientes$Idade)
str(dados_clientes$Idade)
summary(dados_clientes$Idade)



################################################################################
# as.factor => converter para fator, sem alterar o VALOR, só o TIPO! ###########
################################################################################
?as.factor

# Converter a variável que indica pagamentos para o tipo fator
str(dados_clientes)
dados_clientes$PAY_0 <- as.factor(dados_clientes$PAY_0)
dados_clientes$PAY_2 <- as.factor(dados_clientes$PAY_2)
dados_clientes$PAY_3 <- as.factor(dados_clientes$PAY_3)
dados_clientes$PAY_4 <- as.factor(dados_clientes$PAY_4)
dados_clientes$PAY_5 <- as.factor(dados_clientes$PAY_5)
dados_clientes$PAY_6 <- as.factor(dados_clientes$PAY_6)

# Dataset após as conversões
str(dados_clientes)
sapply(dados_clientes, function(x) sum(is.na(x))) # verificar valores missing
missmap(dados_clientes, main = "Valores Missing Observados") # verificar valores missing no mapa

dados_clientes <- na.omit(dados_clientes) # remover os valores missing!
missmap(dados_clientes, main = "valores Missing Observados")
dim(dados_clientes)
View(dados_clientes)




# Alterando a variável target (inadimplente) para o tipo fator -----------------
str(dados_clientes$inadimplente)
colnames(dados_clientes)

dados_clientes$inadimplente <- as.factor(dados_clientes$inadimplente) # só converter O TIPO, n alterar o VALOR
str(dados_clientes$inadimplente)
summary(dados_clientes$inadimplente)



# Total de inadinplentes x não inadinplentes
?table #tb podemos usar a fç table para visualizar a proporçao dos dados em cada categoria
table(dados_clientes$inadimplente)
# Proporções (%)
prop.table(table(dados_clientes$inadimplente)) # OBS: essa proporção não é ideal para o modelo!!!!



# Plot da distribuição (ggplot2)
qplot(inadimplente, data = dados_clientes, goem = "bar")+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


# Set seed
set.seed(12345)



################################################################################



# Amostragem estratificada------------------------------------------------------

# Seleciona as linhas de acordo com a variável inadimplente como strata
?createDataPartition # split dos dados
# p = % dos dados que vão para treinamento
# list = FALSE -> resultado será uma matriz

indice <- createDataPartition(dados_clientes$inadimplente, p = 0.75, list = FALSE)
dim(indice) # matriz de apenas uma col




# Definimos os dados de treinamento como subconjunto do conjunto de dados original 
# com números de indice de linha (conforme identificado acima) e todas as cols
dados_treino <- dados_clientes[indice,]
dim(dados_treino)
table(dados_treino$inadimplente)


# Ver as % entre as classes
prop.table(table(dados_treino$inadimplente))


# Número de registros no dataset de treino 
dim(dados_treino)



# ------------------------------------------------------------------------------
# Construir de maneira visual o entendimento sobre a proporção de classes 
# Comparação entre as % entre as classes de treinamento e dados originais
compara_dados <- cbind(prop.table(table(dados_treino$inadimplente)),
                       prop.table(table(dados_clientes$inadimplente)))

colnames(compara_dados) <- c("Treinamento", "Original")
compara_dados



# Plot da comparação acima  ---------------- Só p visualizar o result ----------

# Melt Data - Converte colunas em linhas
?reshape2::melt
melt_compara_dados <- melt(compara_dados)
melt_compara_dados



# Plot 
ggplot(melt_compara_dados, aes(x=X1, y=value)) +
  geom_bar(aes(fill = X2), stat = "identity", position = "dodge") +
  theme(axis.text = element_text(angle = 90, hjust = 1))
# Vermelho = original 
# verde = treinamento



# Tudo o que não está no dataset de treinamento está no dataset de teste. Observe o sinal (-)
dados_teste <- dados_clientes[-indice,] # tudo (linhas e cols) que não estão no indice (todas)
dim(dados_teste) 
dim(dados_treino)





################################################################################
############################## Machine Learning ################################
################################################################################


 
############################## 1º Versão do Modelo #############################
?randomForest
modelo_v1 <- randomForest(inadimplente ~ ., data = dados_treino)
modelo_v1
# inadimplente = var target
# ponto (.) representa todas as var preditoras


# Avaliação do modelo
plot(modelo_v1)


# Previsões com dados de teste -------------------------------------------------
previsoes_v1 <- predict(modelo_v1, dados_teste)



# Confusion Matrix
?caret::confusionMatrix
cm_v1 <- caret::confusionMatrix(previsoes_v1, dados_teste$inadimplente, positive = "1")
cm_v1





# MÉTRICAS de avaliação do modelo preditivo ------------------------------------

# Calculando Precision, Recall e F1-Score, 
y <- dados_teste$inadimplente
y_pred_v1 <- previsoes_v1



# Precision
precision <- posPredValue(y_pred_v1, y)
precision

# Recall
recall <- sensitivity(y_pred_v1, y)
recall

# F1-Score
F1 <- (2 * precision * recall) / (precision + recall)
F1






# Balanceamento de classe ----------------------Dados não estão balanceados ----
install.packages("DMwR")
library(DMwR) 
?SMOTE #Essa fç resolve probelmas de classes não balanceadas!



# Aplicando o SMOTE - SMOTE: Synthetic Minority Over-sampling Technique --------

# Over-sampling: criar registros sintéticos para preencher a classe 1 (nesse ex)
# Under-sampling: Ou, reduzir os registros da classe 0. 

# https://arxiv.org/pdf/1106.1813.pdf

table(dados_treino$inadimplente) # número de registros
prop.table(table(dados_treino$inadimplente)) # ou a %

set.seed(9560)

dados_treino_bal <- SMOTE(inadimplente ~ ., data  = dados_treino) # técnica de Over-sampling                       
table(dados_treino_bal$inadimplente)
prop.table(table(dados_treino_bal$inadimplente))





############################### 2ª versão do modelo ############################

# Na 2ª versão do modelo, usaremos os dados balanceados!
# O balanceamento só é feito nos dados de treino! 
# Nos  dados de teste não há balanceamento!
modelo_v2 <- randomForest(inadimplente ~ ., data = dados_treino_bal)
modelo_v2




# Avaliando o modelo
plot(modelo_v2)



# Previsões com dados de teste
previsoes_v2 <- predict(modelo_v2, dados_teste)


# Confusion Matrix
?caret::confusionMatrix
cm_v2 <- caret::confusionMatrix(previsoes_v2, dados_teste$inadimplente, positive = "1")
cm_v2 #  Acurácia é menor, porem esse modelo possui um equilíbrio maior!




# MÉTRICAS de avaliação do modelo preditivo ------------------------------------
# Calculando Precision, Recall e F1-Score
y <- dados_teste$inadimplente
y_pred_v2 <- previsoes_v2


# Precision
precision <- posPredValue(y_pred_v2, y)
precision


# Recall 
recall <- sensitivity(y_pred_v2, y)
recall


# F1-Score
F1 <- (2 * precision * recall) / (precision + recall)
F1



# OBS: Note que as métricas estão equilibradas, o modelo acerta mais tanto para classe 0, quanto 
# para classe 1.






################# Listar as VARIÁVEIS mais IMPORTANTES -------------------------


# Importância das variáveis preditoras para as previsões
View(dados_treino_bal)
varImpPlot(modelo_v2) # fç que lista as vars mais importantes



# Obtendo as variáveis mais importantes ----------------------------------------
imp_var <- importance(modelo_v2) # fç importance

varImportance <- data.frame(Variables = row.names(imp_var), 
                            Importance = round(imp_var[ ,'MeanDecreaseGini'],2))



# Criando o rank de variáveis baseado na importância ---------------------------
rankImportance <- varImportance %>% 
  mutate(Rank = paste0('#', dense_rank(desc(Importance))))



# Usando ggplot2 para visualizar a importância relativa das variáveis ----------
ggplot(rankImportance, 
       aes(x = reorder(Variables, Importance), 
           y = Importance, 
           fill = Importance)) + 
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank), 
            hjust = 0, 
            vjust = 0.55, 
            size = 4, 
            colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() 






############################### 3ª versão do modelo ############################

# Construindo a terceira versão do modelo apenas com as variáveis mais importantes
colnames(dados_treino_bal)

modelo_v3 <- randomForest(inadimplente ~ PAY_0 + PAY_2 + PAY_3 + PAY_AMT1 + PAY_AMT2 + PAY_5 + BILL_AMT1, 
                          data = dados_treino_bal)

# OBS:
# + é concatenação das variáveis
# (.) todas as varriáveis
modelo_v3



# Avaliando o modelo
plot(modelo_v3)


# Previsões com dados de teste
previsoes_v3 <- predict(modelo_v3, dados_teste)


# Confusion Matrix
?caret::confusionMatrix
cm_v3 <- caret::confusionMatrix(previsoes_v3, dados_teste$inadimplente, positive = "1")
cm_v3



# MÉTRICAS de avaliação do modelo preditivo ------------------------------------
# Calculando Precision, Recall e F1-Score
y <- dados_teste$inadimplente
y_pred_v3 <- previsoes_v3


# Precision
precision <- posPredValue(y_pred_v3, y)
precision

# Recall 
recall <- sensitivity(y_pred_v3, y)
recall

# F1
F1 <- (2 * precision * recall) / (precision + recall)
F1







# 1º Versão do modelo: 
# 2º Versão do modelo: Com dados balanceados 
# 1º Versão do modelo: Com as variáveis mais importantes




################################################################################
######################## Salvando o modelo em disco ############################
saveRDS(modelo_v3, file = "modelo_v3.rds")

# Carregando o modelo
modelo_final <- readRDS("modelo_v3.rds")






# Esse modelo pode ser usado no PowerBI, Shiny (plataforma de dashboard em R) 
# Salvar em .csv, etc. 


################################################################################
############### Previsões com novos dados de 3 clientes ########################
# Prever a inadimplencia de clientes


# Dados dos 3 clientes -----------------------------------------------------------
# Devemos usar o número de dados iguais o numero de variáveis usadas para treinar o modelo
# Foram usadas 7 variáveis preditoras:
PAY_0 <- c(0, 0, 0) #1
PAY_2 <- c(0, 0, 0) #2
PAY_3 <- c(1, 0, 0) #3
PAY_AMT1 <- c(1100, 1000, 1200) #4
PAY_AMT2 <- c(1500, 1300, 1150) #5
PAY_5 <- c(0, 0, 0) #6
BILL_AMT1 <- c(350, 420, 280) #7

# OBS: Note que os dados não são do tipo fator!!!


# Concatena em um dataframe ----------------------------------------------------
novos_clientes <- data.frame(PAY_0, PAY_2, PAY_3, PAY_AMT1, PAY_AMT2, PAY_5, BILL_AMT1)
View(novos_clientes)



# Previsões --------------------------------------------------------------------
previsoes_novos_clientes <- predict(modelo_final, novos_clientes)


# Checando os tipos de dados (VERIFICAR SE OS NOVOS DADOS DE ENTRADA ESTÃO NO MESMO FORMATO DOS DADOS DE TREINO)
str(dados_treino_bal)
str(novos_clientes)


# Convertendo os tipos de dados-------------------------------------------------
novos_clientes$PAY_0 <- factor(novos_clientes$PAY_0, levels = levels(dados_treino_bal$PAY_0))
novos_clientes$PAY_2 <- factor(novos_clientes$PAY_2, levels = levels(dados_treino_bal$PAY_2))
novos_clientes$PAY_3 <- factor(novos_clientes$PAY_3, levels = levels(dados_treino_bal$PAY_3))
novos_clientes$PAY_5 <- factor(novos_clientes$PAY_5, levels = levels(dados_treino_bal$PAY_5))
# Ou seja, converte a var para o tipo fator e usa como nível o mesmo que temos em treino!!
str(novos_clientes)

# Previsões
previsoes_novos_clientes <- predict(modelo_final, novos_clientes)
View(previsoes_novos_clientes)






################################################################################
# Resultado do modelo: 
# Temos 1 cliente que ficará inadimplente (1)
# E dois clientes que não ficarão inadimplentes (0)
################################################################################



# Fim













