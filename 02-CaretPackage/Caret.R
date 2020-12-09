


# Usando o Pacote Caret Para Criar Modelos de Machine Learning em R




# *Doc ver Técnica de imputação, def valores missing

# Pcte possui:
# algoritmos de ML
# fçs para  otimizar o algoritmo
# fç para dividir os dados de treino e teste
# fç para avaliar os dados do modelo 
# fç estatistica para interpretar os resultados do modelo
# fç para determinar quais variaveis do dataset são as mais relevantes p treinar o modelo


#Def dir
setwd("C:/Users/paloma/Desktop/GitHub/RML/CaretPackage")
getwd()

# Instalando os pacotes
install.packages("caret") #n vem com R
install.packages("randomForest")

# Carregando os pacotes
library(caret)
library(randomForest)
library(datasets)


# Usando o dataset mtcars (vem com R)
View(mtcars)


# Funcao do Caret para divisao dos dados
?createDataPartition
split <- createDataPartition(y = mtcars$mpg, p = 0.7, list = FALSE)



# Criando dados de treino e de teste (com pacote caret)
dados_treino <- mtcars[split,]
dados_teste <- mtcars[-split,]


# Treinando o modelo
?train
names(getModelInfo()) # ver tudo que está dispovível no pcte caret




# ******************************************************************************
# Mostrando a importância das variáveis para a criação do modelo 
?varImp


modelol_v1 <- train(mpg ~ ., data = dados_treino, method = "lm")
varImp(modelol_v1)

#OBS: criamos e usamos todas as variaveis do modelo
# Algoritmo: lm 

# ******************************************************************************



# Agora apenas as 4 var mais imporatantes
# Regressão linear
modelol_v1 <- train(mpg ~ wt + hp + qsec + drat, data = dados_treino, method = "lm")




# Random forest
modelol_v2 <- train(mpg ~ wt + hp + qsec + drat, data = dados_treino, method = "rf")



# Resumo do modelo
summary(modelol_v1) # verifique o R-standart: 0.9191
summary(modelol_v2) # o summary desse modelo n apresenta o nível de precisão, é preciso calcular



# Ajustando o modelo (otimizando)
?expand.grid
?trainControl
controle1 <- trainControl(method = "cv", number = 10) 
#vc = cross validation (dividir os dados em varios pedaços [treino/teste])
#10 folders (10 combinações de dados de treino e teste , totalmente randomicos)
# é basicamente def um parametro p o modelo

modelol_v3 <- train(mpg ~ wt + hp + qsec + drat, 
                    data = dados_treino, 
                    method = "lm", 
                    trControl = controle1, 
                    metric = "Rsquared")


# Resumo do modelo
summary(modelol_v3) 
# basicamente a mesma metrica de V1 (0.9191), ou seja, esse parametro n modifica a performance do modeo p esse exemplo 

# Coletando os residuos (são os erros de previsão do modelo)
residuals <- resid(modelol_v3)
residuals

# Previsoes (pcte estatistica)
?predict
predictedValues <- predict(modelol_v1, dados_teste)
predictedValues
plot(dados_teste$mpg, predictedValues)

# Plot das variáveis mais relevantes no modelo
plot(varImp(modelol_v1))




# Fim 
