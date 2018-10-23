library(caret)
library(caretEnsemble)
library(data.table)
library(ggplot2)
library(ggcorrplot)
library(rattle)
library(pROC)

setwd("/home/luos/Documents/german_credit_data_analysis")
german_credit <- fread("german_credit.csv")

n <- dim(german_credit)[1]
m <- dim(german_credit)[2]

dim(german_credit[complete.cases(german_credit)])[1] == n

int_to_factor <- c("Creditability", "Sex & Marital Status", "Telephone", "Foreign Worker", "Purpose")
german_credit[, (int_to_factor) := lapply(.SD, as.factor), .SDcols = int_to_factor]

german_credit[, `Duration of Credit (month) / Age (years)` := `Duration of Credit (month)` / `Age (years)`]
german_credit[, `Credit Amount / Duration of Credit (month)` := `Credit Amount` / `Duration of Credit (month)`] # quanto paga por mês

set.seed(06061998)
shuffle <- sample(n)
split <- 0.8 * n
train_data <- german_credit[shuffle[1:split], ]
test_data <- german_credit[-shuffle[1:split], ]
train_y <- train_data[, Creditability]
train_x <- train_data[, Creditability := NULL]
test_y <- test_data[, Creditability]
test_x <- test_data[, Creditability := NULL]
levels(train_y) <- make.names(levels(factor(train_y)))
levels(test_y) <- make.names(levels(factor(test_y)))

ctrl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3, # TODO era 5 antes, voltar?
  summaryFunction = twoClassSummary,
  classProbs = TRUE,
  savePredictions = "final",
  allowParallel = TRUE,
  verboseIter = TRUE
)

tree_model <- train(
  x = train_x,
  y = train_y,
  method = "rpart",
  metric = "ROC",
  trControl = ctrl,
  tuneLength = 5
) 
tree_model
fancyRpartPlot(tree_model$finalModel)

rf_model <- train(
  x = train_x,
  y = train_y,
  method = "rf",
  ntree = 1000,
  metric = "ROC",
  importance = TRUE,
  trControl = ctrl,
  tuneLength = 5
)
rf_model
rf_model$bestTune
rf_importance <- varImp(rf_model, scale = FALSE)
ggplot(rf_importance)
ggplot(rf_model)

rf_prob <- predict(rf_model, test_x, type = "prob")
roc_curve <- roc(test_y, rf_prob[, 2])
roc_curve$auc
str(roc_curve)
rf_predicted <- predict(rf_model, test_x)
confusion <- confusionMatrix(rf_predicted, test_y, positive = "X1") # TODO mudar positive CLASS URGENTE
confusion
