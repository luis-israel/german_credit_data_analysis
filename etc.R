# código descartado no relatório

# glm

glm_grid <- expand.grid(
  alpha = 1,
  lambda = seq(0.001, 0.1, by = 0.001)
)
glm_model <- train(
  x = train_x,
  y = train_y,
  method = "glmnet",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = glm_grid
)
glm_model$bestTune

# svm radial

svm_radial_grid <- expand.grid(sigma = 0.015, C = 1.5)
svm_radial_model <- train(
  x = train_x,
  y = train_y,
  method = "svmRadial",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = svm_radial_grid,
  preProcess = c("center", "scale")
)
svm_radial_model

# TODO svm linear

svm_linear_model <- train(
  x = train_x,
  y = train_y,
  method = "svmLinear", 
  metric = "ROC",
  trControl = ctrl,
  tuneLength = 10,
  preProcess = c("center", "scale")
)
svm_linear_model

# lda

lda_model <- train(
  x = train_x,
  y = train_y,
  method = "lda",
  metric = "ROC",
  trControl = ctrl,
  tuneLength = 10
)
lda_model

# qda

qda_model <- train(
  x = train_x,
  y = train_y,
  method = "qda",
  metric = "ROC",
  trControl = ctrl,
  tuneLength = 10
)
qda_model

# TODO nadaraya
# TODO weighted knn

# naive bayes

grid <- data.frame(fL = c(0, 0.5, 1.0), usekernel = TRUE, adjust = c(0, 0.5, 1.0))
nb_model <- train(
  x = train_x,
  y = train_y,
  method = "nb",
  metric = "ROC",
  trControl = ctrl,
  tuneLength = 10
)
nb_model

# knn

knn_model <- train(
  x = train_x,
  y = train_y,
  method = "knn",
  metric = "Accuracy",
  trControl = ctrl,
  tuneLength = 20,
  preProcess = c("center", "scale")
)
knn_model
knn_predicted <- predict(knn_model, test_x) # k = 13
confusionMatrix(knn_predicted, test_y)

# stack

list_control <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5,
  search = "grid",
  summaryFunction = twoClassSummary,
  classProbs = TRUE,
  savePredictions = "final",
  index = createResample(train_y, 10),
  allowParallel = TRUE,
  verboseIter = TRUE
)
model_list <- caretList(
  x = train_x,
  y = train_y,
  trControl = list_control,
  metric = "ROC",
  methodList = c("knn", "rf"),
  tuneList = list(
    knn = caretModelSpec(method = "knn", tuneGrid = expand.grid(k = 13), preProcess = c("center", "scale")),
    rf = caretModelSpec(method = "rf", tuneGrid = expand.grid(mtry = 2), ntree = 1000, importance = TRUE)
  )
)
summary(resamples(model_list))
xyplot(resamples(model_list))

stackControl <- trainControl(method = "repeatedcv", number = 10, repeats = 5, savePredictions = TRUE, classProbs = TRUE, verboseIter = TRUE)
stack <- caretStack(model_list, method = "glm", metric = "ROC", trControl = stackControl)

xgb_grid <- expand.grid(nrounds = c(100, 500, 1000),
                        max_depth = c(2, 6, 10),
                        eta = c(0.001, 0.01, 0.1),
                        gamma = c(0, 1, 3),
                        colsample_bytree = c(0.5, 0.75, 1),
                        min_child_weight = 1,
                        subsample = c(0.5, 0.75, 1))
xgb_model <- train(
  x = train_x,
  y = train_y,
  method = "xgbTree",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = xgb_grid,
  nthread = 4
)

