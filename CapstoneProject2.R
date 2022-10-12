#install package if needed
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")
if(!require(Rborist)) install.packages("Rborist", repos = "http://cran.us.r-project.org")

#initializing packages needed
library(tidyverse)
library(caret)
library(data.table)
library(matrixStats)
library(Rborist)


#download the train set, fashion-mnist_train.csv then save it to train_set
dl_train <- tempfile()
download.file("http://dl.dropboxusercontent.com/s/jjj4596jw9tzpre/fashion-mnist_train.csv?dl=0", dl_train)
train_set <- read.csv(dl_train)

#download the test set, fashion-mnist_test.csv then save it to test_set
dl_test <- tempfile()
download.file("http://dl.dropboxusercontent.com/s/fjm62ndbzlbepa5/fashion-mnist_test.csv?dl=0", dl_test)
test_set <- read.csv(dl_test)

#remove some variables
rm(dl_train, dl_test)

#separation of labels and pixel values for test set
test_fashion <- test_set[-c(1)]
test_labels <- as.factor(test_set$label)

#separation of labels and pixel values for train set
train_fashion <- train_set[-c(1)]
labels <- as.factor(train_set$label)

#Plotting the first image
d <- as.matrix(train_fashion)[1,]
image(1:28, 1:28, matrix(d, 28, 28))


#boxplot for each label
avg <- rowMeans(train_fashion)
tibble(labels = labels, row_averages = avg) %>% ggplot(aes(x = labels, y = row_averages)) + geom_boxplot()

#create matrix for train set pixels
train_matrix <- as.matrix(train_fashion)



#standard deviation for pixels of different images
sds <- colSds(train_matrix)

#bar graph tallying standard deviation of columns
qplot(sds, bins = 256, color = I("black"))

#index of columns with near zero variance (low standard deviation) which will not be used much in training
lowsds <- nearZeroVar(train_matrix)

#col_index for indexes of pixels that does not have near zero variance that will be used for training
col_index <- setdiff(1:ncol(train_matrix), lowsds)

#setting the seed for the random sampling
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`

#train control for lda used repeatedcv method
control_lda  <- trainControl(method="repeatedcv",
                                number = 4,
                                repeats = 5
)

#training model for lda, also note that the data used was without the columns with near zero variance
train_lda <- train(train_matrix[,col_index], labels,
                   method = "lda",
                   trControl = control_lda)

#predict labels on the test set using lda model
y_hat_lda <- predict(train_lda,
                     test_fashion[, col_index],
                     type="raw")

#accuracy for lda model
cm <- confusionMatrix(y_hat_lda, factor(test_labels))
lda_accuracy <- cm$overall["Accuracy"]

#results of lda model
results <- data_frame(Model = "Linear Discriminant Analysis", Accuracy = lda_accuracy)
results %>% knitr::kable()


#setting the seed for the random sampling
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`

#control for knn using method cv
control_knn <- trainControl(method = "cv", number = 5, p = .9)

#training knn to get best k
train_knn <- train(train_matrix[,col_index], labels,
                   method = "knn",
                   tuneGrid = data.frame(k = c(4,5,6,7)),
                   trControl = control_knn)

#plot to check the best k (tuneGrid) for training
ggplot(train_knn)

#best k is 5
best_k <- train_knn$bestTune

#train KNN model using the best tune, also note that the data used was without the columns with near zero variance
fit_knn<- knn3(train_matrix[ ,col_index], labels,  k = best_k)

#predict labels on the test set using knn model
y_hat_knn <- predict(fit_knn,
                     test_fashion[, col_index],
                     type="class")

#accuracy for knn model
cm <- confusionMatrix(y_hat_knn, factor(test_labels))
knn_accuracy <- cm$overall["Accuracy"]

#results of the knn model
results <- bind_rows(results, data_frame(Model="K-Nearest Neighbor", Accuracy = knn_accuracy))
results %>% knitr::kable()

#setting the seed for the random sampling
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`

#control for random forest using cv
control_rf <- trainControl(method="cv", number = 5, p = 0.8)

#grid to be used for tuneGrid
grid <- expand.grid(minNode = c(1) , predFixed = c(20, 30, 40, 50, 60))

#train random forest to find best parameters for actual training
train_rf <-  train(train_matrix[ , col_index],
                   labels,
                   method = "Rborist",
                   nTree = 100,
                   trControl = control_rf,
                   tuneGrid = grid,
                   nSamp = 5000)

#plot to check the best predictors to be used
ggplot(train_rf)


#train Random Forest model using the best tune, also note that the data used was without the columns with near zero variance
fit_rf <- Rborist(train_matrix[, col_index], labels,
                  nTree = 1000,
                  minNode = train_rf$bestTune$minNode,
                  predFixed = train_rf$bestTune$predFixed)

#predict labels on the test set using random forest model
y_hat_rf <- factor(levels(labels)[predict(fit_rf, test_fashion[ ,col_index])$yPred])

#accuracy for random forest model
cm <- confusionMatrix(y_hat_rf, factor(test_labels))
rf_accuracy <- cm$overall["Accuracy"]

#results for random forest model
results <- bind_rows(results, data_frame(Model="Random Forest", Accuracy = rf_accuracy))
results %>% knitr::kable()
