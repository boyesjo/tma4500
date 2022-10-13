library(MASS)
library(dplyr)
library(ISLR)

data(iris)
head(iris)
mod <- lda(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data = iris)

# get accuracy
acc <- mean(predict(mod)$class == iris$Species)
acc

# use cv to get accuracy
library(caret)
set.seed(123)
cv <- train(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data = iris, method = "lda", trControl = trainControl(method = "cv", number = 10))
