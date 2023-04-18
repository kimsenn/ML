# Random Forest

# 데이터 불러오기
library(MASS)
# summary(Boston); str(Boston); ncol(Boston); nrow(Boston)

# train / test split
idx <- sample(1:nrow(Boston), size=nrow(Boston)*0.7, replace=F)
Boston_train <- Boston[idx,] # 354 obs (506 70%)
Boston_test <- Boston[-idx,] # 152 obs (506 30%)
dim(Boston_train); dim(Boston_test) # 354 14   # 152 14

# install.packages("randomForest")
library(randomForest)

#
set.seed(1)
rf.fit <- randomForest(medv~., data=Boston_train, mtry=6, importance=T)
plot(rf.fit)

#
importance(rf.fit)
varImpPlot(rf.fit)

#
rf.yhat <- predict(rf.fit, newdata = Boston_test)
rf_rmse <- mean((rf.yhat-Boston_test$medv)^2)
sqrt(rf_rmse) ; rf_rmse

# 총 4개 알고리즘 중에 성능(평균제곱오차)가 가장 좋은 것은 RF!
