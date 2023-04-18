# 의사결정나무 - 수치예측

# 데이터 불러오기
library(MASS)
# summary(Boston); str(Boston); ncol(Boston); nrow(Boston)

# train / test split
idx <- sample(1:nrow(Boston), size=nrow(Boston)*0.7, replace=F)
Boston_train <- Boston[idx,] # 354 obs (506 70%)
Boston_test <- Boston[-idx,] # 152 obs (506 30%)
dim(Boston_train); dim(Boston_test) # 354 14   # 152 14

# 의사결정트리 모델 학습
# tree 패키지
# install.packages("tree")
library(tree)
tree.fit <- tree(medv~., data = Boston_train)
summary(tree.fit)

# 시각화
plot(tree.fit)
text(tree.fit, pretty = 0)

# rmse
tree.yhat <- predict(tree.fit, newdata = Boston_test)
rmse <- mean((tree.yhat-Boston_test$medv)^2)
sqrt(rmse); rmse

# rpart 패키지 - 변수 중요도 쉽게 확인!
library(rpart)
rpart.fit <- rpart(medv~., data=Boston_train)
summary(rpart.fit) # importance check

# rpart.plot vis~
# install.packages("rpart.plot")
library(rpart.plot)
rpart.plot(rpart.fit, digits = 3, type=0, extra=1, fallen.leaves = F, cex=1)

# rpart rmse
rpart.yhat <- predict(rpart.fit, newdata = Boston_test)
rpart_rmse <- mean((rpart.yhat-Boston_test$medv)^2)
sqrt(rpart_rmse); rpart_rmse
