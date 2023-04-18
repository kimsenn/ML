# 인공신경망 분석 ANN

# 데이터 불러오기
library(MASS)
# summary(Boston); str(Boston); ncol(Boston); nrow(Boston)

# train / test split
idx <- sample(1:nrow(Boston), size=nrow(Boston)*0.7, replace=F)
Boston_train <- Boston[idx,] # 354 obs (506 70%)
Boston_test <- Boston[-idx,] # 152 obs (506 30%)
dim(Boston_train); dim(Boston_test) # 354 14   # 152 14

# 정규화 함수
normalize <- function(x){return ((x-min(x)/max(x)-min(x)))}

# Boston 데이터 정규화
Boston_train_norm <- as.data.frame(sapply(Boston_train, normalize))
Boston_test_norm <- as.data.frame(sapply(Boston_test, normalize))

# 신경망 모델 (nnet)
library(nnet)
nnet.fit <- nnet(medv~., data=Boston_train_norm, size=5)
nnet.yhat <- predict(nnet.fit, newdata = Boston_test_norm, type="raw")

nnet_rmse <- mean((nnet.yhat-Boston_test_norm$medv)^2)
sqrt(nnet_rmse); nnet_rmse

# 신경망 모델 (neuralnet)
# install.packages("neuralnet")
library(neuralnet)
neural.fit <- neuralnet(medv~.,  data = Boston_train_norm, hidden = 5)
neural.results <- compute(neural.fit, Boston_test_norm[1:13])
neural.yhat <- neural.results$net.result

neural_rmse <- mean((neural.yhat-Boston_test_norm$medv)^2)
sqrt(neural_rmse); neural_rmse


# 시각화
plot(neural.fit)
