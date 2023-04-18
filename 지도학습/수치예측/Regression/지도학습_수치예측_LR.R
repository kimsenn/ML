# 다중회귀분석

# 데이터 불러오기
library(MASS)
# summary(Boston); str(Boston); ncol(Boston); nrow(Boston)

# train / test split
idx <- sample(1:nrow(Boston), size=nrow(Boston)*0.7, replace=F)
Boston_train <- Boston[idx,] # 354 obs (506 70%)
Boston_test <- Boston[-idx,] # 152 obs (506 30%)
dim(Boston_train); dim(Boston_test) # 354 14   # 152 14

# 다중회귀분석 모델 학습
lm.fit <- lm(medv~., data=Boston_train) # label Y : medv
summary(lm.fit)
# F-statistic: 77.44, Adjusted R-square: 0.7379
# 변수 age, indus는 유의하지 않으므로 변수 다시 선택해서 적용

# 변수 선택법 step 함수
lm.fit2 <- step(lm.fit, method="both")
summary(lm.fit2)
# F-statistic: 91.57, Adjusted R-square: 0.7384
# lm.fit2 : age, indus제외하고, 모든 변수가 유의하게 작용

# test 예측
lm.yhat2 <- predict(lm.fit2, newdata = Boston_test)
rmse <- mean((lm.yhat2-Boston_test$medv)^2)
rmse; sqrt(rmse)
formula(lm.fit2) # 구해진 최적 모델의 포뮬러

# 시각화
library(ggplot2)
ggplot(data=Boston_test) + geom_point(aes(x=lm.yhat2, y=Boston_test$medv))
