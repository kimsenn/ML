# SVM
# install.packages("e1071")
library(e1071) # iris 데이터터
svm_model <- svm(Species ~ ., data=iris)
svm_model

# 예측하기
s <- subset(iris, select = -Species)
table(predict(svm_model, s), iris$Species)

# 튜닝하기 cost, gamma값 찾아내기
svm_tune <- tune(svm,
                 train.x=subset(iris, select=-Species),
                 train.y=iris$Species,
                 kernel="radial",
                 ranges=list(cost=10^(-1:2), gamma=c(.5,1,2))
                 )
svm_tune # cost 1, gamma 0.5

# 새로운 Parameter로 튜닝 후 예측
svm_model_after_tune <- svm(Species ~ ., data=iris, kernel="radial", cost=1, gamma=0.5)
pred <- predict(svm_model_after_tune,subset(iris, select=-Species))
table(pred,iris$Species)
