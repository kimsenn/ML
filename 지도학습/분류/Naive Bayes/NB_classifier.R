# 나이브베이즈
# install.packages("mlbench") # 투표데이터
library(mlbench)
data(HouseVotes84, package = "mlbench")
model <- naiveBayes(Class ~ ., data = HouseVotes84)
model
pred <- predict(model, HouseVotes84[,-1])
table(pred, HouseVotes84$Class)

# 라플라스 추정기 추가
model <- naiveBayes(Class ~ ., data = HouseVotes84, laplace = 3)
pred <- predict(model, HouseVotes84[,-1])
table(pred, HouseVotes84$Class)
