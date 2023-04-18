# KNN
# 유방암 데이터:
# https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/wisc_bc_data.csv
wbcd <- read.csv("wisc_bc_data.csv")
str(wbcd)

# test, train - 569개 중 69개 테스트
train <- wbcd[1:500,c(-1,-2)]
test <- wbcd[501:569,c(-1,-2)]

# 데이터 예측
# install.packages('class')
library(class)
# k = 3
result_k3 <- knn(train, test, wbcd[1:500, c("diagnosis")], k=3)
result_k3
# 예측모델 평가
test_label <- wbcd[501:569,c("diagnosis")]
table(test_label, result_k3)

# k = 6
result_k6 <- knn(train, test, wbcd[1:500, c("diagnosis")], k=6)
result_k6
# 예측모델 평가
test_label <- wbcd[501:569,c("diagnosis")]
table(test_label, result_k6)

