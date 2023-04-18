# 랜덤포레스트
# 전립선 암 환자의 자료
# 패키지 install 및 라이브러리 load
# install.packages("party")
library(party)
library(randomForest)
# datasets
data(stagec)## rpart 패키지에서 제공하는 데이터
str(stagec) ## 전립선암 환자 146명 데이터

# 데이터셋의 결측값 제거
stagec1 <- subset(stagec, !is.na(g2))
stagec2 <- subset(stagec1, !is.na(gleason))
stagec3 <- subset(stagec2, !is.na(eet))

# train/test split
set.seed(1)
ind <- sample(2, nrow(stagec3), replace = TRUE, prob = c(0.7,0.3))
trainData<-stagec3[ind==1,]
testData<-stagec3[ind==2,]

# RF tree 적합
rf.tree = randomForest(ploidy~., data = trainData)
rf.tree # 500개의 tree생성, 분기점에 활용한 변수 7개

pred = predict(rf.tree, testData)
confusionMatrix(table(pred, testData$ploidy)) #예측결과와 실제결과 비교
table(pred, testData$ploidy)
# 중요도 확인
varImpPlot(rf.tree) # g2가 가장 높음
# g2 : percent of cells in G2 phase,
partialPlot(rf.tree,trainData, g2,"tetraploid")



