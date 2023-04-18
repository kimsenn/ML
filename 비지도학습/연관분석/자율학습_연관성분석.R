# 자율학습 - 연관성 분석

# 패키지, 데이터 준비
# install.packages(c("arules","arulesViz"))
library(arules)
library(arulesViz)
# Groceries # 9835 rows, 169 cols # 30일 동안 169 상품의 데이터
inspect(Groceries[1:10])
summary(Groceries)

# 기본탐색
# absolute : 절대빈도
sort(itemFrequency(Groceries, type="absolute"), decreasing = T)
# relative : 상대빈도
round(sort(itemFrequency(Groceries, type="relative"), decreasing = T),3)

itemFrequencyPlot(Groceries, topN=10, type="absolute")
itemFrequencyPlot(Groceries, topN=10, type="relative")

# 연관성 분석
apriori(Groceries) # set of 0 rules
result_rules <- apriori(Groceries,parameter = list(support=0.005, confidence=0.5, minlen=2))
result_rules # set of 120 rules
summary(result_rules)
inspect(result_rules[1:5,])

# 향상도 높은 순서대로
rules_lift <- sort(result_rules, by="lift", decreasing=T)
inspect(rules_lift[1:5,])

# 신뢰도 높은 순서대로
rules_conf <- sort(result_rules, by="confidence", decreasing=T)
inspect(rules_conf[1:5,])

# 특정 아이템 관련 규칙 찾기
milk_rules <- subset(rules_lift, items %in% "whole milk")
milk_rules
rhs.milk_rules <- subset(rules_lift, rhs %in% "whole milk")
rhs.milk_rules

# 특정 아이템(ex.whole milk)에 대한 연관규칙
wholemilk_rule <- apriori(Groceries,parameter = list(support=0.005, confidence=0.5, minlen=2),
                          appearance = list(default="lhs", rhs="whole milk"))
wholemilk_rule_lift <- sort(wholemilk_rule, by="lift", decreasing=T)
inspect(wholemilk_rule_lift[1:5,])

# 시각화
library(arulesViz)
plot(wholemilk_rule[1:10], method = "graph", measure = "lift", shading = "confidence")
