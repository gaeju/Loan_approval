
# 필요한 패키지 불러오기
library(dplyr)
library(caret)
library(rpart)
library(rpart.plot)

# 데이터셋 불러오기
loan <- read.csv("loan.csv")

# 이상치 제거 (Applicant_Income, Coapplicant_Income, Loan_Amount 컬럼에 대해서만 적용)
loan <- loan %>% 
  mutate(Applicant_Income = ifelse(Applicant_Income > quantile(Applicant_Income, 0.99), 
                                   NA, Applicant_Income),
         Coapplicant_Income = ifelse(Coapplicant_Income > quantile(Coapplicant_Income, 0.99), 
                                     NA, Coapplicant_Income),
         Loan_Amount = ifelse(Loan_Amount > quantile(Loan_Amount, 0.99), NA, Loan_Amount))

colSums(is.na(loan))

# 결측치 확인
colSums(is.na(loan))
sum(is.na(loan))

# 결측치 대체 
median_Term <- median(loan$Term, na.rm = TRUE)
loan$Term[is.na(loan$Term)] <- median_Term

loan <- na.omit(loan)
sum(is.na(loan))


# 스케일링 (Applicant_Income, Coapplicant_Income, Loan_Amount 컬럼에 대해서만 스케일링)
preproc <- preProcess(loan[,c("Applicant_Income", "Coapplicant_Income", "Loan_Amount")], method = c("center", "scale"))
loan[,c("Applicant_Income", "Coapplicant_Income", "Loan_Amount")] <- predict(preproc, loan[,c("Applicant_Income", "Coapplicant_Income", "Loan_Amount")])

# 교차 검증
set.seed(123)
k <- 5
folds <- createFolds(loan$Status, k = k, list = TRUE, returnTrain = TRUE)

accuracy_vec <- numeric(k)
precision_vec <- numeric(k)
recall_vec <- numeric(k)

for (i in seq_along(folds)) {
  # 데이터셋 분할
  train <- loan[folds[[i]], ]
  test <- loan[-folds[[i]], ]
  
  # 의사결정나무 모델링
  loan_tree <- rpart(Status ~ ., data = train, method = "class")
  
  # 모델 성능 평가
  predicted <- predict(loan_tree, newdata = test, type = "class")
  accuracy_vec[i] <- sum(predicted == test$Status) / length(test$Status)
  
  # Positive 클래스(대출 승인)에 대한 precision 계산
  predicted_positive <- predicted == "Y"
  actual_positive <- test$Status == "Y"
  precision_vec[i] <- sum(predicted_positive & actual_positive) / sum(predicted_positive)
  
  # Positive 클래스(대출 승인)에 대한 recall 계산
  recall_vec[i] <- sum(predicted_positive & actual_positive) / sum(actual_positive)
}

summary(loan_tree)

# 교차 검증 결과 출력
cat("Accuracy:", mean(accuracy_vec), "\n")
cat("Precision:", mean(precision_vec), "\n")
cat("Recall:", mean(recall_vec), "\n")

