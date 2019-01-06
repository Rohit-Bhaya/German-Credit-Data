## German Credit Data Risk Analysis
## Libraries Used

library(knitr)
library(dplyr)
library(tidyr)
library(reshape2)
library(RColorBrewer)
library(GGally)
library(ggplot2)
library(caret)
library(glmnet)
library(boot)
library(verification)

## Exploratory Data Analysis

### Data Import
german_credit <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")

colnames(german_credit) <- c("chk_acct", "duration", "credit_his", "purpose", 
                             "amount", "saving_acct", "present_emp", "installment_rate", "sex", "other_debtor", 
                             "present_resid", "property", "age", "other_install", "housing", "n_credits", 
                             "job", "n_people", "telephone", "foreign", "response")

german_credit$response <- german_credit$response - 1
german_credit$response <- as.factor(german_credit$response)

### Data Structure
glimpse(german_credit)

### Summary Statistics
summary(german_credit)

### EDA for continuous variables

#### `Duration` variable
amount.mean <- german_credit %>% dplyr::select(amount, response) %>% group_by(response) %>% summarise(m =mean(amount))
duration.mean <- german_credit %>% dplyr::select(duration, response) %>%group_by(response) %>% summarise( m =mean(duration))

ggplot(german_credit, aes(duration, fill=response)) + 
  geom_density(alpha=.5) + geom_vline(data=duration.mean, aes(xintercept=m,  colour=response),
                                      linetype="dashed", size=1)

test.m <- german_credit[,c(2,5,8,13,16,18,21)]
test.m$response <- as.numeric(test.m$response)
ggplot(melt(german_credit[,c(2,21)]), aes(x = variable, y = value, fill = response)) + geom_boxplot() + xlab("response") + ylab("duration")

#### `Installment Rate` variable
ggplot(german_credit, aes(factor(installment_rate), ..count..)) + 
  geom_bar(aes(fill = response), position = "dodge") + xlab("Installment Rates")

#### `Amount` variable
ggplot(german_credit, aes(amount, fill=response)) + 
  geom_density(alpha=.5) + geom_vline(data=amount.mean, aes(xintercept=m,  colour=response),
                                      linetype="dashed", size=1)
ggplot(melt(german_credit[,c(5,21)]), aes(x = variable, y = value, fill = response)) + 
  geom_boxplot() + xlab("response") + ylab("amount")

#### `Age` variable
ggplot(melt(german_credit[,c(13,21)]), aes(x = variable, y = value, fill = response)) + 
  geom_boxplot()+ xlab("response") + ylab("age")

#### `n_credits` variable
ggplot(melt(german_credit[,c(16,21)]), aes(x = variable, y = value, fill = response)) + 
  geom_boxplot()


### EDA for categorical variables
#### `chk_acct` variable
ggplot(german_credit, aes(chk_acct, ..count..)) + 
  geom_bar(aes(fill = response), position = "dodge") 

#### `credit_his` variable
ggplot(german_credit, aes(credit_his, ..count..)) + 
  geom_bar(aes(fill = response), position = "dodge") 

#### `purpose` variable
ggplot(german_credit, aes(purpose, ..count..)) + 
  geom_bar(aes(fill = response), position = "dodge") 

#### `saving_acct` variable
ggplot(german_credit, aes(saving_acct, ..count..)) + 
  geom_bar(aes(fill = response), position = "dodge")

#### `other_debtor` variable
ggplot(german_credit, aes(other_debtor, ..count..)) + 
  geom_bar(aes(fill = response), position = "dodge")

#### `sex` variable
ggplot(german_credit, aes(sex, ..count..)) + 
  geom_bar(aes(fill = response), position = "dodge") 

#### `other_install` variable
ggplot(german_credit, aes(other_install, ..count..)) + 
  geom_bar(aes(fill = response), position = "dodge") 

#### `foreign` variable
ggplot(german_credit, aes(foreign, ..count..)) + 
  geom_bar(aes(fill = response), position = "dodge")


## Logistic Regression to Predict Riskiness

### Train/Test split

set.seed(12420246)
in.train <- createDataPartition(as.factor(german_credit$response), p=0.8, list=FALSE)
german_credit.train <- german_credit[in.train,]
german_credit.test <- german_credit[-in.train,]

### Stepwise variable selection using AIC

credit.glm0 <- glm(response ~ ., family = binomial, german_credit.train)
credit.glm.step <- step(credit.glm0, direction = "backward")

summary(credit.glm.step)

### Stepwise variable selection using BIC

credit.glm.step.bic <- step(credit.glm0, k = log(nrow(german_credit.train)))
summary(credit.glm.step.bic)

### Lasso variable selection

factor_var <- c(1,3,4,6,7,9,10,12,14,15,17,19,20,21)
num_var <- c(2,5,8,11,13,16,18)
train2 <- german_credit.train
train2[num_var] <- scale(train2[num_var])
train2[factor_var] <- sapply(train2[factor_var] , as.numeric)

X.train <- as.matrix(train2[,1:20])
Y.train <- as.matrix(train2[,21])
lasso.fit<- glmnet(x=X.train, y=Y.train, family = "binomial", alpha = 1)
plot(lasso.fit, xvar = "lambda", label=TRUE)

cv.lasso<- cv.glmnet(x=X.train, y=Y.train, family = "binomial", alpha = 1, nfolds = 10)
plot(cv.lasso)
cv.lasso$lambda.1se

coef(lasso.fit, s=cv.lasso$lambda.1se)

### Final logistic model for GLM

credit.glm.final <- glm(response ~ chk_acct + duration + credit_his + amount + saving_acct + other_install + installment_rate, family = binomial, german_credit.train)

summary(credit.glm.final)

## Model Evaluation

### In-sample misclassification rate

prob.glm1.insample <- predict(credit.glm.final, type = "response")
predicted.glm1.insample <- prob.glm1.insample > 0.1667
predicted.glm1.insample <- as.numeric(predicted.glm1.insample)
mean(ifelse(german_credit.train$response != predicted.glm1.insample, 1, 0))

### Confusion Matrix

table(german_credit.train$response, predicted.glm1.insample, dnn = c("Truth", "Predicted"))

### ROC Plot

roc.plot(german_credit.train$response == "1", prob.glm1.insample)
roc.plot(german_credit.train$response == "1", prob.glm1.insample)$roc.vol$Area

### Out of sample misclassification rate and AUC score

prob.glm1.outsample <- predict(credit.glm.final, german_credit.test, type = "response")
predicted.glm1.outsample <- prob.glm1.outsample > 0.1667
predicted.glm1.outsample <- as.numeric(predicted.glm1.outsample)
table(german_credit.test$response, predicted.glm1.outsample, dnn = c("Truth", "Predicted"))

mean(ifelse(german_credit.test$response != predicted.glm1.outsample, 1, 0))

roc.plot(german_credit.test$response == "1", prob.glm1.outsample)
roc.plot(german_credit.test$response == "1", prob.glm1.outsample)$roc.vol$Area

### Asymmetric misclassification rate giving more penalty for false positives

cost1 <- function(r, pi) {
  weight1 = 5
  weight0 = 1
  c1 = (r == 1) & (pi < 0.17)  #logical vector - true if actual 1 but predict 0
  c0 = (r == 0) & (pi > 0.17)  #logical vecotr - true if actual 0 but predict 1
  return(mean(weight1 * c1 + weight0 * c0))
}

cost1(german_credit.test$response,predicted.glm1.outsample)