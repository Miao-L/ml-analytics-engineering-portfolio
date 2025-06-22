#Bank Customer Churn Model
#Identify what factors contribute to customer churn
#Customer churn also known as customer attrition
#I Google and searched for the data set
#I found this dataset randomly on Kaggle https://www.kaggle.com/mathchi/churn-for-bank-customers


#install.packages("caret")
#install.packages("rpart")
#install.packages("rpart.plot")
#install.packages("randomForest")
#install.packages("dplyr")
#install.packages("janitor")
#install.packages("tidyverse")
library("randomForest")
library(caret)
library(dplyr)
library(rpart)
library(rpart.plot)
library("randomForest")
library(rminer)
library(nnet)
library(ggplot2)
library(Amelia)
library(caTools)
library(dummies)
library("tidyverse")
library("janitor")
library(magrittr)

#Import data
churn <-read.csv("/Users/miao/Downloads/Kaggle_Churn_Data.csv")

#Check the data
str(churn)
head(churn)

#It looks like "HasCrCard","IsActiveMember",and "Exited" are categorical variables; convert to factors
churn$HasCrCard = as.factor(churn$HasCrCard)
churn$IsActiveMember = as.factor(churn$IsActiveMember)
churn$Exited = as.factor(churn$Exited)

#Dataframe summary
summary(churn)

#Check historical churn
prop = tabyl(churn$Exited)
prop

#Change 0/1 to yes/no for Random Forest
churn$churn_flag <- ifelse(churn$Exited== 1,'Yes','No')

#Split data into TRAINING and TESTING dataset
set.seed(10)
trainId = createDataPartition(churn$Exited, p=0.7,list=FALSE,times=1)
train_db = churn[trainId,]
test_db = churn[-trainId,]

nrow(train_db)
nrow(test_db)

#Since several of the variables have big difference in scale; e.g. Age, Tenure, Balance
#We need to rescale them
#Check the distribution
gather_train =gather(train_db %>% 
                       select(CustomerId, Balance,EstimatedSalary, Tenure),
                     variable, value,
                     -CustomerId)

ggplot(gather_train , aes(value)) + facet_wrap(~variable, scales = 'free_x') +
  geom_histogram() + theme_bw()

#After checking the plot results, none of the variable have a gaussian distribution, we rescale them without standardization
normalize = function(x) {
  result = (x - min(x, na.rm = TRUE)
  ) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
  return(result)
}
norm.train = lapply(train_db %>% 
                      select(CustomerId, Balance,EstimatedSalary, Tenure),
                    normalize)
norm.train = do.call(cbind, norm.train) %>%
  as.data.frame()

####################
#Decision tree
####################
names(churn)

tree_fit = rpart(Exited ~., data = train_db %>% 
               select(-CustomerId), method="class")


####################
#Random Forest
####################
control = trainControl(method = "cv", number=5, 
                    classProbs = TRUE, summaryFunction = twoClassSummary)

#Change the Exited from 0/1 to yes/no
rm_model = train(churn_flag ~., data = train_db %>% select(-CustomerId, -Surname,-RowNumber, -Exited),
                 method = "rf",
                 ntree = 75,
                 tuneLength = 5,
                 metric = "ROC",
                 trControl = control)

rm_model

####################
#Logistic Regression
####################
fit.log = step(glm(Exited ~.,data = train_db %>% 
                     select(-CustomerId, -Exited, -churn_flag, -RowNumber, -Surname), 
                     family=binomial(link='logit')), 
                     direction="both")




names(train_db)
head(train_db)
