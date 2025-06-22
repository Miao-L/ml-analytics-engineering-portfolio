# Subset Selection Methods

# Best Subset Selection
# according to BIC, the best performer is the model with 6 variables. 
# According to  Cp , 10 variables. 
# Adjusted  R2  suggests that 11 might be best. 
# Again, no one measure is going to give us an entirely accurate picture... 
# but they all agree that a model with 5 or fewer predictors is insufficient, 
# and a model with more than 12 is overfitting.
library(ISLR)
library(dplyr)
head(Hitters)

fix(Hitters)
names(Hitters)
dim(Hitters)
sum(is.na(Hitters$Salary))
Hitters=na.omit(Hitters)
dim(Hitters)
sum(is.na(Hitters))

#regsubsets() function (part of the leaps library) performs best subset selection by identifying the best model that contains a given numbers of predictors, 
# where best is quantified using RSS. 
install.packages("leaps")
library(leaps)
regfit.full=regsubsets(Salary~.,Hitters)
summary(regfit.full)
# An asterisk ("*") indicates that a given variable is included in the corresponding model. 
# For instance, this output indicates that the best two-variable model contains only Hits and CRBI. 
# By default, regsubsets() only reports results up to the best eight-variable model. 
# But the nvmax option can be used in order to return as many variables as are desired.
regfit.full=regsubsets(Salary~.,data=Hitters,nvmax=19)
reg.summary=summary(regfit.full)
names(reg.summary)
# The R2 statistic increases from 0.3214501 when only one variable is included in the model to 0.5461159 when all variables are included. 
reg.summary$rsq
par(mfrow=c(2,2))
plot(reg.summary$rss,xlab="Number of Variables",ylab="RSS",type="l")
plot(reg.summary$adjr2,xlab="Number of Variables",ylab="Adjusted RSq",type="l")
# Find the max adjusted R squared
which.max(reg.summary$adjr2)
points(11,reg.summary$adjr2[11], col="red",cex=2,pch=20)
# For CP, BIC 
plot(reg.summary$cp,xlab="Number of Variables",ylab="Cp",type='l')
which.min(reg.summary$cp) # 10
points(10,reg.summary$cp[10],col="red",cex=2,pch=20)
which.min(reg.summary$bic) # 6
plot(reg.summary$bic,xlab="Number of Variables",ylab="BIC",type='l')
points(6,reg.summary$bic[6],col="red",cex=2,pch=20)
plot(regfit.full,scale="r2")
plot(regfit.full,scale="adjr2")
plot(regfit.full,scale="Cp")
plot(regfit.full,scale="bic")
coef(regfit.full,6)

# Forward and Backward Stepwise Selection
regfit.fwd=regsubsets(Salary~.,data=Hitters,nvmax=19,method="forward")
summary(regfit.fwd)
regfit.bwd=regsubsets(Salary~.,data=Hitters,nvmax=19,method="backward")
summary(regfit.bwd)
coef(regfit.full,7)
coef(regfit.fwd,7)
coef(regfit.bwd,7)

# Choosing Among Models
set.seed(1)
train=sample(c(TRUE,FALSE), nrow(Hitters),rep=TRUE)
test=(!train)
regfit.best=regsubsets(Salary~.,data=Hitters[train,],nvmax=19)
test.mat=model.matrix(Salary~.,data=Hitters[test,])
val.errors=rep(NA,19)
for(i in 1:19){
  coefi=coef(regfit.best,id=i)
  pred=test.mat[,names(coefi)]%*%coefi
  val.errors[i]=mean((Hitters$Salary[test]-pred)^2)
}
val.errors
which.min(val.errors)
coef(regfit.best,10)
predict.regsubsets=function(object,newdata,id,...){
  form=as.formula(object$call[[2]])
  mat=model.matrix(form,newdata)
  coefi=coef(object,id=id)
  xvars=names(coefi)
  mat[,xvars]%*%coefi
}
regfit.best=regsubsets(Salary~.,data=Hitters,nvmax=19)
coef(regfit.best,10)
k=10
set.seed(1)
folds=sample(1:k,nrow(Hitters),replace=TRUE)
cv.errors=matrix(NA,k,19, dimnames=list(NULL, paste(1:19)))
for(j in 1:k){
  best.fit=regsubsets(Salary~.,data=Hitters[folds!=j,],nvmax=19)
  for(i in 1:19){
    pred=predict(best.fit,Hitters[folds==j,],id=i)
    cv.errors[j,i]=mean( (Hitters$Salary[folds==j]-pred)^2)
  }
}
mean.cv.errors=apply(cv.errors,2,mean)
mean.cv.errors
par(mfrow=c(1,1))
plot(mean.cv.errors,type='b')
reg.best=regsubsets(Salary~.,data=Hitters, nvmax=19)
coef(reg.best,11)


# Credit: 
# #Credit: An Introduction to Statistical Learning https://courses.edx.org/courses/course-v1:StanfordOnline+STATSX0001+1T2020/course/
#  http://www.science.smith.edu/~jcrouser/SDS293/labs/lab8-r.html