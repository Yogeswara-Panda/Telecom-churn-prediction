######################################################################
###  CASE STUDY ANALSYSIS :Telecom customer churn key indicators
######################################################################

##### Importing data
library(readr)
dataT=read.csv("C:/Users/panda/Desktop/DATA science/Classification/Telecom_Churn.csv")

dataT2 = dataT
sum(is.na(dataT)) # checking for NA values
library(dplyr)
library(data.table)
# Checking data structure
str(dataT2)
# To viewing table
View(dataT2)

### Histogram of the response variable ###

library(ggplot2)
library(corrplot)

qplot(dataT2$Customer.service.call,
      geom="histogram",
      binwidth=1,  
      main="Customer Churn", 
      xlab="Customer Service Calls", 
      fill=I("gray"), 
      col=I("red"))+theme.bw()


table(dataT$Churn, useNA = 'ifany') 
# class imbalance exists. False=572 True=95

# ------------------------------------------------------------
# Bifurcations of response variables in two category= 0,1
# -------------------------------------------------------------
dataTC0=subset(dataT,(dataT$Churn)==0)
dataTC1=subset(dataT,(dataT$Churn)==1)

dataT2C0=dataTC0[,c(9,12,15,19)]
dataT2C1=dataTC1[,c(9,12,15,19)]

### Obtaining descriptive statistics ###

# Load library package 'pastecs' needed for obtaining descriptive stats 
library(pastecs)
options(scipen=999)

# stat.desc(): function for displaying the descriptive statistics - mean, median, SD etc.
stat.desc(dataT2C0$Customer.service.calls)   
stat.desc(dataT2C1$Customer.service.calls)

library(glmnet)
hist(dataT2C0$Customer.service.calls)
hist(dataT2C1$Customer.service.calls)

### perform shapiro test
set.seed(123)
shapiro.test(dataT2C0$Customer.service.calls)
shapiro.test(dataT2C1$Customer.service.calls)
# As P < 0.05 hence the the distribution is a normal distribution

###Performing t test
t.test(dataT2C0$Customer.service.calls,dataT2C1$Customer.service.calls)
#As per above t test p-value < 0.05 

### Creating training and test set

nn=nrow(dataT)
set.seed(123)
indx=sample(1:nn,0.7*nn)
trdata = dataT[indx,]
tedata = dataT[-indx,]

#### Fitting full logistic regression (LR) model with all features
str(trdata)
table(trdata$Churn,useNA ='ifany')

FM=glm(Churn ~. ,
            data = trdata,family = binomial()
)
summary(FM)
names(FM)

#### Selecting features for fitting reduced logistic regression model
library(MASS)
step=stepAIC(FM, direction='both', trace = TRUE)
summary(step)

mod1=glm(formula = Churn ~ International.plan + Voice.mail.plan + 
           Total.day.minutes + Total.eve.charge + Total.night.minutes + 
           Total.intl.calls + Total.intl.charge + Customer.service.calls, 
         family = binomial(), data = trdata)
summary(mod1)

### predicting success probabilities using the LR model

pred.prob=predict(mod1,tedata[,-20], type = 'response')
hist(pred.prob)

### predicting success probability for an individual

#### Plotting ROC 
library(pROC)
roc1=roc(tedata$Churn,pred.prob,plot=TRUE,legacy.axes=TRUE)
roc1$auc
# Area under Curve value is 0.8061 ----Thus, we can say that the model prediction is good

#### Using ROC in deciding threshold
thresval=data.frame(sen=roc1$sensitivities, spec=roc1$specificities,thresholds=roc1$thresholds)
head(thresval)
thresval[thresval$sen>0.65&thresval$spec>0.5,] %>% write.csv("threshold.csv", row.names = F)

getwd()
View(thresval)

library(data.table)
setDT(tedata)
tedata$Churn
tedata[, Churn:= ifelse(Churn=='TRUE',1,0)]
library(ggplot2)
library(caret)
# After deciding Threshold = sensitivity 0.6666667	Specificity 0.8055152	threshold 0.2056829
pred.Y=ifelse(pred.prob>0.2056829,1,0)   
confusionMatrix(as.factor(tedata$Churn), as.factor(pred.Y))
str(tedata)
table(tedata$Churn,pred.Y)

###############################
## Random Forest
###############################
library(randomForest)

dataT3=dataT2
str(dataT3)
View(dataT3)

#Normalizing data 
dataT3$Account.length=(dataT3$Account.length-mean(dataT3$Account.length))/sd(dataT3$Account.length)
dataT3$Area.code=(dataT3$Area.code-mean(dataT3$Area.code))/sd(dataT3$Area.code)
dataT3$Number.vmail.messages=(dataT3$Number.vmail.messages-mean(dataT3$Number.vmail.messages))/sd(dataT3$Number.vmail.messages)
dataT3$Total.day.minutes=(dataT3$Total.day.minutes-mean(dataT3$Total.day.minutes))/sd(dataT3$Total.day.minutes)
dataT3$Total.day.calls=(dataT3$Total.day.calls-mean(dataT3$Total.day.calls))/sd(dataT3$Total.day.calls)
dataT3$Total.day.charge=(dataT3$Total.day.charge-mean(dataT3$Total.day.charge))/sd(dataT3$Total.day.charge)
dataT3$Total.eve.minutes=(dataT3$Total.eve.minutes-mean(dataT3$Total.eve.minutes))/sd(dataT3$Total.eve.minutes)
dataT3$Total.eve.calls=(dataT3$Total.eve.calls-mean(dataT3$Total.eve.calls))/sd(dataT3$Total.eve.calls)
dataT3$Total.eve.charge=(dataT3$Total.eve.charge-mean(dataT3$Total.eve.charge))/sd(dataT3$Total.eve.charge)
dataT3$Total.night.minutes=(dataT3$Total.night.minutes-mean(dataT3$Total.night.minutes))/sd(dataT3$Total.night.minutes)
dataT3$Total.night.calls=(dataT3$Total.night.calls-mean(dataT3$Total.night.calls))/sd(dataT3$Total.night.calls)
dataT3$Total.night.charge=(dataT3$Total.night.charge-mean(dataT3$Total.night.charge))/sd(dataT3$Total.night.charge)
dataT3$Total.intl.minutes=(dataT3$Total.intl.minutes-mean(dataT3$Total.intl.minutes))/sd(dataT3$Total.intl.minutes)
dataT3$Total.intl.calls=(dataT3$Total.intl.calls-mean(dataT3$Total.intl.calls))/sd(dataT3$Total.intl.calls)
dataT3$Total.intl.charge=(dataT3$Total.intl.charge-mean(dataT3$Total.intl.charge))/sd(dataT3$Total.intl.charge)
dataT3$Customer.service.calls=(dataT3$Customer.service.calls-mean(dataT3$Customer.service.calls))/sd(dataT3$Customer.service.calls)


dataT3$State=as.factor(dataT3$State)
dataT3$International.plan=as.factor(dataT3$International.plan)
dataT3$Voice.mail.plan=as.factor(dataT3$Voice.mail.plan)
dataT3$Churn=as.factor(dataT3$Churn)

View(dataT3)
####create train and test data
set.seed(123)
index=sample(nrow(dataT3), 0.7*nrow(dataT3))
train=dataT3[index,]
test=dataT3[-index,]

###RF model
train$Churn = as.factor(train$Churn)
table(factor(train$Churn))
library(randomForest)
modRF=randomForest(Churn ~., data=train, ntree=100, mtry=9)
modRF

###predicting success probabilities using the RF model
rf.pred0=predict(modRF,train[,-20],type="response")
confusionMatrix(as.factor(train$Churn), rf.pred0)

rf.pred=predict(modRF,test[,-20],type="response")
confusionMatrix(as.factor(test$Churn), rf.pred)

library(pROC)
roc3=roc(test$Churn,as.numeric(rf.pred),plot=TRUE,legacy.axes=TRUE)
roc3$auc
#Area Under Curve = 0.87. Model prediction is good.
