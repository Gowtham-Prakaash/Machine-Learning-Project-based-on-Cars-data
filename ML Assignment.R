rm(list=ls(all=TRUE))
## Let us first set the working directory path and import the data
getwd()

setwd("/Users/gowthamprakaash/Desktop/ML Assignment")

library('ggplot2') # visualization
## Warning: package 'ggplot2' was built under R version 3.4.1
library('car') # visualization
## Warning: package 'car' was built under R version 3.4.2
library('scales') # visualization
## Warning: package 'scales' was built under R version 3.4.3
##install.packages('psy')
library('AER') #Coefficients #install AER
require("tidyr")
library('corrplot')
library('caret')
library('purrr')
library('coefplot')
library('psych')
library('MASS')

library('leaflet.extras')#install these packages
library("PerformanceAnalytics")
library('psy')

library('nnet')
library('plyr')

library("e1071")
library('ggcorrplot')
##install.packages("sqldf")
library('mlogit')  # for multiple class logistic regression
library('caTools')
library('rpart.plot')
library('glmnet')
library('ggplot2')
library('RColorBrewer')
library('dummies') # for converting categorical into dummy one
library('caret')
library('pscl') ## for  McFadden R2
library('sqldf')
##install.packages('DMwR')
library('StatMeasures')
library('gains')
library('lubridate')
library('gbm')
library('VIM')  
library('DMwR') 

#Data Loading:

Trans_avail<- read.csv('Cars.csv')
summary(Trans_avail)

#Total rows 
nrow(Trans_avail)

#Type of datasets:
str(Trans_avail)

#Continous variable
hist(Trans_avail$Work.Exp, col = 'light blue')

#Categorical-License variable
hist(Trans_avail$license, col = 'dark green')

#Viz- summary of Gender vs Transpotation type:
viZ_summ <- ggplot(Trans_avail, aes(x = Trans_avail$Salary, y = Trans_avail$Work.Exp)) +
  facet_grid(~ Trans_avail$Gender + Trans_avail$Transport)+
  geom_boxplot(na.rm = TRUE, colour = "#3366FF",outlier.colour = "red", outlier.shape = 1) +
  labs(x = "Work Experience", y = "Salary") +
  scale_x_continuous() +
  scale_y_continuous() +
  theme(legend.position="bottom", legend.direction="horizontal")

viZ_summ

#Data preparation-Treating NULL values:
#We can use either KNN or MICE package, here we will be using KNN for handling null values
Trans_avail_imputed <- Trans_avail
summary(Trans_avail_imputed)

#We will use VIM:KNN for handling null value in MBA column
Trans_avail_imputed <- VIM::kNN(data=Trans_avail,variable =c("MBA"),k=7)  
## here explictly package name has to be added bcz the function name is conflicting with other package of SMOTE

summary(Trans_avail_imputed)

#After handling Null value imputation:
Trans_avail_Final <- subset(Trans_avail_imputed, select = Age:Transport)
Trans_avail_Final_boost <- subset(Trans_avail_imputed, select = Age:Transport)
#Trans_avail_Final_logit <- subset(Trans_avail_imputed, select = Age:Transport)
Trans_avail_Final
nrow(Trans_avail_Final)

#Checking for the balancing distribution:
table(Trans_avail_Final$Transport)

#It is evident that there are 3 types of transport which is preferred 
print(prop.table(table(Trans_avail_Final$Transport)))

#Model parameter Findings from the tune output
#We find the best parameter by tuning cost and episilon
summary(Trans_avail_Final)



##########Execute the following if its not taking time
svm_tune <- tune(svm, Transport~., data = Trans_avail_Final, ranges = list(cross = 7, epsilon = seq(0,1,0.01), cost = 2^(2:9)))

#Printing the function:
print(svm_tune)


Best_model <- svm_tune$best.model
svm_tune$performances

svm_tune$best.parameters$epsilon
svm_tune$best.parameters$cost
svm_tune$best.parameters$cross
################### 

#Now we will deploy SVM model -they ca be used to separate multiple hyperplanesdata space is divided into segments and each segment contains only one kind of data.
##we will proceed with  7-fold cross validation

svm_model<-svm(Trans_avail_Final$Transport~., data=Trans_avail_Final, kernel="radial", tolerance=0.0001, shrinking=TRUE, cross=7, fitted=TRUE)

summary(svm_model)

#Understanding the model accuracy:
pred <- predict(svm_model, Trans_avail_Final)
Trans_avail_Final$TransportPredicted <- pred

EmployeeTransport <- table(actualclass=Trans_avail_Final$Transport, predictedclass=Trans_avail_Final$TransportPredicted)

EmployeeTransport
confusionMatrix(EmployeeTransport)


###Now we will predict using the developed tune function: --If the above tuning function is exxecuted we can run the below code:
#plot(svm_tune)
##################################################################
mysvm <- svm(Trans_avail_Final$Transport~., data=Trans_avail_Final, cost = svm_tune$best.parameters$cost, epsilon = svm_tune$best.parameters$epsilon)
summary(mysvm)

pred_tuned <- predict(Best_model, Trans_avail_Final)
Trans_avail_Final$TransportPredictedTuned <- pred_tuned

EmployeeTransportTuned <- table(actualclass=Trans_avail_Final$Transport, predictedclass=Trans_avail_Final$TransportPredictedTuned)

EmployeeTransportTuned

confusionMatrix(EmployeeTransportTuned)

#################################################################

#Test the model with the new data:
transport_employee_aval_test <- read.csv('test.csv')
transport_employee_aval_test_addl <- transport_employee_aval_test

transport_employee_aval_test$TransportPredicted <- predict(svm_model, transport_employee_aval_test)
transport_employee_aval_test
levels(transport_employee_aval_test$PredictedTransport)



#MODEL which explains the decision to use CAR as main mode of transport
#Our aim here is to understand why car is being selected as transport. So we are not interested into three level of transport "2Wheeler","Car" and "Public Transport". 
#Our aim which will be "Car" and "NoCar" Data is biassed towards non-car

Trans_avail_Final_boost$Transport <- ifelse(Trans_avail_Final_boost$Transport == "Car",1,0)
table(Trans_avail_Final_boost$Transport )


summary(Trans_avail_Final_boost)
  library("dummies")

Trans_avail_Final_boost <- dummy.data.frame(Trans_avail_Final_boost, sep = ".")
summary(Trans_avail_Final_boost)


#REgression model:
#We will take initially all the variables into regression and then will further reliminate through stepAIC
Reg_Trans <- glm(Trans_avail_Final_boost$Transport ~.,family=binomial(link='logit'),data=Trans_avail_Final_boost)

summary(Reg_Trans)
boxplot(Trans_avail_Final_boost$Age ~ Trans_avail_Final_boost$Transport,main="Age vs Transport",xlab="Transport", ylab="Age")

boxplot(Trans_avail_Final_boost$Work.Exp ~ Trans_avail_Final_boost$Transport,main="Work Experience vs Transport",xlab="Transport", ylab="Work Experience")
boxplot(Trans_avail_Final_boost$Distance ~ Trans_avail_Final_boost$Transport,main="Distance vs Transport",xlab="Transport", ylab="Distance")
boxplot(Trans_avail_Final_boost$Salary ~ Trans_avail_Final_boost$Transport,main="Salary vs Transport",xlab="Transport", ylab="Salary")
ggplot(data=Trans_avail_Final_boost, aes(x= Trans_avail_Final_boost$Gender.Female)) + 
  geom_histogram(col="red",fill="light blue", bins = 25) +
  facet_grid(~ Trans_avail_Final_boost$Transport)+
  theme_bw()



#Steo AIC and Cofficients of the Model
#We will run stepAIC for finding out optimal ones and then will use the optimised one for final regression model
stepAIC(Reg_Trans, direction='both', steps = 1000, trace=TRUE)

reg_transport_final <- glm(formula = Trans_avail_Final_boost$Transport ~ 
                             Age + Gender.Female + MBA + Work.Exp + Salary + Distance + 
                             license, family = binomial(link = "logit"), data = Trans_avail_Final_boost)

coefficients(reg_transport_final)

library("coefplot")

coefplot.glm(reg_transport_final,parm = -1)

#model If VIF is more than 10, multicolinearity is strongly suggested and here we see there are two variable Age and Work Exp are having values more than 10
vif(reg_transport_final)



#Regularisation of the model
#Result shows that Age, Gender.Female, MBA and license are the key factor for determining transport as Car
#convert training data to matrix format
xInput_transport <- model.matrix(Trans_avail_Final_boost$Transport~.,Trans_avail_Final_boost)
yResponse <- Trans_avail_Final_boost$Transport

#perform grid search to find optimal value of lambda #family= binomial => logistic regression, alpha=1 => lasso 

Transport.out <- cv.glmnet(xInput_transport,yResponse, alpha=1, family="binomial", type.measure = "class")
#plot result
plot(Transport.out)

#min value of lambda
lambda_min <- Transport.out$lambda.min
#best value of lambda
lambda_1se <- Transport.out$lambda.1se
lambda_1se

#regression coefficients
coef(Transport.out,s=lambda_1se)

# Key Observations for critical variables seelcted after regularisation are:
# Box Plot shows users who are using car as transport have Median age much higher than non-car passenger. So higher age seems to be a driving factor for transport mode selection
# Box Plot shows users who are using car as transport have Median work experience much higher than non-car passenger. So higher work experience seems to be a driving factor for transport mode selection. This is also clear from age as higher work experience employee will have higher age.
# Box Plot shows users who are travelling log distance prefers to use car as preffered mode of transport
# Higher salary seems to be driving factor for choosing car as preferred one but there are quite a few outliers for non-car owner also who are earning higher salary
# No of Men seems to be much higher w.r.t Women for preffered mode of transport as Car or Non Car. This may be due to gender-inquality in Job.

