# Project Of Patient Adherence Data

# Loading required libraries
library(dplyr); library(caret); library(DataExplorer); library(Hmisc)
library(caTools); library(randomForest); library(VIM); library(gbm)
library(xgboost); library(readr); library(stringr); library(caret)
library(car); library(MASS)


# Reading the data
PA_data <- read.csv("Patient Adherence - Data.csv", T, ",")

# Checking the head of the dataset
summary(PA_data)
head(PA_data)

# Checking for NAs in the dataset
table(is.na(PA_data))
tail(PA_data)

# Looking at the data the NA in the amount paid column can be replaced with 205
PA_data$AmountPaid<- with(PA_data, impute(AmountPaid, 14.35))

# Checking for imputed value
table(is.na(PA_data))
tail(PA_data)

# Removing the value of PHARMACY 1535, as it contains only one 
# row and will create error while dividing the data into train and test 
# and then including the column in the model
PA_data <- PA_data[-which(PA_data$Pharmacy == "PHARMACY 1535"),]

# Converting formats
PA_data$PatientID <- as.factor(PA_data$PatientID)
PA_data$Date <- as.Date(PA_data$Date, format = "%Y-%m-%d")
str(PA_data)


# Creating group of PateintID & Medication and also creating a column for the 
# difference of days between next purchase
PA_Group <- PA_data %>%
  group_by(PatientID, Medication) %>%
  mutate(Diff_days = Date - lag(Date))


View(PA_Group)
str(PA_Group)
summary(PA_Group$Diff_days)

# Performing EDA of the Data Set.
plot_str(PA_Group)
plot_histogram(PA_Group$AmountPaid, title = "Histogram Of Amount Paid")
plot_correlation(PA_Group, title = "Correlation Matrix")
plot_bar(PA_Group)

# Changing the class of the new variable created (No_Of_Days)
PA_Group$Diff_days <- as.integer(PA_Group$Diff_days)


# Creating of lapse of days column
PA_Group$Laps_days <- PA_Group$Diff_days - PA_Group$For_How_Many_Days


# Replacing NAs Value with 0
PA_Group[is.na(PA_Group)] <- 0
table(PA_Group$Diff_days)


# Creating a column for Patient Adherence
PA_Group$Patient_Adherence <- PA_Group$For_How_Many_Days >= PA_Group$Diff_days
PA_Group$Patient_Adherence[PA_Group$Patient_Adherence == 'TRUE'] <- 1
PA_Group$Patient_Adherence[PA_Group$Patient_Adherence == 'FALSE'] <- 0
table(PA_Group$Patient_Adherence)


# Changing the class of the variables as required
PA_Group$Patient_Adherence <- as.factor(PA_Group$Patient_Adherence)
PA_Group$AmountPaid <- as.numeric(PA_Group$AmountPaid)
View(PA_Group)
str(PA_Group)


# Removing Patient ID column as we don't need that to create model
PA_Group <- PA_Group[-1]

# Dividing the data using stratified sampling technique
set.seed(121)
train_rows <- sample.split(PA_Group, SplitRatio = 0.7)
PA_Group_train = PA_Group[train_rows,]
PA_Group_test = PA_Group[!train_rows,]


# Logistic Regression Model
# For creating the model we eliminated Medication, Date, For_How_Many_Days,
# Diff_days and Laps_days as either these column were used to derive the target variable
# or cannot be used for creating model.
fit <- glm(Patient_Adherence~.+
             Purchased.By+RETAIL.MAIL+QTY+AmountPaid,
           data = PA_Group_train, family = 'binomial')
summary(fit)


# Performing Step_AIC
step_fit <- stepAIC(fit)

# Step_AIC Model
fit_AIC <- glm(Patient_Adherence ~ Sex + Pharmacy + QTY, 
               family = binomial(link = "logit"), data = PA_Group_train)
summary(fit_AIC)

# Random forest Model
fit_rf <- randomForest(Patient_Adherence~. -Medication-Laps_days,
                         ntree = 500,
                       n_estimator = 100,
                       min_samples_split=4,
                       oob_score = T,
                       PA_Group_train)
fit_rf

# Checking for important variable
varImp(fit_rf)

# Creating a cross validation scheme
control <- trainControl(method = 'repeatedcv',
                        number = 5,
                        repeats = 5)
seed <-7
metric <- 'Accuracy'
set.seed(seed)
fit_default <- train(Patient_Adherence~. -Medication-Laps_days, 
                     data = PA_Group_train,
                     method = 'bayesglm',
                     metric = metric,
                     tuneLength = 15,
                     trControl = control)
print(fit_default)


# gbm using caret
gbm_caret <- train(Patient_Adherence~. -Medication-Laps_days,
                   data=PA_Group_train,
               metric='Accuracy',
               method='gbm',
               verbose=TRUE)

# XGBoost Model
# Creating dummy variable
results <- fastDummies::dummy_cols(PA_Group_train, 
                                   select_columns = c("Sex", "Pharmacy", 
                                                      "Purchased.By", 
                                                      "RETAIL.MAIL",
                                                      "Patient_Adherence"))

results_test <- fastDummies::dummy_cols(PA_Group_test, 
                                   select_columns = c("Sex", "Pharmacy", 
                                                      "Purchased.By", 
                                                      "RETAIL.MAIL",
                                                      "Patient_Adherence"))

View(results)
head(results)
matrix_xgboost <- as.matrix(results[,14:34])
output_xgboost <- as.matrix(results[13])
matrix_xgboost_test <- as.matrix(results_test[,14:34])
output_xgboost_test <- as.matrix(results_test[13])

fit_xgb <- xgboost(data = matrix_xgboost, label = output_xgboost, max.depth = 4,
                   eta = 1, nthread = 2, nrounds = 10,objective = "binary:logistic")

fit_xgb

# ANOVA on base model
anova(fit,test = 'Chisq')

# ANOVA from reduced model after applying the Step AIC
anova(fit_AIC,test = 'Chisq')


# Plot the fitted model
plot(fit$fitted.values)
plot(fit_AIC$fitted.values)


# Predicting Using Random forest Model
predict_rf <- predict(fit_rf, PA_Group_test)
confusionMatrix(predict_rf,PA_Group_test$Patient_Adherence)


# Predicting using bayesglm Model
predict_bysglm <- predict(fit_default, PA_Group_test)
confusionMatrix(predict_bysglm, PA_Group_test$Patient_Adherence)


# Predicting using GBM
predict_gbm <- predict(gbm_caret, newdata=PA_Group_test)
confusionMatrix(predict_gbm, PA_Group_test$Patient_Adherence)

# Predicting using XGBoost
predict_xgb <- predict(fit_xgb, matrix_xgboost_test)
predict_xgb1 <- as.factor(ifelse(predict_xgb>0.5,1,0))
output_xgboost1 <- as.factor(output_xgboost_test)
confusionMatrix(predict_xgb1, output_xgboost1)

# Ensemble Model
# Predicting the probabilities
predict_rf_pb <- predict(fit_rf, PA_Group_test, type = 'prob')
predict_bysglm_pb <- predict(fit_default, PA_Group_test, type = 'prob')
predict_gbm_pb <- predict(gbm_caret, newdata=PA_Group_test, type = 'prob')
predict_xgb_pb <- predict(fit_xgb, matrix_xgboost, type = 'prob')

# Changing the class of the dataset
predict_rf_pb <- as.data.frame(predict_rf_pb)

# Taking average of predictions
predict_avg <- (predict_rf_pb$`1`+predict_bysglm_pb$`1`+predict_gbm_pb$`1`)/3

# Splitting into binary classes at 0.5
predict_avg <- as.factor(ifelse(predict_avg>0.5,1,0))

# Checking for the accuracy of the predicted probability
confusionMatrix(predict_avg, PA_Group_test$Patient_Adherence)

#The majority vote
predict_majority <- as.factor(ifelse(predict_rf=='1' 
                                     & predict_bysglm=='1',
                                     '1',ifelse(predict_rf=='1' 
                                                & predict_gbm=='1','1',
                                                ifelse(predict_gbm=='1' 
                                                       & predict_bysglm=='1','1','0'))))

# Checking for accuracy of the predicted
confusionMatrix(predict_majority, PA_Group_test$Patient_Adherence)


# Final prediction done using Ensemble Model Averaging technique
PA_Group_test$PA_pred <- predict_gbm
confusionMatrix(PA_Group_test$Patient_Adherence, PA_Group_test$PA_pred)


