### Yelp Logistic Regression St. Louis Business data

### Status: 3 folds of logistic regression are complete, need to calculate 
#   weighted avg. metrics to determine the best results to include 


data.path <- 'C:\\Users\\Elaine\\Documents\\Desk R\\YelpR\\';
data.file <- paste(data.path,'df_stlouis.csv',sep='');

df = read.csv(data.file,header=TRUE);

options(max.print = 100000)  

# drop rows to match linear regression 1, keep stars in the df

df2 = subset(df, select = -c(business_id,name,address,city,state,postal_code,latitude,
                             longitude,attributes,categories,BusinessParking,market, 
                             Alcohol_None, BYOBCorkage_yes_free, NoiseLevel_average, 
                             NoiseLevel_quiet, Smoking_yes, WiFi_no, Nightlife, 
                             Bars, Sandwiches) )

library(nnet)  #for multinomial logistic regression
library(tidyverse)
library(caret)

### LOGISTIC MODEL 1

# Split the data into training and test set
set.seed(123)
training.samples <- df2$stars %>% 
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- df2[training.samples, ]
test.data <- df2[-training.samples, ]

# Fit the model
logregmodel <- nnet::multinom(stars ~., data = train.data)
# Summarize the model
summary(logregmodel)
# Make predictions

options(dplyr.print_max = 1e9)

#check pvalue of logistic regression result

library(broom)
tidy(logregmodel, conf.int= TRUE)

# all pvalues <0.05

# Make predictions
predicted.classes <- logregmodel %>% predict(test.data)
head(predicted.classes)

# Model accuracy
mean_accuracy <- mean(predicted.classes == test.data$stars)

# Building classification table
ctable <- table(test.data$stars, predicted.classes)

#get csv of confusion Matrix table
write.table(ctable, file = "~/Desk R/YelpR/stllogisticctable1.csv", sep="," )

#use caret to get performance metrics
cm <- confusionMatrix(ctable)
print(cm)

# extract F1 score for all classes
f1pca = cm[["byClass"]][ , "F1"]
print(f1pca)

#extract precision
precisionpca = cm[["byClass"]][ , "Precision"]

#extract RECALL
recallpca = cm[["byClass"]][ , "Recall"]

logisticresults = data.frame(mean_accuracy, f1pca, precisionpca, recallpca )

write_csv(logisticresults, "~/Desk R/YelpR/stllogisticmod1.csv")

# Variable Importance Model 1

imp <- varImp(logregmodel, scale = FALSE)

logmod1importance <- data.frame(overall = imp$Overall,
                                names   = rownames(imp))
logmod1importance[order(logmod1importance$overall,decreasing = T),]

write_csv(logmod1importance, "~/Desk R/YelpR/logisticmod1import.csv")

### ST LOUIS LOGISTIC MODEL 2
set.seed(608)
training.samples <- df2$stars %>% 
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- df2[training.samples, ]
test.data <- df2[-training.samples, ]

# Fit the model
logregmodel <- nnet::multinom(stars ~., data = train.data)
# Summarize the model
summary(logregmodel)

# Make predictions
predicted.classes <- logregmodel %>% predict(test.data)
head(predicted.classes)

# Model accuracy
mean_accuracy <- mean(predicted.classes == test.data$stars)

# Building classification table
ctable <- table(test.data$stars, predicted.classes)

#get csv of confusion Matrix table
write.table(ctable, file = "~/Desk R/YelpR/stllogisticctable2.csv", sep="," )

#use caret to get performance metrics
cm <- confusionMatrix(ctable)
print(cm)

# extract F1 score for all classes
f1pca = cm[["byClass"]][ , "F1"]
print(f1pca)

#extract precision
precisionpca = cm[["byClass"]][ , "Precision"]

#extract RECALL
recallpca = cm[["byClass"]][ , "Recall"]

logisticresults = data.frame(mean_accuracy, f1pca, precisionpca, recallpca )

write_csv(logisticresults, "~/Desk R/YelpR/stllogisticmod2.csv")

# Variable Importance Model 2

imp <- varImp(logregmodel, scale = FALSE)

logmodimportance <- data.frame(overall = imp$Overall,
                                names   = rownames(imp))
logmodimportance[order(logmodimportance$overall,decreasing = T),]

write_csv(logmodimportance, "~/Desk R/YelpR/stlouislogisticmod2import.csv")


### ST LOUIS LOGISTIC MODEL 3
set.seed(1987)
training.samples <- df2$stars %>% 
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- df2[training.samples, ]
test.data <- df2[-training.samples, ]

# Fit the model
logregmodel <- nnet::multinom(stars ~., data = train.data)
# Summarize the model
summary(logregmodel)

# Make predictions
predicted.classes <- logregmodel %>% predict(test.data)
head(predicted.classes)

# Model accuracy
mean_accuracy <- mean(predicted.classes == test.data$stars)

# Building classification table
ctable <- table(test.data$stars, predicted.classes)

#get csv of confusion Matrix table
write.table(ctable, file = "~/Desk R/YelpR/stllogisticctable3.csv", sep="," )

#use caret to get performance metrics
cm <- confusionMatrix(ctable)
print(cm)

# extract F1 score for all classes
f1pca = cm[["byClass"]][ , "F1"]
print(f1pca)

#extract precision
precisionpca = cm[["byClass"]][ , "Precision"]

#extract RECALL
recallpca = cm[["byClass"]][ , "Recall"]

logisticresults = data.frame(mean_accuracy, f1pca, precisionpca, recallpca )

write_csv(logisticresults, "~/Desk R/YelpR/stllogisticmod3.csv")

# Variable Importance Model 3

imp <- varImp(logregmodel, scale = FALSE)

logmodimportance <- data.frame(overall = imp$Overall,
                               names   = rownames(imp))
logmodimportance[order(logmodimportance$overall,decreasing = T),]

write_csv(logmodimportance, "~/Desk R/YelpR/stlouislogisticmod3import.csv")