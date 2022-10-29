### Yelp business logistic regression

### Results:
#   Model 2 chosen for the paper due to highest accuracy: 0.347


data.path <- 'C:\\Users\\Elaine\\Documents\\Desk R\\YelpR\\';
data.file <- paste(data.path,'yelp_business_clean_version6.csv',sep='');

df = read.csv(data.file,header=TRUE);

options(max.print = 100000)  

# drop rows to match linear regression 1, keep stars in the df

df2 = subset(df, select = -c(business_id,name,address,city,state,postal_code,latitude,
                             longitude,attributes,categories,BusinessParking,market, 
                             Alcohol_None, BYOBCorkage_yes_free, NoiseLevel_average, 
                             NoiseLevel_quiet, Smoking_yes, WiFi_no, Nightlife, 
                             Bars, Sandwiches) )



### LOGISTIC REGRESSION

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
predicted.classes <- logregmodel %>% predict(test.data)
head(predicted.classes)
# Model accuracy
mean_accuracy <- mean(predicted.classes == test.data$stars)

# Building classification table
ctable <- table(test.data$stars, predicted.classes)

#use caret to get performance metrics
cm <- confusionMatrix(ctable)
print(cm)

# extract F1 score for all classes
f1pca = cm[["byClass"]][ , "F1"] #for multiclass classification problems
#print(mean(f1pca))

#extract precision
precisionpca = cm[["byClass"]][ , "Precision"]
#print(precisionpca)
#print(mean(precisionpca))

#extract RECALL
recallpca = cm[["byClass"]][ , "Recall"]
#print(recallpca)
#print(mean(recallpca))

logisticresults = data.frame(mean_accuracy, f1pca, precisionpca, recallpca )

cat("\n\nLogistic Model Results:\n")
print(round(logisticresults, digits = 3))

write_csv(logisticresults, "~/Desk R/YelpR/logisticmod1.csv")

### LOGISTIC MODEL 2

# Split the data into training and test set
set.seed(1123)
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

#use caret to get performance metrics
cm <- confusionMatrix(ctable)
print(cm)

#get csv of confusion Matrix table
write.table(ctable, file = "~/Desk R/YelpR/logisticctable2.csv", sep="," )

# extract F1 score for all classes
f1pca = cm[["byClass"]][ , "F1"] #for multiclass classification problems
#print(mean(f1pca))

#extract precision
precisionpca = cm[["byClass"]][ , "Precision"]
#print(precisionpca)
#print(mean(precisionpca))

#extract RECALL
recallpca = cm[["byClass"]][ , "Recall"]
#print(recallpca)
#print(mean(recallpca))

logisticresults = data.frame(mean_accuracy, f1pca, precisionpca, recallpca )

cat("\n\nLogistic Model Results:\n")
print(round(logisticresults, digits = 3))

write_csv(logisticresults, "~/Desk R/YelpR/logisticmod2.csv")

### LOGISTIC MODEL 3

# Split the data into training and test set
set.seed(5523)
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

#use caret to get performance metrics
cm <- confusionMatrix(ctable)
print(cm)

# extract F1 score for all classes
f1pca = cm[["byClass"]][ , "F1"] #for multiclass classification problems
#print(mean(f1pca))

#extract precision
precisionpca = cm[["byClass"]][ , "Precision"]
#print(precisionpca)
#print(mean(precisionpca))

#extract RECALL
recallpca = cm[["byClass"]][ , "Recall"]
#print(recallpca)
#print(mean(recallpca))

logisticresults = data.frame(mean_accuracy, f1pca, precisionpca, recallpca )

cat("\n\nLogistic Model Results:\n")
print(round(logisticresults, digits = 3))

write_csv(logisticresults, "~/Desk R/YelpR/logisticmod3.csv")
