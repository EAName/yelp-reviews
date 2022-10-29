### YELP PCA MODEL ON BUSINESS DATA

### PCA ANALYSIS AND PRINCIPAL COMPONENTS FED INTO LOGISTIC REGRESSION

### Results:
#   31 principal components = 95% of the variance in the data
#   Logistic Regression with 31 principal components run across 3 trials to create results means.
#   Mean accuracy: 0.3323
#   Mean F1 score (only classes 1.5 - 4.5): 0.2031
#   Mean Precision (all classes):0.1610
#   Mean Recall: (only classes 1.5-4.5): 0.3035
#   ROC Curves produced for all 3 models


data.path <- 'C:\\Users\\Elaine\\Documents\\Desk R\\YelpR\\';
data.file <- paste(data.path,'yelp_business_clean_version6.csv',sep='');

df = read.csv(data.file,header=TRUE);

options(max.print = 100000)  

# drop rows to match linear regression 1

df2 = subset(df, select = -c(stars,business_id,name,address,city,state,postal_code,latitude,
                             longitude,attributes,categories,BusinessParking,market, 
                             Alcohol_None, BYOBCorkage_yes_free, NoiseLevel_average, 
                             NoiseLevel_quiet, Smoking_yes, WiFi_no, Nightlife, 
                             Bars, Sandwiches) )



#scale the review_count and restaurantpricerange 2 before PCA
library(caret)   #apply scaling between 0-1

#trainTransformed <- predict(preProcValues, training)

#create df for just scaled columns

df_scale = subset(df2, select = c(review_count,RestaurantsPriceRange2))

#pp <- preProcess(df2, method = list("range", scale = names(df2)['review_count']))

preprocvalues <- preProcess(df_scale, method = c("range"))

df_scale2 <- predict(preprocvalues, df_scale)

#add scaled columns to df2

df2$review_count_scale = df_scale2$review_count
df2$RestaurantsPriceRange2_scale = df_scale2$RestaurantsPriceRange2

#create new df to drop unscaled columns

df3 <- subset(df2, select= -c(review_count,RestaurantsPriceRange2))

### PCA 

library(psych)
library(DataExplorer)

plot_correlation(df3)
#parking NA columns have a high correlation to each other

pc <- prcomp(df3, scale = FALSE)

summary(pc)

#Top 31 PCs = 95% of the variance
    
# look at pc loadings since there are too many variables to plot
pc$rotation

#export loadings to csv
pc_loadings <- as.data.frame(pc$rotation)

library(readr)

coln <- colnames(df3)

pc_loadings$coln <-coln

write_csv(pc_loadings, "~/Desk R/YelpR/pcloadings.csv")

### pass pcs to logistic regression

# isolate the principal components that explain 95% of the variation

yelp.pcs <- pc$x[,1:31]

yelp.pcsdf <- as.data.frame(yelp.pcs)

yelp.pcsdf$stars <- df$stars

#yelp_modelpc = "stars ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 +
#                  PC9 + PC10 + PC11 + PC12+ PC13 + PC14 + PC15 + PC16 + PC17 + PC18 + PC19 + 
#                  PC20 + PC21+ PC22 + PC23 + PC24 + PC25+ PC26 + PC27 + PC28 + PC29+ PC30 + PC31" 

### LOGISTIC REGRESSION

library(nnet)  #for multinomial logistic regression
library(tidyverse)


# Split the data into training and test set
set.seed(123)
training.samples <- yelp.pcsdf$stars %>% 
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- yelp.pcsdf[training.samples, ]
test.data <- yelp.pcsdf[-training.samples, ]

# Fit the model
pcalogregmodel <- nnet::multinom(stars ~., data = train.data)
# Summarize the model
summary(pcalogregmodel)
# Make predictions
predicted.classes <- pcalogregmodel %>% predict(test.data)
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

cat("\n\nLogistic Model with PCA Results:\n")
print(round(logisticresults, digits = 3))

write_csv(logisticresults, "~/Desk R/YelpR/logisticpca1.csv")

# create AUC-ROC curve for model 1 & 3 for comparison
library(pROC)
library(multiROC)

roc1 <- pROC::multiclass.roc(test.data$stars, as.numeric(predicted.classes))

plot.roc(smooth(roc1$rocs[[1]]), 
         print.auc=T,
         legacy.axes = T,
         print.auc.adj = c(-1,-5))
plot.roc(smooth(roc1$rocs[[2]]),
         add=T, col = 'red',
         print.auc = T,
         legacy.axes = T,
         print.auc.adj = c(-1,-3))
plot.roc(smooth(roc1$rocs[[3]]),add=T, col = 'blue',
         print.auc=T,
         legacy.axes = T,
         print.auc.adj = c(-1,-1))
plot.roc(smooth(roc1$rocs[[4]]),add=T, col = 'green',
         print.auc=T,
         legacy.axes = T,
         print.auc.adj = c(-1,1))
plot.roc(smooth(roc1$rocs[[5]]),add=T, col = 'orange',
         print.auc=T,
         legacy.axes = T,
         print.auc.adj = c(-1,3))
plot.roc(smooth(roc1$rocs[[6]]),add=T, col = 'purple',
         print.auc=T,
         legacy.axes = T,
         print.auc.adj = c(-1,5))
plot.roc(smooth(roc1$rocs[[7]]),add=T, col = 'pink',
         print.auc=T,
         legacy.axes = T,
         print.auc.adj = c(-1,7))
plot.roc(smooth(roc1$rocs[[8]]),add=T, col = 'brown',
         print.auc=T,
         legacy.axes = T,
         print.auc.adj = c(-1,9))
plot.roc(smooth(roc1$rocs[[9]]),add=T, col = 'cadetblue1',
         print.auc=T,
         legacy.axes = T,
         print.auc.adj = c(-1,11))
legend('bottomright',
       legend = c('1',
                  '1.5',
                  '2',
                  '2.5',
                  '3',
                  '3.5',
                  '4',
                  '4.5',
                  '5'),
       col=c('black','red','blue', 'green', 'orange', 'purple', 'pink', 'brown',
             'cadetblue1'),lwd=2)
title("ROC-AUC Logistic Regression with PCA 1", line = 2.5)


### PCA LOGISTIC REGRESSION MODEL2
# Split the data into training and test set
set.seed(789)
training.samples <- yelp.pcsdf$stars %>% 
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- yelp.pcsdf[training.samples, ]
test.data <- yelp.pcsdf[-training.samples, ]

# Fit the model
pcalogregmodel <- nnet::multinom(stars ~., data = train.data)
# Summarize the model
summary(pcalogregmodel)
# Make predictions
predicted.classes <- pcalogregmodel %>% predict(test.data)
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

cat("\n\nLogistic Model with PCA Results:\n")
print(round(logisticresults, digits = 3))

write_csv(logisticresults, "~/Desk R/YelpR/logisticpca2.csv")

# ROC-AUC CURVE FOR PCA LOGISTIC MODEL 2
roc1 <- pROC::multiclass.roc(test.data$stars, as.numeric(predicted.classes))

plot.roc(smooth(roc1$rocs[[1]]), 
         print.auc=T,
         legacy.axes = T,
         print.auc.adj = c(-1,-5))
plot.roc(smooth(roc1$rocs[[2]]),
         add=T, col = 'red',
         print.auc = T,
         legacy.axes = T,
         print.auc.adj = c(-1,-3))
plot.roc(smooth(roc1$rocs[[3]]),add=T, col = 'blue',
         print.auc=T,
         legacy.axes = T,
         print.auc.adj = c(-1,-1))
plot.roc(smooth(roc1$rocs[[4]]),add=T, col = 'green',
         print.auc=T,
         legacy.axes = T,
         print.auc.adj = c(-1,1))
plot.roc(smooth(roc1$rocs[[5]]),add=T, col = 'orange',
         print.auc=T,
         legacy.axes = T,
         print.auc.adj = c(-1,3))
plot.roc(smooth(roc1$rocs[[6]]),add=T, col = 'purple',
         print.auc=T,
         legacy.axes = T,
         print.auc.adj = c(-1,5))
plot.roc(smooth(roc1$rocs[[7]]),add=T, col = 'pink',
         print.auc=T,
         legacy.axes = T,
         print.auc.adj = c(-1,7))
plot.roc(smooth(roc1$rocs[[8]]),add=T, col = 'brown',
         print.auc=T,
         legacy.axes = T,
         print.auc.adj = c(-1,9))
plot.roc(smooth(roc1$rocs[[9]]),add=T, col = 'cadetblue1',
         print.auc=T,
         legacy.axes = T,
         print.auc.adj = c(-1,11))
legend('bottomright',
       legend = c('1',
                  '1.5',
                  '2',
                  '2.5',
                  '3',
                  '3.5',
                  '4',
                  '4.5',
                  '5'),
       col=c('black','red','blue', 'green', 'orange', 'purple', 'pink', 'brown',
             'cadetblue1'),lwd=2)
title("ROC-AUC Logistic Regression with PCA 2", line = 2.5)


### PCA LOGISTIC REGRESSION MODEL3
# Split the data into training and test set
set.seed(2003)
training.samples <- yelp.pcsdf$stars %>% 
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- yelp.pcsdf[training.samples, ]
test.data <- yelp.pcsdf[-training.samples, ]

# Fit the model
pcalogregmodel <- nnet::multinom(stars ~., data = train.data)
# Summarize the model
summary(pcalogregmodel)
# Make predictions
predicted.classes <- pcalogregmodel %>% predict(test.data)
head(predicted.classes)
# Model accuracy
mean_accuracy <-mean(predicted.classes == test.data$stars)

# Building classification table
ctable <- table(test.data$stars, predicted.classes)

#use caret to get performance metrics
cm <- confusionMatrix(ctable)
#print(cm)

# extract F1 score for all classes
f1pca = cm[["byClass"]][ , "F1"] #for multiclass classification problems
#print(f1pca)

#extract precision
precisionpca = cm[["byClass"]][ , "Precision"]
#print(precisionpca)
#print(mean(precisionpca))

#extract RECALL
recallpca = cm[["byClass"]][ , "Recall"]
#print(recallpca)

logisticresults = data.frame(mean_accuracy, f1pca, precisionpca, recallpca )

cat("\n\nLogistic Model with PCA Results:\n")
print(round(logisticresults, digits = 3))

write_csv(logisticresults, "~/Desk R/YelpR/logisticpca3.csv")

# ROC-AUC CURVE FOR PCA LOGISTIC MODEL 3
roc1 <- pROC::multiclass.roc(test.data$stars, as.numeric(predicted.classes))

plot.roc(smooth(roc1$rocs[[1]]), 
         print.auc=T,
         legacy.axes = T,
         print.auc.adj = c(-1,-5))
plot.roc(smooth(roc1$rocs[[2]]),
         add=T, col = 'red',
         print.auc = T,
         legacy.axes = T,
         print.auc.adj = c(-1,-3))
plot.roc(smooth(roc1$rocs[[3]]),add=T, col = 'blue',
         print.auc=T,
         legacy.axes = T,
         print.auc.adj = c(-1,-1))
plot.roc(smooth(roc1$rocs[[4]]),add=T, col = 'green',
         print.auc=T,
         legacy.axes = T,
         print.auc.adj = c(-1,1))
plot.roc(smooth(roc1$rocs[[5]]),add=T, col = 'orange',
         print.auc=T,
         legacy.axes = T,
         print.auc.adj = c(-1,3))
plot.roc(smooth(roc1$rocs[[6]]),add=T, col = 'purple',
         print.auc=T,
         legacy.axes = T,
         print.auc.adj = c(-1,5))
plot.roc(smooth(roc1$rocs[[7]]),add=T, col = 'pink',
         print.auc=T,
         legacy.axes = T,
         print.auc.adj = c(-1,7))
plot.roc(smooth(roc1$rocs[[8]]),add=T, col = 'brown',
         print.auc=T,
         legacy.axes = T,
         print.auc.adj = c(-1,9))
plot.roc(smooth(roc1$rocs[[9]]),add=T, col = 'cadetblue1',
         print.auc=T,
         legacy.axes = T,
         print.auc.adj = c(-1,11))
legend('bottomright',
       legend = c('1',
                  '1.5',
                  '2',
                  '2.5',
                  '3',
                  '3.5',
                  '4',
                  '4.5',
                  '5'),
       col=c('black','red','blue', 'green', 'orange', 'purple', 'pink', 'brown',
             'cadetblue1'),lwd=2)
title("ROC-AUC Logistic Regression with PCA 3", line = 2.5)