### YELP CLUSTER MODEL ON BUSINESS DATA

### Cluster model results:
#   Tried: Kmeans, kmodes, block cluster
#   Kmeans yielded somewhat distinct 3 clusters, but no significant difference
#   in average stars per cluster. So no support to develop a classification 
#   model based on clustering the business data.


data.path <- 'C:\\Users\\Elaine\\Documents\\Desk R\\YelpR\\';
data.file <- paste(data.path,'yelp_business_clean_version6.csv',sep='');

df = read.csv(data.file,header=TRUE);

options(max.print = 100000)  

# drop rows to match linear reg1

df2 = subset(df, select = -c(stars,business_id,name,address,city,state,postal_code,latitude,
                             longitude,attributes,categories,BusinessParking,market, 
                             Alcohol_None, BYOBCorkage_yes_free, NoiseLevel_average, 
                             NoiseLevel_quiet, Smoking_yes, WiFi_no, Nightlife, 
                             Bars, Sandwiches) )

#scale the review_count and restaurantpricerange 2 before PCA and cluster
library(caret)   #apply scaling between 0-1

df_scale = subset(df2, select = c(review_count,RestaurantsPriceRange2))

preprocvalues <- preProcess(df_scale, method = c("range"))

df_scale2 <- predict(preprocvalues, df_scale)

#add scaled columns to df2

df2$review_count_scale = df_scale2$review_count
df2$RestaurantsPriceRange2_scale = df_scale2$RestaurantsPriceRange2

#create new df to drop unscaled columns

df3 <- subset(df2, select= -c(review_count,RestaurantsPriceRange2))

#KMEANS CLUSTER
library(readr)
library(tidyverse)
library(DataExplorer)
library(cluster)
library(factoextra)

# trying nbclust to find optimal k size
library(NbClust)
set.seed(222)
dfsample <- df3[sample(1:nrow(df3), size = 3000, replace = FALSE),] 

clusterNo=NbClust(dfsample,distance="euclidean",
                  min.nc=2,max.nc=15,method="complete",index="all")

set.seed(1109)
dfsample1 <- df3[sample(1:nrow(df3), size = 4700, replace = FALSE),] 

clusterNo=NbClust(dfsample1,distance="binary",
                  min.nc=2,max.nc=15,method="complete",index="all")

# 2 iterations of nbclust suggest optimal k = 3

set.seed(1234)
k3 <- kmeans(df3, centers=3, nstart = 25)
fviz_cluster(k3, data = df3)

# try k =5
set.seed(1234)
k5 <- kmeans(df3, centers=5, nstart = 25)
fviz_cluster(k5, data = df3)

# cluster sizes
#k3 size
k3$size
#k5 size
k5$size

# get summary of 3 clusters by converting k3 to original numbers
#create df using names review_count scale and price range scale with original values

df4 <- df2
df4$review_count_scale <- df4$review_count
df4$RestaurantsPriceRange2_scale <- df4$RestaurantsPriceRange2

#drop review_count and pricerange2
df4 <- subset(df4, select= -c(review_count,RestaurantsPriceRange2))


clconv3 <- df4 %>%
  mutate(Cluster = k3$cluster) %>%
  group_by(Cluster) %>%
  summarise_all("mean")

print(clconv3, width = Inf)

#use readr to write csv - copy in your filepath
write_csv(clconv3, "~/Desk R/YelpR/clustkm3summ.csv")

#get mean star rating by cluster
df4$stars <- df$stars

#assign cluster label to row
df4$Cluster <- k3$cluster

df4 %>%
  group_by(Cluster) %>%
  summarise_at(vars(stars), list(name = mean))

cluster_tibble <- df4  %>%
  group_by(Cluster) %>%
  summarise(
    n = n(),
    mean_stars = mean(stars, na.rm=T),
    mean_review_count = mean(review_count_scale, na.rm = T),
    mean_price_range = mean(RestaurantsPriceRange2_scale, na.rm = T))


# Export tibble
write.table(cluster_tibble, file = "~/Desk R/YelpR/cluster_tibble_summary.csv", sep="," )


# Try co-clustering due to discrete values
library(blockcluster)

#df3.matrix = train.credit.dummies
#train.credit.matrix = dplyr::arrange(train.credit.matrix,log_credit_amount)
df3.matrix=as.matrix(df3)
coclus.m1 = coclusterBinary(df3.matrix,nbcocluster=c(2,5))
plot(coclus.m1)

summary(coclus.m1)


#try blockcluster with just the binary data

df_binary <-subset(df3, select= -c(review_count_scale,RestaurantsPriceRange2_scale))

v_rowMeans <- rowMeans(df_binary)
v_colMeans <- colMeans(df_binary)

df_binarym <- df_binary[order(v_rowMeans), order(v_colMeans)]

dfbinm.matrix=as.matrix(df_binarym)
coclus.m2 = coclusterBinary(df3.matrix,nbcocluster=c(3,3))
plot(coclus.m2)

#look at dispersion plot
plot(coclus.m2, type = "distribution")

# try (2,4)
coclus.m3 = coclusterBinary(dfbinm.matrix,nbcocluster=c(2,4))
plot(coclus.m3)

# block clustering did not produce meaningful distinct clusters

#TRY KMODES WITH KLAR
library(klaR)

kmodes.1 <- kmodes(df_binarym, 3, iter.max = 10, weighted = FALSE)
par(mar = c(1, 1, 1, 1))
plot(df_binarym,col= kmodes.1$cluster)
#points(kmodes.1$modes,, col = 1:47, pch = 8)
par("mar")

kmodes.1$size

kmodesout <- kmodes.1$modes

#kmodes clusters are not meaningful
#use readr to write csv - copy in your filepath
write_csv(kmodesout, "~/Desk R/YelpR/kmodes3.csv")


