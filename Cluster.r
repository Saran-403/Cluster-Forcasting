# Install and load necessary libraries
if (!require("readxl")) install.packages("readxl")
if (!require("dplyr")) install.packages("dplyr")
if (!require("cluster")) install.packages("cluster")
if (!require("factoextra")) install.packages("factoextra")
if (!require("NbClust")) install.packages("NbClust")
if (!require("fpc")) install.packages("fpc")
if (!require("stats")) install.packages("stats")
if (!require("ggplot2")) install.packages("ggplot2")

library(readxl)
library(dplyr)
library(cluster)
library(factoextra)
library(NbClust)
library(fpc)
library(stats)
library(ggplot2)

# Load the data
wine_data <- read_excel("D:/IIT/2nd Sem/Machine learning/Coursework/test/Whitewine_v6.xlsx")

# Convert to dataframe for ggplot
wine_df <- as.data.frame(wine_data)

# Preprocess the data: Scaling
scaled_data <- scale(wine_data[, 1:11])  # Scale first 11 attributes

# Convert scaled data to dataframe for plotting
scaled_df <- as.data.frame(scaled_data)
colnames(scaled_df) <- colnames(wine_data)[1:11]

# Plot initial data with potential outliers
boxplot(scaled_df, main = "Attributes with Outliers", las = 2, col = 'gray', ylab = "Value")

# Outlier detection
outlier_flags <- apply(scaled_data, 2, function(x) {
  qnts <- quantile(x, probs=c(0.25, 0.75))
  iqr <- IQR(x)
  x < qnts[1] - 1.5 * iqr | x > qnts[2] + 1.5 * iqr
})
outliers <- rowSums(outlier_flags) > 0
clean_data_df <- scaled_df[!outliers,]

# Plot data after removing outliers
boxplot(clean_data_df, main = "Attributes without Outliers", las = 2, col = 'lightblue', ylab = "Value")

# Determine the number of clusters using various methods
# Elbow Method
fviz_nbclust(clean_data_df, kmeans, method = "wss")

# Silhouette Method
fviz_nbclust(clean_data_df, kmeans, method = "silhouette")

# Gap Statistic
gap_stat <- clusGap(clean_data_df, FUN = kmeans, nstart = 25, K.max = 10, B = 50)
fviz_gap_stat(gap_stat)

# NbClust
nb <- NbClust(clean_data_df, min.nc = 2, max.nc = 10, method = "kmeans", index = "all")
barplot(table(nb$Best.nc[1,]), main = "Best Number of Clusters", xlab = "Number of Clusters", ylab = "Frequency")

# Conduct K-means Clustering with optimal k
optimal_k <- as.numeric(names(which.max(table(nb$Best.nc[1,]))))
set.seed(123)
kmeans_result <- kmeans(clean_data_df, centers = optimal_k, nstart = 25)
print(kmeans_result)

# Evaluate clustering performance
fviz_cluster(kmeans_result, data = clean_data_df)

# Silhouette analysis
sil <- silhouette(kmeans_result$cluster, dist(clean_data_df))
plot(sil, main = "Silhouette Plot")

# Apply PCA and choose PCs that explain at least 85% cumulative variance
pca <- prcomp(clean_data_df)
pca_summary <- summary(pca)
fviz_eig(pca, addlabels = TRUE, ylim = c(0, 100))

# Select principal components that contribute to at least 85% variance
cum_var <- cumsum(pca_summary$importance[2,])
pca_data <- pca$x[, cum_var <= 0.85]

# Redo clustering on PCA-transformed data
kmeans_pca_result <- kmeans(pca_data, centers = optimal_k, nstart = 25)
fviz_cluster(kmeans_pca_result, data = pca_data)

# Silhouette plot for PCA-based clustering
sil_pca <- silhouette(kmeans_pca_result$cluster, dist(pca_data))
plot(sil_pca, main = "Silhouette Plot for PCA-based Clustering")

# Calculate the Calinski-Harabasz index for a range of cluster numbers
calinski_values <- sapply(2:10, function(k) {
  set.seed(123)
  model <- kmeans(clean_data_df, centers = k, nstart = 25)
  calinhara(clean_data_df, model$cluster)
})

# Plot the Calinski-Harabasz Index
plot(2:10, calinski_values, type = "b", pch = 19, frame = FALSE, xlab = "Number of Clusters", ylab = "Calinski-Harabasz Index",
     main = "Calinski Criterion Graph", col = "red")

# Create a color palette
colors <- rainbow(length(unique(kmeans_result$cluster)))

# Assuming kmeans_result is your K-means result variable
plot(kmeans_result$cluster, col = colors[kmeans_result$cluster], pch = 19,
     main = "K-means Partitions Comparison", xlab = "Objects", ylab = "Number of Groups in Each Partition")
legend("topright", legend = unique(kmeans_result$cluster), fill = colors, title = "Clusters")


# Integrate this into your existing workflow right after you perform k-means clustering


# Calculate Calinski-Harabasz Index
calinski_index <- calinhara(pca_data, kmeans_pca_result$cluster)
print(calinski_index)