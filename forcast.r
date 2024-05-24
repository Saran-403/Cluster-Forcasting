knitr::opts_chunk$set(echo = TRUE)
install.packages("tidymodels", dependencies = TRUE)
install.packages("ggplot2", upgrade = TRUE)
if (!require(xts)) {
  install.packages('xts')
}

if (!require(quantmod)) {
  install.packages("quantmod")
}

library(xts)
library(quantmod)


library(tidyverse)
library(readxl)
library(lubridate)
library(zoo)
library(tidymodels)
library(readxl)
library(neuralnet)
library(knitr)

ExchangeUSD <- read_excel("D:/IIT/2nd Sem/Machine learning/Coursework/test/ExchangeUSD (2).xlsx") %>%
  janitor::clean_names() %>%
  mutate(date_in_ymd = ymd(yyyy_mm_dd)) %>%
  select(-1) %>%
  select(date_in_ymd,everything())

#all the input is in only one dataframe to be able to preserve the testing and training
#dataset for the two sets of input variables
usd_exchange_full = ExchangeUSD %>%
  mutate(previous_one_day_set_a = lag(ExchangeUSD$usd_eur,1),
         previous_one_day_set_b = lag(ExchangeUSD$usd_eur,1),
         previous_two_day_set_b = lag(ExchangeUSD$usd_eur,2),
         previous_one_day_set_c = lag(ExchangeUSD$usd_eur,1),
         previous_two_day_set_c = lag(ExchangeUSD$usd_eur,2),
         previous_three_day_set_c = lag(ExchangeUSD$usd_eur,3),
         previous_one_day_set_d = lag(ExchangeUSD$usd_eur,1),
         previous_two_day_set_d = lag(ExchangeUSD$usd_eur,2),
         five_day_rolling = rollmean(usd_eur,5, fill = NA),
         ten_day_rolling = rollmean(usd_eur,10, fill = NA)) %>%
  drop_na()

usd_exchange_full %>%
  pivot_longer(cols = 3,names_to = "kind",values_to = "rate") %>%
  ggplot(aes(date_in_ymd,rate, color = kind)) +
  geom_line() +
  facet_wrap(~kind) + theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=1
  )) +
  labs(x = "",
       title = "First Set of Input Variables") +
  theme(legend.position = "none")

usd_exchange_full %>%
  pivot_longer(cols = c(4,5),names_to = "kind",values_to = "rate") %>%
  ggplot(aes(date_in_ymd,rate, color = kind)) +
  geom_line() +
  facet_wrap(~kind) + theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=1
  )) +
  labs(x = "",
       title = "Second Set of Input Variables") +
  theme(legend.position = "none")

usd_exchange_full %>%
  pivot_longer(cols = 6:8,names_to = "kind",values_to = "rate") %>%
  ggplot(aes(date_in_ymd,rate, color = kind)) +
  geom_line() +
  facet_wrap(~kind) + theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=1
  )) +
  labs(x = "",
       title = "Third Set of Input Variables") +
  theme(legend.position = "none")

usd_exchange_full %>%
  pivot_longer(cols = 9:12,names_to = "kind",values_to = "rate") %>%
  ggplot(aes(date_in_ymd,rate, color = kind)) +
  geom_line() +
  facet_wrap(~kind) + theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=1
  )) +
  labs(x = "",
       title = "Fourth Set of Input Variables") +
  theme(legend.position = "none")

# We can create a function to normalize the data from 0 to 1
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }
# All the variables are normalized
normalized_usd = usd_exchange_full %>%
  mutate(across(3:12, ~normalize(.x)))
# Look at the data that has been normalized
summary(normalized_usd)

set.seed(123)
usd_train <- normalized_usd[1:400,]
usd_test <- normalized_usd[401:491,]

# We can create a function to unnormalize the data=
unnormalize <- function(x, min, max) {
  return( (max - min)*x + min ) }
# Get the min and max of the original training values
usd_min_train <- min(usd_exchange_full[1:400,3])
usd_max_train <- max(usd_exchange_full[1:400,3])
# Get the min and max of the original testing values
usd_min_test <- min(usd_exchange_full[401:491,3])
usd_max_test <- max(usd_exchange_full[401:491,3])
# Check the range of the min and max of the training dataset
usd_min_test
usd_min_train
usd_max_test
usd_max_train
relevant_pred_stat <- function(true_value, predicted_value, model_kind) {
  rbind((tibble(truth = true_value,
                prediction = predicted_value) %>%
           metrics(truth,prediction) %>%
           mutate(type = model_kind)),(tibble(truth = true_value,
                                              prediction = predicted_value) %>%
                                         mape(truth,prediction) %>%
                                         mutate(type = model_kind)))
}

set.seed(12345)
# function setup that creates 2 layer model
model_two_hidden_layers = function(hidden,sec_hidden) {
  nn_model_true = neuralnet(usd_eur ~ previous_one_day_set_a, data=usd_train, hidden=c(
    hidden,sec_hidden), linear.output=TRUE)
  train_results = compute(nn_model_true,usd_test[,3:4])
  truthcol = usd_exchange_full[401:491,3]$usd_eur
  predcol = unnormalize(train_results$net.result,usd_min_train, usd_max_train)[,1]
  relevant_pred_stat(truthcol,predcol,
                     "Two Hidden Layers") %>%
    mutate(hiddel_layers = paste0(hidden, " and ",sec_hidden),
           input_set = "A") %>%
    filter(.metric != "rsq")
}
# creation of different models with varying number of nodes
results_two_hidden_layers = bind_rows(
  lapply(1:10, function(n) {
    bind_rows(
      lapply(1:5, function(m) {
        model_two_hidden_layers(n,m)
      })
    )
  })) %>%
  janitor::clean_names()
# save the stat indices to a dataframe
set_a_models_two_layers = results_two_hidden_layers %>%
  select(-estimator) %>%
  pivot_wider(names_from = metric, values_from = estimate) %>%
  arrange(rmse)
kable(set_a_models_two_layers[1:10,])

# Combine the dataframes
set_a_models = rbind(set_a_models_two_layers[1:2,],set_a_models_two_layers)

###################################testing2##################################

# We can create a function to normalize the data from 0 to 1
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }
# All the variables are normalized
normalized_gbp = usd_exchange_full %>%
  mutate(across(3:12, ~normalize(.x)))
# Look at the data that has been normalized
summary(normalized_gbp)

set.seed(123)
gbp_train <- normalized_gbp[1:400,]
gbp_test <- normalized_gbp[401:491,]

# We can create a function to unnormalize the data=
unnormalize <- function(x, min, max) {
  return( (max - min)*x + min ) }
# Get the min and max of the original training values
gbp_min_train <- min(usd_exchange_full[1:400,3])
gbp_max_train <- max(usd_exchange_full[1:400,3])
# Get the min and max of the original testing values
gbp_min_test <- min(usd_exchange_full[401:491,3])
gbp_max_test <- max(usd_exchange_full[401:491,3])
# Check the range of the min and max of the training dataset
gbp_min_test
gbp_min_train
gbp_max_test
gbp_max_train


data <- read_excel('D:/IIT/2nd Sem/Machine learning/Coursework/test/ExchangeUSD (2).xlsx')%>%
  janitor::clean_names() %>%
  mutate(date_in_ymd = ymd(yyyy_mm_dd)) %>%
  select(-1) %>%
  select(date_in_ymd,everything())
spy <- xts(data[,3],order.by = as.Date(data$date_in_ymd) )
rspy <- dailyReturn(spy)
rspy1 <-stats::lag(rspy,k=1)
rspyall <- cbind(rspy,rspy1)
colnames(rspyall) <- c('rspy','rspy1')
rspyall <- na.exclude(rspyall)

rspyt = window(rspyall,end = '2013-03-11')
rspyf =window(rspyall, start ='2013-03-11' ,end = '2013-10-02' )


ann = neuralnet(usd_eur~previous_one_day_set_a,data = usd_train,hidden = c(1,6),act.fct = function(x){x})
ann$result.matrix
plot(ann)

ann1 = neuralnet(rspy~rspy1,data = rspyt,hidden = 2,act.fct = function(x){x})
ann1$result.matrix
plot(ann1)

ann3 = neuralnet(usd_eur~previous_one_day_set_a,data = usd_train,hidden =2,act.fct = function(x){x})
ann3$result.matrix
plot(ann3)

preds=compute(ann,usd_test)
view(preds$net.result)
truthcol=usd_exchange_full[401:491,3]
predcol = unnormalize(preds$net.result,usd_min_train, usd_max_train)[,1]

view(predcol)
view(truthcol)

rmse=sqrt(mean((predcol-truthcol$usd_eur)^2))
rmse
summary(ann$call)
table(truthcol$usd_eur[26],predcol[26])

relevant_pred_stat(truthcol$usd_eur,predcol,
                   "Two Hidden Layers")

# Ensure that 'rspy1' column exists in 'usd_test' dataframe
usd_test$rspy1 <- stats::lag(usd_test$usd_eur, k = 1)

preds1 = compute(ann1, usd_test)

view(preds1$net.result)
truthcol1=usd_exchange_full[401:491,3]
predcol1 = unnormalize(preds1$net.result,usd_min_train, usd_max_train)[,1]
view(predcol1)
relevant_pred_stat(truthcol1$usd_eur,predcol1,
                   "Two Hidden Layers")

preds2=predict(ann3,newdata = usd_test)
view(preds2)
truthcol2=usd_exchange_full[401:491,3]
predcol2 = unnormalize(preds2, usd_min_train, usd_max_train)[,1]
view(predcol2)
relevant_pred_stat(truthcol2$usd_eur,predcol2,
                   "Two Hidden Layers")


# Scatter plot of predicted vs. actual exchange rate values
plot_df <- tibble(
  Truth = truthcol2$usd_eur,
  Prediction = predcol2
)

ggplot(plot_df, aes(x = Truth, y = Prediction)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(x = "Actual Exchange Rate (EUR to USD)",
       y = "Predicted Exchange Rate (EUR to USD)",
       title = "Scatter Plot of Predicted vs. Actual Exchange Rate Values") +
  theme_minimal()

# Calculate RMSE and MAPE
rmse <- sqrt(mean((predcol2 - truthcol2$usd_eur)^2))
mape <- mean(abs((truthcol2$usd_eur - predcol2) / truthcol2$usd_eur)) * 100

# Display RMSE and MAPE
cat("Root Mean Squared Error (RMSE):", rmse, "\n")
cat("Mean Absolute Percentage Error (MAPE):", mape, "%\n")
