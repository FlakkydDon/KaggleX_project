if(!is.null(dev.list())) dev.off()  # clear out the past 
rm(list = ls())
cat("\014")

#load the packages
library(modeltime)
library(modeltime.ensemble)
# load reshape models
library(tidymodels)

library(tidyverse)
library(timetk)

#Load my data
tdf<- read.csv("D:/KaggleX/train.csv", header= TRUE)

summary(tdf)

#tdf$family<- as.factor(tdf$family)

# Convert the date column to a Date type
tdf$date <- as.Date(tdf$date)

attach(tdf)

tdf_new<- tdf %>% select(c(date, id, store_nbr,family, sales, onpromotion))


# checking for NAs and Missing data
sum(is.na(tdf_new))

# Visualizing 4 items
#selecting 4 items
sku4 <- tdf_new %>% 
  filter(family %in% c("AUTOMOTIVE", "BABY CARE", "HARDWARE", "LADIESWEAR"))

library(lubridate)

# Ensure the 'date' column is in a Date format
#sku4$date <- as.Date(sku4$date)

library(ggplot2)

#plot the series using ggplot2

sku4 %>%
  ggplot(aes(x = date, y = sales)) +
  geom_line() +
  facet_wrap(~ family, ncol = 4) +
  labs(title = "Time Series Plot for Different Families", x = "Date", y = "Sales") +
  theme_minimal()

#Create a Nested Time Series

nested_sku<- tdf_new %>% 
  group_by(as.factor(family))%>%
  extend_timeseries(
    .id_var = family,
    .date_var = date,
    .length_future = 30
    ) %>% 
  nest_timeseries(
    .id_var = family,
    .length_future = 30) %>%
  split_nested_timeseries(
    .length_test = 30)

nested_sku %>% tail()  

#Data Processing

#Create Recipes

rcp_xgb<- recipe(sales~., extract_nested_train_split(nested_sku)) %>%
  step_timeseries_signature(date)%>%
  step_rm(date)%>% 
  step_rm(all_predictors()) %>% 
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

#create a sample to see the recipe outcome
bake(prep(rcp_xgb),extract_nested_train_split(nested_sku))


#XGBoost Models (Using different Learning rates)
#LR = 0.35
sku_xg1 <- workflow() %>%
  add_model(boost_tree("regression",learn_rate =0.35)) %>%
  add_recipe(rcp_xgb)

#LR =0.50
sku_xg2 <- workflow() %>%
  add_model(boost_tree("regression", learn_rate= 0.50)) %>%
  add_recipe(rcp_xgb)

# Using THIEF- Temporal Hierarchical Forecasting

library(thief)

#Thief workflow
sku_thf <- workflow() %>%
  add_model(temporal_hierarchy() %>% set_engine("thief"))%>%
  add_recipe(recipe(sales~.,extract_nested_train_split(nested_sku)))


# Radial Basis Function (RBF) SVM
#library(kernlab)

#sku_rbf <- workflow() %>%
  #add_model(svm_rbf() %>% set_engine("kernlab")) %>%
  #add_recipe(rcp_xgb)


#Testing the Workflows With only 1 item to identify and debug errors

xample_sku<- nested_sku %>%
  slice(1) %>%
  modeltime_nested_fit(
    
    model_list = list(
      sku_xg1,
      sku_xg2,
      sku_thf
    ),
    control = control_nested_fit(
      verbose = TRUE,
      allow_par = FALSE
    )
  )

glimpse(tdf_new)

error_report <- extract_nested_error_report(xample_sku)
print(error_report)

