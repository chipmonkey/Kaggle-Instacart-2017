###########################################################################################################
#
# Borrowed From: https://www.kaggle.com/fabienvs/instacart-xgboost-starter-lb-0-3791/code
# And From: https://github.com/sh1ng/imba/
# ---Chipmonkey
#
###########################################################################################################

options(scipen = 999) # Mostly disable scientific notation in display and file writes
options(stringsAsFactors=F)   # Disable conversion of strings to factors on read.  Sometimes useful.
rm(list = ls())

gc() ; Sys.time() ; start_time <- Sys.time()  # Run this at the beginning

setwd("./scripts")  # I build projects at the root level; this can fail without issue

library(tidyverse)
# library(tidytext)
# library(Matrix)

# Load Data ---------------------------------------------------------------
aisles <- read_csv('../input/aisles.csv.zip')
departments <- read_csv('../input/departments.csv.zip')
orderp <- read_csv('../input/order_products__prior.csv.zip')
ordert <- read_csv('../input/order_products__train.csv.zip')
orders <- read_csv('../input/orders.csv.zip')
products <- read_csv('../input/products.csv.zip')
ss <- read_csv('../input/sample_submission.csv.zip')

gc() ; Sys.time() - start_time ; print("finished reading files")

# https://github.com/sh1ng/imba/blob/master/create_products.py:

my_opu <- orders[orders$eval_set == 'prior', c('order_id', 'user_id')]
user_product <- orderp %>%
                inner_join(my_opu, by = 'order_id') %>% 
                select(user_id, product_id) %>% 
                unique()

gc() ; Sys.time() - start_time ; print("finished user previous_products")

my_odtt <- orders[orders$eval_set %in% c('train', 'test'),]

user_product <- user_product %>%
                inner_join(my_odtt[,c('order_id', 'user_id', 'eval_set')], by='user_id') %>%
                select(-user_id)

ordert <- select(ordert, -add_to_cart_order)

gc() ; Sys.time() - start_time ; print("data is loaded")

oids <- unique(user_product$order_id)

# The example has a folding process here to split files, but let's just do it all at once.
# We have petabytes of RAM, right?

order_x <- right_join(ordert, user_product, by=c('order_id', 'product_id'))
order_x[is.na(order_x)] <- 0

# Add some summarization...
gc() ; Sys.time() - start_time ; print("data is loaded")

order_cumsum <- orders %>% group_by(user_id, order_number) %>%
      summarize(days_since_prior_order_cumsum <- cumsum(days_since_prior_order))
  
  # summarise(
  #   days_since_prior_order_comsum = sum(days_since_prior_order),
  # )
