
###########################################################################################################
#
# Borrowed From: https://www.kaggle.com/fabienvs/instacart-xgboost-starter-lb-0-3791/code
# ---Chipmonkey
#
###########################################################################################################

rm(list = ls())
setwd("./scripts")

# library(data.table)
# library(dplyr)
# library(tidyr)
library(tidyverse)
library(tidytext)
library(Matrix)

# Load Data ---------------------------------------------------------------
aisles <- read_csv('../input/aisles.csv.zip')
departments <- read_csv('../input/departments.csv.zip')
orderp <- read_csv('../input/order_products__prior.csv.zip')
ordert <- read_csv('../input/order_products__train.csv.zip')
orders <- read_csv('../input/orders.csv.zip')
products <- read_csv('../input/products.csv.zip')
ss <- read_csv('../input/sample_submission.csv.zip')


# Reshape data ------------------------------------------------------------
aisles$aisle <- as.factor(aisles$aisle)
departments$department <- as.factor(departments$department)
orders$eval_set <- as.factor(orders$eval_set)

products <- products %>% 
  inner_join(aisles) %>% inner_join(departments) %>% 
  select(-aisle_id, -department_id)
rm(aisles, departments)

ordert$user_id <- orders$user_id[match(ordert$order_id, orders$order_id)]

orders_products <- orders %>% inner_join(orderp, by = "order_id")

rm(orderp)
gc()


# Products ----------------------------------------------------------------
prd <- orders_products %>%
  arrange(user_id, order_number, product_id) %>%
  group_by(user_id, product_id) %>%
  mutate(product_time = row_number()) %>%
  ungroup() %>%
  group_by(product_id) %>%
  summarise(
    prod_orders = n(),
    prod_reorders = sum(reordered),
    prod_first_orders = sum(product_time == 1),
    prod_second_orders = sum(product_time == 2)
  )

prd_text <- products %>% unnest_tokens(words, product_name, token="words") %>%
   select('product_id', 'words')

prd_2gram <- products %>% unnest_tokens(words, product_name, token="ngrams", n=2) %>%
  select('product_id', 'words')

prd_tokens <- rbind(prd_text, prd_2gram)

topwords <- count(prd_tokens, words, sort = TRUE) %>%
  filter(!words %in% stop_words$word)
topwords <- topwords[topwords$n>500,] # ARBITRARY!  Tune this... Or use other mechanisms

prd_tokens_short <- prd_tokens[prd_tokens$words %in% topwords$words,] %>%
  filter(product_id %in% prd$product_id)  # Because some aren't.  That's weird

# rm(prd_text) ; rm(prd_2gram)

prd$prod_reorder_probability <- prd$prod_second_orders / prd$prod_first_orders
prd$prod_reorder_times <- 1 + prd$prod_reorders / prd$prod_first_orders
prd$prod_reorder_ratio <- prd$prod_reorders / prd$prod_orders

prd <- prd %>% select(-prod_reorders, -prod_first_orders, -prod_second_orders)

m_rows <-  prd$product_id
m_cols <- unique(prd_tokens_short$words)
m_row_ids <- match(prd_tokens_short$product_id, m_rows)
m_col_ids <- match(prd_tokens_short$words, m_cols)
m_x_y <- cbind(m_row_ids, m_col_ids)

wordhash <- matrix(0, nrow=length(m_rows), ncol=length(m_cols),
                   dimnames = list(as.character(m_rows),m_cols)) # , sparse = TRUE)

wordhash[m_x_y] <- 1
wordhash <- as_data_frame(wordhash)
# head(wordhash)

# nrow(prd)
# nrow(wordhash)
# prd[10000,]
# wordhash[10000,]
# prd_text[prd_text$product_id == 10003,]

# prdx <- as.matrix(prd)
# prdx <- cbind(prdx, wordhash)

prd <- cbind(prd, wordhash)

# rm(products)
gc()

# Users -------------------------------------------------------------------
users <- orders %>%
  filter(eval_set == "prior") %>%
  group_by(user_id) %>%
  summarise(
    user_orders = max(order_number),
    user_period = sum(days_since_prior_order, na.rm = T),
    user_mean_days_since_prior = mean(days_since_prior_order, na.rm = T)
  )

us <- orders_products %>%
  group_by(user_id) %>%
  summarise(
    user_total_products = n(),
    user_reorder_ratio = sum(reordered == 1) / sum(order_number > 1),
    user_distinct_products = n_distinct(product_id)
  )

users <- users %>% inner_join(us)
users$user_average_basket <- users$user_total_products / users$user_orders

us <- orders %>%
  filter(eval_set != "prior") %>%
  select(user_id, order_id, eval_set,
         time_since_last_order = days_since_prior_order)

users <- users %>% inner_join(us)

# usersx <- as.matrix(users, dimnames=list(as.character(users$user_id), colnames(users)))
# colnames(usersx)
# rownames(usersx)
# rownames(usersx) <- users$user_id  # why didn't dimnames work?

rm(us)
gc()


# Database ----------------------------------------------------------------
data <- orders_products %>%
  group_by(user_id, product_id) %>% 
  summarise(
    up_orders = n(),
    up_first_order = min(order_number),
    up_last_order = max(order_number),
    up_average_cart_position = mean(add_to_cart_order))

# rm(orders_products, orders)  # Later for memory, but not while debugging

data <- data %>% 
  inner_join(prd, by = "product_id") %>%
  inner_join(users, by = "user_id")

data$up_order_rate <- data$up_orders / data$user_orders
data$up_orders_since_last_order <- data$user_orders - data$up_last_order
data$up_order_rate_since_first_order <- data$up_orders / (data$user_orders - data$up_first_order + 1)

data <- data %>% 
  left_join(ordert %>% select(user_id, product_id, reordered), 
            by = c("user_id", "product_id"))

rm(ordert, prd, users)
gc()


# Train / Test datasets ---------------------------------------------------
train <- as.data.frame(data[data$eval_set == "train",])
train$eval_set <- NULL
train$user_id <- NULL
train$product_id <- NULL
train$order_id <- NULL
train$reordered[is.na(train$reordered)] <- 0

test <- as.data.frame(data[data$eval_set == "test",])
test$eval_set <- NULL
test$user_id <- NULL
test$reordered <- NULL

rm(data)
gc()


# Model -------------------------------------------------------------------
library(xgboost)

params <- list(
  "objective"           = "reg:logistic",
  "eval_metric"         = "logloss",
  "eta"                 = 0.08,
  "max_depth"           = 6,
  "min_child_weight"    = 11,
  "gamma"               = 0.70,
  "subsample"           = 0.76,
  "colsample_bytree"    = 0.95,
  "alpha"               = 2e-05,
  "lambda"              = 10
)

subtrain <- train %>% sample_frac(0.75)
X <- xgb.DMatrix(as.matrix(subtrain %>% select(-reordered)), label = subtrain$reordered)
model <- xgboost(data = X, params = params, nrounds = 150)

importance <- xgb.importance(colnames(X), model = model)
xgb.ggplot.importance(importance)

rm(X, importance, subtrain)
gc()


# Apply model -------------------------------------------------------------
X <- xgb.DMatrix(as.matrix(test %>% select(-order_id, -product_id)))
test$reordered <- predict(model, X)

test$reordered <- (test$reordered > 0.21) * 1

submission <- test %>%
  filter(reordered == 1) %>%
  group_by(order_id) %>%
  summarise(
    products = paste(product_id, collapse = " ")
  )

missing <- data.frame(
  order_id = unique(test$order_id[!test$order_id %in% submission$order_id]),
  products = "None"
)

submission <- submission %>% bind_rows(missing) %>% arrange(order_id)

file_timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")  # Generate a timestamp with 1-second accuracy
sub_filename <- paste('../submission_', file_timestamp, '.csv', sep="")  # Create a timestamp filename

write.csv(submission, file=sub_filename, row.names = F)
