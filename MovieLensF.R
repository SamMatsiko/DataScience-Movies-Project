
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
            
rm(dl, ratings, movies, test_index, temp, movielens, removed)

#Partioning the edx set into train and test sets

test_index <- createDataPartition(y = edx$rating, times = 1,
                                  p = 0.5, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

# To make sure userId and movieId in train set are also in test set
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

#Taking the average rating of all users in the train set as an approximation of every users' rating
mu<-mean(train_set$rating)

#Determining how the average of the ratings performs against the test set ratings using RMSE function
naivemodel<-RMSE(test_set$rating,mu)

#Creating a table to show values as we move along
rmse_results <- tibble(method = "Just the average", RMSE = naivemodel)
rmse_results

#Training, predicting and comparing on considering the "movie effect"
movie_avgs <- train_set %>% group_by(movieId) %>% summarize(b_i=mean(rating - mu))

predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

model_bi_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie Effect Model",
                                 RMSE = model_bi_rmse ))
rmse_results %>% knitr::kable()

#Training, predicting and comparing on considering both the "movie effect" and "user effect"
user_avgs <- train_set %>% left_join(movie_avgs, by='movieId') %>% group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

model_bu_rmse<-RMSE(predicted_ratings,  test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie + User Effects Model",  
                                 RMSE = model_bu_rmse ))
rmse_results %>% knitr::kable()



#Deploying regularization to penalize the large mistakes observed whenever a movie is rated by very few users

#Determining the optimal tuning parameter(lambda) by cross-validation 
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})

lambda <- lambdas[which.min(rmses)]

#Training, predicting and comparing using regularization for both "movie effect" and "user effect"
movie_reg_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda)) 


user_reg_avgs <- train_set %>% left_join(movie_avgs, by='movieId')%>% group_by(userId) %>% 
  summarize(b_u = sum(rating - mu-b_i)/(n()+lambda)) 

predicted_ratings <- test_set %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  left_join(user_reg_avgs,by='userId') %>%
  mutate(pred = mu + b_i+ b_u) %>%
  .$pred

model_reg_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularization for both Movie and user Effects",  
                                     RMSE = model_reg_rmse ))
rmse_results %>% knitr::kable()


#Since the RMSE obtained is satisfactory, the edx set is now retrained as done with the train set

#The average rating for the edx set
mu<-mean(edx$rating) 

#Training the edx set for the movie effect
movie_avgs <- edx %>% group_by(movieId) %>% summarize(b_i=mean(rating - mu))


#Training the edx set for the user effect
user_avgs <- edx %>% left_join(movie_avgs, by='movieId') %>% group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))


#Regularization of edx set
#Regularization Training of the movie effect
movie_reg_avgs <- edx %>% group_by(movieId) %>% summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

#Regularization Training of both the movie and user effects
user_reg_avgs <- edx %>% left_join(movie_avgs, by='movieId')%>% group_by(userId) %>% 
  summarize(b_u = sum(rating - mu-b_i)/(n()+lambda), n_i = n()) 

#Predicting the ratings of the validation set
predicted_ratings <- validation %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  left_join(user_reg_avgs,by='userId') %>%
  mutate(pred = mu + b_i+ b_u) %>%
  .$pred

#Comparing the predicated ratings of the validation set and actual ratings
model_val_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Test with the validation set",  
                                 RMSE = model_val_rmse ))
rmse_results %>% knitr::kable()



 