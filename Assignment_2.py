#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os as os

import scipy.sparse as sp
from scipy.sparse.linalg import svds


# In[24]:


movie = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
link = pd.read_csv('links.csv')
tag = pd.read_csv('tags.csv')


# In[25]:


mov


# In[26]:


rat


# In[27]:


master = mov.merge(rat)
master


# In[28]:


master.shape


# In[29]:


training_data = master.sample(frac=0.9)
testing_data = master.drop(training_data.index)

print(f"No. of training examples: {training_data.shape[0]}")
print(f"No. of testing examples: {testing_data.shape[0]}")


# In[30]:


training_data


# In[31]:


testing_data


# In[35]:


movie['movieId'] = movie['movieId'].apply(pd.to_numeric)
movie.head()


# In[36]:


ratings.head()


# In[66]:


Ratings = ratings.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
Ratings.head()


# In[78]:


Rating = Ratings.values
user_ratings_mean = np.mean(Rating, axis = 1)
Assume = Rating - user_ratings_mean.reshape(-1, 1)


# In[79]:


from scipy.sparse.linalg import svds
U, sigma, Vt = svds(Assume, k = 50)


# In[80]:


sigma = np.diag(sigma)


# In[81]:


all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = Ratings.columns)


# In[83]:


def recommend_movies(preds_df, userID, movie, master, num_recommendations=5):
    
    # Getting and sorting User predictions
    user_row_number = userID - 1 # UserID starts at 1
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False) # UserID starts from 1

    # Getting the user's data and merging it into the movie information.
    user_data = master[master.userId == (userID)]
    user_full = (user_data.merge(movie, how = 'left', left_on = 'movieId', right_on = 'movieId').
                     sort_values(['rating'], ascending=False)
                 )

    # Recommending the highest predicted rating movies that the user hasn't been watched yet.
    recommendations = (movie[~movie['movieId'].isin(user_full['movieId'])]).merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left', left_on = 'movieId',
               right_on = 'movieId').rename(columns = {user_row_number: 'Predictions'}).sort_values('Predictions', ascending = False).iloc[:num_recommendations, :-1]
                      

    return user_full, recommendations
already_rated, predictions = recommend_movies(preds_df, 330, movie, ratings, 18)
already_rated.head(18)
predictions


# In[ ]:




