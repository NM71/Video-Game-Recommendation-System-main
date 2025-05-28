#!/usr/bin/env python
# coding: utf-8

# ## Video Games Recommendation System
# 
# The aim of the project is to offer recommendations for video games based on a particular game title and platform <em>(optional)</em> as the input. This initiative will benefit individuals who are interested in exploring and finding new games.
# 
# ### Importing and Transforming Dataset

# In[39]:


get_ipython().run_line_magic('pip', 'install seaborn')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")


# In[40]:


def getFileAbsolutePath(filename):
    if '__file__' in globals():
        # Get the directory of the script file
        nb_dir = os.path.dirname(os.path.abspath(__file__)) 
    else:
        # Get the directory of the notebook
        nb_dir = os.getcwd() 

    data_dir = os.path.join(nb_dir, 'Dataset')
    data_file = os.path.join(data_dir, filename)

    return data_file


# The dataset was obtained from <a href="https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings">Video Game Sales with Ratings</a> in Kaggle, which were web scraped by Gregory Smith from VGChartz Video Games Sales. The collection of data includes details such as the game's title, genre, the platform it runs on, the company that published it, and other relevant information. From year 1980 up to 2020, the dataset includes a wide range of video game releases that spans over four decades.

# In[41]:


data_file = getFileAbsolutePath('Video Games Sales.csv')
video_games_df = pd.read_csv(data_file)

print(f"No. of records: {video_games_df.shape[0]}")
video_games_df.head(5)


# We selected only the features that are relevant for our recommendation system.

# In[42]:


video_games_filtered_df = video_games_df[['Name', 'Platform', 'Genre', 'Critic_Score', 'User_Score', 'Rating']]
video_games_filtered_df.info()


# ### Exploratory Data Analysis
# 
# Check the total number of missing values for each feature in the dataset 

# In[43]:


video_games_filtered_df.isna().sum().sort_values(ascending=False)


# Remove the records with missing data in the `Name`, `Genre` and `Ratings` features.

# In[44]:


# Remove missing values
video_games_filtered_df.dropna(subset=['Name', 'Genre', 'Rating'], axis=0, inplace=True)
video_games_filtered_df = video_games_filtered_df.reset_index(drop=True)

video_games_filtered_df[['Name', 'Genre', 'Rating']].isna().sum()


# Examine the frequency of data types for each categorical feature: `Genre`, `Platform`, and `Rating`.

# In[45]:


features = video_games_filtered_df[['Genre', 'Platform', 'Rating']].columns

for idx, feature in enumerate(features):
    plt.figure(figsize = (13,4))
    sns.countplot(data=video_games_filtered_df, x=feature)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(" Data Distribution of Video Game " + feature + "s")
plt.show()


# From the charts above, we can say that there is a scarcity of data available for certain platforms such as DC, and certain ratings such as 'K-A', 'AO’, 'EC' and 'RP'. 

# Create additional features that correspond to `User_Score` and `Critic_score` variables. Replace all missing and 'tbd' values with a specific value -- the imputed data is calculated as the mean value of the respective feature within a particular genre, e.g. the average of all scores under the 'Action' category.

# In[46]:


# Replace 'tbd' value to NaN
video_games_filtered_df['User_Score'] = np.where(video_games_filtered_df['User_Score'] == 'tbd', 
                                                 np.nan, 
                                                 video_games_filtered_df['User_Score']).astype(float)

# Group the records by Genre, then aggregate them calculating the average of both Critic Score and User Score
video_game_grpby_genre = video_games_filtered_df[['Genre', 'Critic_Score', 'User_Score']].groupby('Genre', as_index=False)
video_game_score_mean = video_game_grpby_genre.agg(Ave_Critic_Score = ('Critic_Score', 'mean'), Ave_User_Score = ('User_Score', 'mean'))

# Merge the average scores with the main dataframe
video_games_filtered_df = video_games_filtered_df.merge(video_game_score_mean, on='Genre')
video_games_filtered_df


# In[47]:


video_games_filtered_df['Critic_Score_Imputed'] = np.where(video_games_filtered_df['Critic_Score'].isna(), 
                                                           video_games_filtered_df['Ave_Critic_Score'], 
                                                           video_games_filtered_df['Critic_Score'])

video_games_filtered_df['User_Score_Imputed'] = np.where(video_games_filtered_df['User_Score'].isna(), 
                                                         video_games_filtered_df['Ave_User_Score'], 
                                                         video_games_filtered_df['User_Score'])
video_games_filtered_df


# Compare the summary statistics of `User_Score` and `Critic_Score` and the new feature with imputed values, i.e.`User_Score_Imputed` and `Critic_Score_Imputed`. The results below show that filling in missing values has no significant impact on the average and the standard deviation.

# In[48]:


video_games_filtered_df[['Critic_Score', 'Critic_Score_Imputed', 'User_Score', 'User_Score_Imputed']].describe()


# Drop all the fields related to critic and user scores except for the new features with imputed values.

# In[49]:


video_games_final_df = video_games_filtered_df.drop(columns=['User_Score', 'Critic_Score', 'Ave_Critic_Score', 'Ave_User_Score'], axis=1)
video_games_final_df = video_games_final_df.reset_index(drop=True)

video_games_final_df = video_games_final_df.rename(columns={'Critic_Score_Imputed':'Critic_Score', 'User_Score_Imputed':'User_Score'})
video_games_final_df.info()


# Analyze the data distribution for `Critic_Score` and `User_Score`, and assess the correlation between these two features.

# In[50]:


hist, bins = np.histogram(video_games_final_df['Critic_Score'], bins=10, range=(0, 100))

plt.figure(figsize = (8,4))
plt.bar(bins[:-1], hist, width=(bins[1]-bins[0]), align='edge')
plt.xlabel('Critic Score')
plt.ylabel('Frequency')
plt.title("Data Distribution of Critic Scores")
plt.show()


# In[51]:


hist, bins = np.histogram(video_games_final_df['User_Score'], bins=10, range=(0, 10))

plt.figure(figsize = (8,4))
plt.bar(bins[:-1], hist, width=(bins[1]-bins[0]), align='edge')
plt.xlabel('User Score')
plt.ylabel('Frequency')
plt.title("Data Distribution of User Scores")
plt.show()


# In[52]:


plt.figure(figsize=(8, 8))
ax = sns.regplot(x=video_games_final_df['User_Score'], y=video_games_final_df['Critic_Score'], 
                 line_kws={"color": "black"}, scatter_kws={'s': 4})
ax.set(xlabel ="User Score", ylabel = "Critic Score", title="User Scores vs. Critic Scores")


# Display the dataframe information to quickly understand its structure and characteristics.

# In[53]:


video_games_final_df.info()


# ### Converting Categorical Features to Dummy Indicators
# 
# Obtain all categorical features, except for the title of the game.

# In[54]:


categorical_columns = [name for name in video_games_final_df.columns if video_games_final_df[name].dtype=='O']
categorical_columns = categorical_columns[1:]

print(f'There are {len(categorical_columns)} categorical features:\n')
print(", ".join(categorical_columns))


# Transform all categorical attributes into binary dummy variables where the value is 0 (representing No) or 1 (representing Yes).

# In[55]:


video_games_df_dummy = pd.get_dummies(data=video_games_final_df, columns=categorical_columns)
video_games_df_dummy.head(5)


# After the conversion, the variables have expanded from the original 6 columns to a total of 40 columns.

# In[56]:


video_games_df_dummy.info()


# ### Standardizing the Numerical Features
# 
# Transform numerical data to a standardized form by scaling them to have a mean of 0 and a standard deviation of 1. The purpose of standardization is to ensure that all features are on a similar scale and have equal importance in determining the output variable.

# In[57]:


features = video_games_df_dummy.drop(columns=['Name'], axis=1)

scale = StandardScaler()
scaled_features = scale.fit_transform(features)
scaled_features = pd.DataFrame(scaled_features, columns=features.columns)

scaled_features.head(5)


# ### Creating a Model
# 
# The machine learning algorithm `NearestNeighbors` will be utilized to identify the data points nearest to a given input, with the aid of the `cosine similarity` measurement to determine the similarity or dissimilarity between data points.

# In[58]:


model = NearestNeighbors(n_neighbors=11, metric='cosine', algorithm='brute').fit(scaled_features)
print(model)


# As we included `n_neighbors=1` as a parameter for our model, it will generate 11 indices and distances of games that are similar to the user input, including the input itself.

# In[59]:


vg_distances, vg_indices = model.kneighbors(scaled_features)

print("List of indexes and distances for the first 5 games:\n")
print(vg_indices[:5], "\n")
print(vg_distances[:5])


# `TfidfVectorizer` is a feature extraction method commonly used in natural language processing and information retrieval tasks. In this case, it is used to suggest a video game title based on the user input (i.e. game that doesn't exist in the records) by evaluating the importance of words in the input relative to the existing records.

# In[60]:


game_names = video_games_df_dummy['Name'].drop_duplicates()
game_names = game_names.reset_index(drop=True)

vectorizer = TfidfVectorizer(use_idf=True).fit(game_names)
print(vectorizer)


# In[61]:


game_title_vectors = vectorizer.transform(game_names)

print("List of game title vectors for the first 5 games:\n")
print(pd.DataFrame(game_title_vectors.toarray()).head(5))


# ### Evaluating the Model
# 
# The program utilizes the above-mentioned model to provide video game recommendations to users. It will ask user to enter the game's name and, optionally, the platform to filter the results. The list of recommended games will be arranged in ascending order based on the calculated distances. On the other hand, if the game's name is not in the record, the program will suggest a new name of the game that has the closest match to the input.

# In[66]:


def VideoGameTitleRecommender(video_game_name):
    '''
    This function will recommend a game title that has the closest match to the input
    '''
    query_vector = vectorizer.transform([video_game_name])
    similarity_scores = cosine_similarity(query_vector, game_title_vectors)

    closest_match_index = similarity_scores.argmax()
    closest_match_game_name = game_names[closest_match_index]

    return closest_match_game_name


def VideoGameRecommender(video_game_name, video_game_platform='Any'):
    '''
    This function will provide game recommendations based on various features of the game
    '''
    default_platform = 'Any'

    # User input: Game Title and Platform
    if video_game_platform != default_platform:
        video_game_idx = video_games_final_df.query("Name == @video_game_name & Platform == @video_game_platform").index

        if video_game_idx.empty:
            video_game_idx = video_games_final_df.query("Name == @video_game_name").index

            if not video_game_idx.empty:
                print(f"Note: Recommendations will be based on the title of the game as it is not available on the specified platform.\n")
                video_game_platform = default_platform

    # User input: Game Title only
    else:
        video_game_idx = video_games_final_df.query("Name == @video_game_name").index  

    if video_game_idx.empty:
        # If the game entered by the user doesn't exist in the records, the program will recommend a new game similar to the input
        closest_match_game_name = VideoGameTitleRecommender(video_game_name)

        print(f"'{video_game_name}' doesn't exist in the records.\n")
        print(f"You may want to try '{closest_match_game_name}', which is the closest match to the input.")

    else:
        # User input: Game Title only
        if video_game_platform == default_platform:

            # Place in a separate dataframe the indices and distances, then sort the record by distance in ascending order       
            vg_combined_dist_idx_df = pd.DataFrame()
            for idx in video_game_idx:
                # Remove from the list any game that shares the same name as the input
                vg_dist_idx_df = pd.concat([pd.DataFrame(vg_indices[idx][1:]), pd.DataFrame(vg_distances[idx][1:])], axis=1)
                vg_combined_dist_idx_df = pd.concat([vg_combined_dist_idx_df, vg_dist_idx_df])

            vg_combined_dist_idx_df = vg_combined_dist_idx_df.set_axis(['Index', 'Distance'], axis=1)
            vg_combined_dist_idx_df = vg_combined_dist_idx_df.reset_index(drop=True)
            vg_combined_dist_idx_df = vg_combined_dist_idx_df.sort_values(by='Distance', ascending=True)

            video_game_list = video_games_final_df.iloc[vg_combined_dist_idx_df['Index']]

            # Remove any duplicate game names to provide the user with a diverse selection of recommended games
            video_game_list = video_game_list.drop_duplicates(subset=['Name'], keep='first')

            # Get the first 10 games in the list
            video_game_list = video_game_list.head(10)

            # Get the distance of the games similar to the input
            recommended_distances = np.array(vg_combined_dist_idx_df['Distance'].head(10))

        # User input: Game Title and Platform
        else:
            # Remove from the list any game that shares the same name as the input
            recommended_idx = vg_indices[video_game_idx[0]][1:]
            video_game_list = video_games_final_df.iloc[recommended_idx]

            # Get the distance of the games similar to the input
            recommended_distances = np.array(vg_distances[video_game_idx[0]][1:])

        print(f"Top 10 Recommended Video Games for '{video_game_name}' [platform:{video_game_platform}]")

        video_game_list = video_game_list.reset_index(drop=True)
        recommended_video_game_list = pd.concat([video_game_list, 
                                                 pd.DataFrame(recommended_distances, columns=['Similarity_Distance'])], axis=1)

        # display(recommended_video_game_list.style.hide_index())
        display(recommended_video_game_list.style.hide(axis="index"))


# __TEST CASE #1__
# 
# __Input:__ Video Game Title</br>
# __Expected Result:__ The program merges recommendations from all platforms of the game, arranges the similiarity distances in ascending order, then displays only the first 10 games that has the smallest calculated distance.

# In[67]:


VideoGameRecommender('Call of Duty: World at War')


# __TEST CASE #2__
# 
# __Input:__ Video Game Title and Platform</br>
# __Expected Result:__ The platform helps to limit the results and display only recommended games based on the specified game and platform.
# 
# NOTE: If a platform has limited data like DC, the program might suggest games from other platforms based on various factors when calculating the features similarity.

# In[68]:


VideoGameRecommender('Call of Duty: World at War', 'PC')


# __TEST CASE #3__
# 
# __Input:__ Video Game Title and Platform</br>
# __Constraint:__ Video game is not available on the specified platform</br>
# __Expected Result:__ Since the video game is not available on the specified platform, the recommendation is based solely on the game title and ignores the platform.

# In[69]:


VideoGameRecommender('Call of Duty: World at War', 'XB')


# __TEST CASE #4__
# 
# __Input:__ Video Game Title</br>
# __Constraint:__ Video game is not available in the records</br>
# __Expected Result:__ No recommendation is shown but the program provides the user with the game title that has closest match to the input.

# In[70]:


VideoGameRecommender('Call of Duty')


# ### Assumptions
# - Removed records with missing values in `Name`, `Genre` and `Rating` features
# - Conducted data-imputation on missing and 'tbd' values in `User_Score` and `Critic_Score` features. The imputed data was calculated as the mean value of the `User_Score` or `Critic_score` variable within a particular genre, e.g. the average of all scores under the 'Action' category.

# ### References
# 
# 1. https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings
# 2. https://thecleverprogrammer.com/2021/01/17/book-recommendation-system/
# 3. https://aman-makwana101932.medium.com/understanding-recommendation-system-and-knn-with-project-book-recommendation-system-c648e47ff4f6
# 4. https://www.aurigait.com/blog/recommendation-system-using-knn/
# 5. https://towardsdatascience.com/cosine-similarity-how-does-it-measure-the-similarity-maths-behind-and-usage-in-python-50ad30aad7db

# > BDM-3014 Winter 2023 Project (Group 12)
