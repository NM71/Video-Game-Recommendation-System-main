import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="Video Game Recommender",
    page_icon="🎮",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 1.5rem;
        color: #2196F3;
        margin-top: 20px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>🎮 Video Game Recommendation System</h1>", unsafe_allow_html=True)

# Function to get file path
def getFileAbsolutePath(filename):
    if '__file__' in globals():
        # Get the directory of the script file
        nb_dir = os.path.dirname(os.path.abspath(__file__)) 
    else:
        # Get the directory of the notebook/app
        nb_dir = os.getcwd() 

    data_dir = os.path.join(nb_dir, 'Dataset')
    data_file = os.path.join(data_dir, filename)

    return data_file

# Load and process data
@st.cache_data
def load_and_process_data():
    try:
        # Load data
        data_file = getFileAbsolutePath('Video Games Sales.csv')
        video_games_df = pd.read_csv(data_file)
        
        # Filter columns
        video_games_filtered_df = video_games_df[['Name', 'Platform', 'Genre', 'Critic_Score', 'User_Score', 'Rating']]
        
        # Remove missing values
        video_games_filtered_df.dropna(subset=['Name', 'Genre', 'Rating'], axis=0, inplace=True)
        video_games_filtered_df = video_games_filtered_df.reset_index(drop=True)
        
        # Replace 'tbd' value to NaN
        video_games_filtered_df['User_Score'] = np.where(video_games_filtered_df['User_Score'] == 'tbd', 
                                                        np.nan, 
                                                        video_games_filtered_df['User_Score']).astype(float)

        # Group by Genre and calculate average scores
        video_game_grpby_genre = video_games_filtered_df[['Genre', 'Critic_Score', 'User_Score']].groupby('Genre', as_index=False)
        video_game_score_mean = video_game_grpby_genre.agg(Ave_Critic_Score = ('Critic_Score', 'mean'), Ave_User_Score = ('User_Score', 'mean'))

        # Merge the average scores with the main dataframe
        video_games_filtered_df = video_games_filtered_df.merge(video_game_score_mean, on='Genre')
        
        # Impute missing values
        video_games_filtered_df['Critic_Score_Imputed'] = np.where(video_games_filtered_df['Critic_Score'].isna(), 
                                                                video_games_filtered_df['Ave_Critic_Score'], 
                                                                video_games_filtered_df['Critic_Score'])

        video_games_filtered_df['User_Score_Imputed'] = np.where(video_games_filtered_df['User_Score'].isna(), 
                                                                video_games_filtered_df['Ave_User_Score'], 
                                                                video_games_filtered_df['User_Score'])
        
        # Create final dataframe
        video_games_final_df = video_games_filtered_df.drop(columns=['User_Score', 'Critic_Score', 'Ave_Critic_Score', 'Ave_User_Score'], axis=1)
        video_games_final_df = video_games_final_df.reset_index(drop=True)
        video_games_final_df = video_games_final_df.rename(columns={'Critic_Score_Imputed':'Critic_Score', 'User_Score_Imputed':'User_Score'})
        
        return video_games_final_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Build recommendation model
@st.cache_resource
def build_recommendation_model(video_games_final_df):
    try:
        # Get categorical columns
        categorical_columns = [name for name in video_games_final_df.columns if video_games_final_df[name].dtype=='O']
        categorical_columns = categorical_columns[1:]  # Exclude 'Name'
        
        # Convert to dummy variables
        video_games_df_dummy = pd.get_dummies(data=video_games_final_df, columns=categorical_columns)
        
        # Standardize numerical features
        features = video_games_df_dummy.drop(columns=['Name'], axis=1)
        scale = StandardScaler()
        scaled_features = scale.fit_transform(features)
        scaled_features = pd.DataFrame(scaled_features, columns=features.columns)
        
        # Build NearestNeighbors model
        model = NearestNeighbors(n_neighbors=11, metric='cosine', algorithm='brute').fit(scaled_features)
        
        # Get distances and indices
        vg_distances, vg_indices = model.kneighbors(scaled_features)
        
        # Build TF-IDF vectorizer for game titles
        game_names = video_games_df_dummy['Name'].drop_duplicates()
        game_names = game_names.reset_index(drop=True)
        vectorizer = TfidfVectorizer(use_idf=True).fit(game_names)
        game_title_vectors = vectorizer.transform(game_names)
        
        return {
            'model': model,
            'vg_distances': vg_distances,
            'vg_indices': vg_indices,
            'video_games_df_dummy': video_games_df_dummy,
            'video_games_final_df': video_games_final_df,
            'vectorizer': vectorizer,
            'game_names': game_names,
            'game_title_vectors': game_title_vectors
        }
    except Exception as e:
        st.error(f"Error building model: {e}")
        return None

# Function to recommend a game title
def VideoGameTitleRecommender(video_game_name, model_data):
    '''
    This function will recommend a game title that has the closest match to the input
    '''
    vectorizer = model_data['vectorizer']
    game_names = model_data['game_names']
    game_title_vectors = model_data['game_title_vectors']
    
    query_vector = vectorizer.transform([video_game_name])
    similarity_scores = cosine_similarity(query_vector, game_title_vectors)

    closest_match_index = similarity_scores.argmax()
    closest_match_game_name = game_names[closest_match_index]

    return closest_match_game_name

# Function to recommend video games
def VideoGameRecommender(video_game_name, video_game_platform, model_data):
    '''
    This function will provide game recommendations based on various features of the game
    '''
    video_games_final_df = model_data['video_games_final_df']
    vg_indices = model_data['vg_indices']
    vg_distances = model_data['vg_distances']
    
    default_platform = 'Any'

    # User input: Game Title and Platform
    if video_game_platform != default_platform:
        video_game_idx = video_games_final_df.query("Name == @video_game_name & Platform == @video_game_platform").index

        if video_game_idx.empty:
            video_game_idx = video_games_final_df.query("Name == @video_game_name").index

            if not video_game_idx.empty:
                st.info(f"Note: Recommendations will be based on the title of the game as it is not available on the specified platform.")
                video_game_platform = default_platform

    # User input: Game Title only
    else:
        video_game_idx = video_games_final_df.query("Name == @video_game_name").index  

    if video_game_idx.empty:
        # If the game entered by the user doesn't exist in the records, the program will recommend a new game similar to the input
        closest_match_game_name = VideoGameTitleRecommender(video_game_name, model_data)

        st.warning(f"'{video_game_name}' doesn't exist in the records.")
        st.info(f"You may want to try '{closest_match_game_name}', which is the closest match to the input.")
        return None

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

        st.markdown(f"<h2 class='subheader'>Top 10 Recommended Video Games for '{video_game_name}' [platform:{video_game_platform}]</h2>", unsafe_allow_html=True)

        video_game_list = video_game_list.reset_index(drop=True)
        recommended_video_game_list = pd.concat([video_game_list, 
                                                pd.DataFrame(recommended_distances, columns=['Similarity_Distance'])], axis=1)
        
        return recommended_video_game_list

# Main app
def main():
    # Load data
    with st.spinner("Loading and processing data..."):
        video_games_final_df = load_and_process_data()
    
    if video_games_final_df is None:
        st.error("Failed to load the game database. Please check if the CSV file exists in the Dataset folder.")
        return
    
    # Build model
    with st.spinner("Building recommendation model..."):
        model_data = build_recommendation_model(video_games_final_df)
    
    if model_data is None:
        st.error("Failed to build the recommendation model.")
        return
    
    # Sidebar for inputs
    st.sidebar.header("Find Your Next Game")
    
    # Get unique game names for dropdown
    game_names = sorted(video_games_final_df['Name'].unique())
    
    # User input method
    input_method = st.sidebar.radio("Select input method:", ["Dropdown", "Text Input"])
    
    if input_method == "Dropdown":
        game_name = st.sidebar.selectbox("Select a game:", game_names)
    else:
        game_name = st.sidebar.text_input("Enter a game title:", "Call of Duty: World at War")
    
    # Platform selection
    platforms = sorted(video_games_final_df['Platform'].unique())
    platform = st.sidebar.selectbox("Select platform (optional):", ["Any"] + list(platforms))
    
    # Get recommendations
    if st.sidebar.button("Get Recommendations"):
        with st.spinner("Finding similar games..."):
            recommendations = VideoGameRecommender(game_name, platform, model_data)
        
        if recommendations is not None:
            # Display recommendations
            st.dataframe(
                recommendations.style.format({
                    'Critic_Score': '{:.1f}',
                    'User_Score': '{:.1f}',
                    'Similarity_Distance': '{:.4f}'
                }),
                use_container_width=True
            )
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.header("About")
    st.sidebar.info("""
    This recommendation system uses machine learning to find similar games based on:
    - Genre
    - Platform
    - Rating
    - Critic and User Scores
    
    Data source: [Kaggle - Video Game Sales with Ratings](https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings)
    """)
    
    # Instructions
    st.markdown("""
    ## How to Use
    1. Select a game from the dropdown or enter a game title
    2. Optionally select a specific platform
    3. Click "Get Recommendations" to find similar games
    
    ## About the System
    This recommendation system uses machine learning to analyze game features and find similar titles based on:
    - Game genre
    - Platform
    - ESRB rating
    - Critic and user scores
    
        The system uses the K-Nearest Neighbors algorithm with cosine similarity to find games that are most similar to your selection.
    
    ## Dataset
    The dataset contains information about video games including:
    - Game titles
    - Platforms
    - Genres
    - Critic scores from Metacritic
    - User scores from Metacritic
    - ESRB ratings
    
    The data was obtained from [Kaggle - Video Game Sales with Ratings](https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings).
    """)
    
    # Add data visualization section
    if st.checkbox("Show Data Visualizations"):
        st.markdown("<h2 class='subheader'>Data Visualizations</h2>", unsafe_allow_html=True)
        
        viz_type = st.selectbox("Select visualization:", 
                               ["Genre Distribution", "Platform Distribution", "Rating Distribution", 
                                "Critic Score Distribution", "User Score Distribution", "Critic vs User Scores"])
        
        if viz_type == "Genre Distribution":
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(data=video_games_final_df, x='Genre', ax=ax)
            plt.xticks(rotation=45, ha='right')
            plt.title("Distribution of Video Game Genres")
            plt.tight_layout()
            st.pyplot(fig)
            
        elif viz_type == "Platform Distribution":
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.countplot(data=video_games_final_df, x='Platform', ax=ax)
            plt.xticks(rotation=45, ha='right')
            plt.title("Distribution of Video Game Platforms")
            plt.tight_layout()
            st.pyplot(fig)
            
        elif viz_type == "Rating Distribution":
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(data=video_games_final_df, x='Rating', ax=ax)
            plt.title("Distribution of Video Game Ratings")
            plt.tight_layout()
            st.pyplot(fig)
            
        elif viz_type == "Critic Score Distribution":
            fig, ax = plt.subplots(figsize=(10, 6))
            hist, bins = np.histogram(video_games_final_df['Critic_Score'], bins=10, range=(0, 100))
            plt.bar(bins[:-1], hist, width=(bins[1]-bins[0]), align='edge')
            plt.xlabel('Critic Score')
            plt.ylabel('Frequency')
            plt.title("Distribution of Critic Scores")
            st.pyplot(fig)
            
        elif viz_type == "User Score Distribution":
            fig, ax = plt.subplots(figsize=(10, 6))
            hist, bins = np.histogram(video_games_final_df['User_Score'], bins=10, range=(0, 10))
            plt.bar(bins[:-1], hist, width=(bins[1]-bins[0]), align='edge')
            plt.xlabel('User Score')
            plt.ylabel('Frequency')
            plt.title("Distribution of User Scores")
            st.pyplot(fig)
            
        elif viz_type == "Critic vs User Scores":
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.regplot(x=video_games_final_df['User_Score'], y=video_games_final_df['Critic_Score'], 
                       line_kws={"color": "black"}, scatter_kws={'s': 4}, ax=ax)
            ax.set(xlabel="User Score", ylabel="Critic Score", title="User Scores vs. Critic Scores")
            st.pyplot(fig)

if __name__ == '__main__':
    main()
