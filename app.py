import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from difflib import get_close_matches

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# Initialize session state
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'last_search' not in st.session_state:
    st.session_state.last_search = None

# Set page config
st.set_page_config(
    page_title="Video Game Recommender",
    page_icon="🎮",
    layout="wide"
)

# Add custom CSS for both dark and light mode
st.markdown("""
<style>
    /* Common styles */
    .main-header {
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 1.5rem;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    
    /* Dark mode styles */
    @media (prefers-color-scheme: dark) {
        .main-header {
            color: #4CAF50;
        }
        .subheader {
            color: #2196F3;
        }
        .stMetric {
            background-color: rgba(30, 30, 30, 0.3);
            border: 1px solid rgba(150, 150, 150, 0.2);
        }
    }
    
    /* Light mode styles */
    @media (prefers-color-scheme: light) {
        .main-header {
            color: #2E7D32;
        }
        .subheader {
            color: #1565C0;
        }
        .stMetric {
            background-color: #f0f8ff;
            border: 1px solid #e0e0e0;
        }
    }
    
    /* Card styles for both modes */
    .game-card {
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        position: relative;
        transition: transform 0.2s, box-shadow 0.2s;
        cursor: pointer;
    }
    .game-card:hover {
        transform: translateY(-3px);
    }
    .game-title {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
        padding-right: 60px;
    }
    .game-platform, .game-genre {
        margin-bottom: 5px;
    }
    .game-rating {
        position: absolute;
        top: 15px;
        right: 15px;
        padding: 5px 10px;
        border-radius: 4px;
        font-weight: bold;
        color: white;
    }
    .rating-E {
        background-color: #4CAF50;
    }
    .rating-T {
        background-color: #2196F3;
    }
    .rating-M {
        background-color: #F44336;
    }
    .rating-E10 {
        background-color: #8BC34A;
    }
    .rating-other {
        background-color: #9E9E9E;
    }
    .game-scores {
        display: flex;
        justify-content: space-between;
        padding-top: 10px;
        margin-top: 5px;
    }
    .score-box {
        text-align: center;
    }
    .score-label {
        font-size: 12px;
    }
    .score-value {
        font-size: 16px;
        font-weight: bold;
    }
    .steam-link {
        text-decoration: none;
        color: inherit;
    }
    .steam-link:hover {
        text-decoration: none;
        color: inherit;
    }
    
    /* Dark mode card styles */
    @media (prefers-color-scheme: dark) {
        .game-card {
            background-color: rgba(50, 50, 50, 0.2);
            border: 1px solid rgba(150, 150, 150, 0.2);
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        }
        .game-card:hover {
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }
        .game-title {
            color: rgba(255, 255, 255, 0.9);
        }
        .game-platform, .game-genre {
            color: rgba(255, 255, 255, 0.7);
        }
        .game-scores {
            border-top: 1px solid rgba(150, 150, 150, 0.2);
        }
        .score-label {
            color: rgba(255, 255, 255, 0.6);
        }
        .score-value {
            color: rgba(255, 255, 255, 0.9);
        }
    }
    
    /* Light mode card styles */
    @media (prefers-color-scheme: light) {
        .game-card {
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .game-card:hover {
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        }
        .game-title {
            color: #333;
        }
        .game-platform, .game-genre {
            color: #555;
        }
        .game-scores {
            border-top: 1px solid #eee;
        }
        .score-label {
            color: #777;
        }
        .score-value {
            color: #333;
        }
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

# Enhanced search functionality
def enhanced_game_search(query, game_names, threshold=0.6):
    """Enhanced search with fuzzy matching"""
    # Exact match first
    if query in game_names:
        return query
    
    # Fuzzy matching
    matches = get_close_matches(query, game_names, n=5, cutoff=threshold)
    return matches

# Generate Steam search URL
def get_steam_search_url(game_name):
    """Generate a Steam store search URL for a game"""
    # Ensure game_name is a string
    game_name_str = str(game_name)
    
    # Format the game name for a URL (replace spaces with +)
    formatted_name = game_name_str.replace(' ', '+')
    return f"https://store.steampowered.com/search/?term={formatted_name}"

# Export recommendations to CSV
def export_recommendations(recommendations, game_name):
    """Export recommendations to CSV"""
    if recommendations is not None:
        csv = recommendations.to_csv(index=False)
        st.download_button(
            label="📥 Download Recommendations as CSV",
            data=csv,
            file_name=f"recommendations_for_{game_name.replace(' ', '_')}.csv",
            mime="text/csv"
        )

# Format value with fallback
def format_value(value, default="N/A"):
    """Format a value with a fallback for NaN values"""
    if pd.isna(value) or value == "nan":
        return default
    return value

# Load and process data
@st.cache_data
def load_and_process_data():
    try:
        # Load data
        data_file = getFileAbsolutePath('Video Games Sales.csv')
        
        # Check if file exists
        if not os.path.exists(data_file):
            st.error(f"Dataset file not found at: {data_file}")
            st.info("Please ensure 'Video Games Sales.csv' is in the 'Dataset' folder")
            return None
            
        video_games_df = pd.read_csv(data_file)
        
        # Filter columns
        video_games_filtered_df = video_games_df[['Name', 'Platform', 'Genre', 'Critic_Score', 'User_Score', 'Rating']]
        
        # Remove missing values
        video_games_filtered_df.dropna(subset=['Name', 'Genre'], axis=0, inplace=True)
        
        # Fill missing ratings with 'Not Rated'
        video_games_filtered_df['Rating'] = video_games_filtered_df['Rating'].fillna('Not Rated')
        
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
        
        # Fill any remaining NaN values
        video_games_final_df['Critic_Score'] = video_games_final_df['Critic_Score'].fillna(0)
        video_games_final_df['User_Score'] = video_games_final_df['User_Score'].fillna(0)
        
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
def VideoGameRecommender(video_game_name, video_game_platform, model_data, min_critic_score=0, min_user_score=0, selected_genres=None):
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
        
        # Also suggest similar titles using enhanced search
        matches = enhanced_game_search(video_game_name, video_games_final_df['Name'].unique().tolist(), threshold=0.5)
        if matches:
            st.info(f"Other similar titles you might try: {', '.join(matches)}")
            
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

        # Reset index and start from 1 instead of 0
        video_game_list = video_game_list.reset_index(drop=True)
        video_game_list.index = video_game_list.index + 1  # Start index from 1
        
        recommended_video_game_list = pd.concat([video_game_list, 
                                                pd.DataFrame(recommended_distances, columns=['Similarity_Distance'])], axis=1)
        
        # Apply filters
        if min_critic_score > 0:
            recommended_video_game_list = recommended_video_game_list[recommended_video_game_list['Critic_Score'] >= min_critic_score]
        
        if min_user_score > 0:
            recommended_video_game_list = recommended_video_game_list[recommended_video_game_list['User_Score'] >= min_user_score]
            
        if selected_genres and len(selected_genres) > 0:
            recommended_video_game_list = recommended_video_game_list[recommended_video_game_list['Genre'].isin(selected_genres)]
            
        # If filters removed all recommendations
        if len(recommended_video_game_list) == 0:
            st.warning("No games match your filter criteria. Try relaxing your filters.")
            return None
            
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
        # Show suggestions as user types
        if game_name and len(game_name) > 3 and game_name not in game_names:
            suggestions = enhanced_game_search(game_name, game_names, threshold=0.6)
            if suggestions:
                st.sidebar.write("Did you mean:")
                for suggestion in suggestions:
                    if st.sidebar.button(suggestion, key=f"suggestion_{suggestion}"):
                        game_name = suggestion
    
    # Platform selection
    platforms = sorted(video_games_final_df['Platform'].unique())
    platform = st.sidebar.selectbox("Select platform (optional):", ["Any"] + list(platforms))
    
    # Add filtering options under Advanced Options
    st.sidebar.markdown("---")
    show_advanced = st.sidebar.expander("Advanced Options", expanded=False)
    
    with show_advanced:
        st.header("Filter Recommendations")
        min_critic_score = st.slider("Minimum Critic Score", 0, 100, 0)
        min_user_score = st.slider("Minimum User Score", 0.0, 10.0, 0.0, 0.1)
        selected_genres = st.multiselect("Filter by Genre", 
                                        sorted(video_games_final_df['Genre'].unique()))
    
    # Create tabs at the top of the page
    tab1, tab2, tab3 = st.tabs(["Recommendations", "About", "Data Visualizations"])
    
    
    with tab1:
        # Get recommendations
        if st.sidebar.button("Get Recommendations") or st.session_state.last_search == game_name:
            st.session_state.last_search = game_name
            with st.spinner("Finding similar games..."):
                recommendations = VideoGameRecommender(
                    game_name,
                    platform,
                    model_data,
                    min_critic_score,
                    min_user_score,
                    selected_genres
                )
                st.session_state.recommendations = recommendations
            
            if recommendations is not None:
                # Display recommendations in a better layout
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"<h2 class='subheader'>Top Recommended Video Games for '{game_name}' [platform:{platform}]</h2>", unsafe_allow_html=True)
                    
                    # Generate cards for each game
                    for idx, row in recommendations.iterrows():
                        # Ensure all values are properly formatted and handle NaN values
                        game_name_str = format_value(row['Name'])
                        platform_str = format_value(row['Platform'])
                        genre_str = format_value(row['Genre'])
                        rating_str = format_value(row['Rating'])
                        
                        # Handle numeric values
                        try:
                            user_score = float(row['User_Score'])
                            user_score_display = f"{user_score:.1f}"
                        except (ValueError, TypeError):
                            user_score_display = "N/A"
                            
                        try:
                            critic_score = float(row['Critic_Score'])
                            critic_score_display = f"{critic_score:.1f}"
                        except (ValueError, TypeError):
                            critic_score_display = "N/A"
                            
                        try:
                            similarity = float(row['Similarity_Distance'])
                            similarity_display = f"{similarity:.4f}"
                        except (ValueError, TypeError):
                            similarity_display = "N/A"
                        
                        # Determine rating class for color
                        if rating_str == 'E':
                            rating_class = 'rating-E'
                        elif rating_str == 'T':
                            rating_class = 'rating-T'
                        elif rating_str == 'M':
                            rating_class = 'rating-M'
                        elif rating_str == 'E10+':
                            rating_class = 'rating-E10'
                        else:
                            rating_class = 'rating-other'
                        
                        # Generate Steam search URL
                        steam_url = get_steam_search_url(game_name_str)
                        
                        # Create card HTML with link
                        card_html = f"""
                        <a href="{steam_url}" target="_blank" class="steam-link">
                            <div class="game-card">
                                <div class="game-title">{idx}. {game_name_str}</div>
                                <div class="game-rating {rating_class}">{rating_str}</div>
                                <div class="game-platform">Platform: {platform_str}</div>
                                <div class="game-genre">Genre: {genre_str}</div>
                                <div class="game-scores">
                                    <div class="score-box">
                                        <div class="score-label">User Score</div>
                                        <div class="score-value">{user_score_display}</div>
                                    </div>
                                    <div class="score-box">
                                        <div class="score-label">Critic Score</div>
                                        <div class="score-value">{critic_score_display}</div>
                                    </div>
                                    <div class="score-box">
                                        <div class="score-label">Similarity</div>
                                        <div class="score-value">{similarity_display}</div>
                                    </div>
                                </div>
                            </div>
                        </a>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)
                
                with col2:
                    # Add recommendation statistics
                    st.metric("Average Critic Score", f"{recommendations['Critic_Score'].mean():.1f}")
                    st.metric("Average User Score", f"{recommendations['User_Score'].mean():.1f}")
                    
                    # Get most common genre safely
                    if not recommendations['Genre'].empty:
                        most_common_genre = recommendations['Genre'].mode()[0]
                        st.metric("Most Common Genre", most_common_genre)
                    
                    # Add export functionality
                    export_recommendations(recommendations, game_name)
            else:
                st.info("No recommendations found. Try adjusting your filters or selecting a different game.")
        else:
            st.info("""
            ## How to Use
            1. Select a game from the dropdown or enter a game title
            2. Optionally select a specific platform
            3. Click "Get Recommendations" to find similar games
            4. For more options, expand "Advanced Options" in the sidebar
            5. Download your recommendations as CSV if needed
            """)
    
    with tab2:
        st.markdown("""
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
    with tab3:
        viz_col1, viz_col2 = st.columns([1, 3])
        
        with viz_col1:
            viz_type = st.selectbox("Select visualization:", 
                                ["Genre Distribution", 
                                "Platform Distribution", 
                                "Rating Distribution", 
                                "Critic Score Distribution", 
                                "User Score Distribution", 
                                "Critic vs User Scores",
                                "Top Genres by Score"])
        
        with viz_col2:
            if viz_type == "Genre Distribution":
                fig, ax = plt.subplots(figsize=(10, 6))
                genre_counts = video_games_final_df['Genre'].value_counts()
                sns.barplot(x=genre_counts.index, y=genre_counts.values, ax=ax)
                plt.xticks(rotation=45, ha='right')
                plt.title("Distribution of Video Game Genres")
                plt.ylabel("Number of Games")
                plt.tight_layout()
                st.pyplot(fig)
                
            elif viz_type == "Platform Distribution":
                fig, ax = plt.subplots(figsize=(12, 6))
                platform_counts = video_games_final_df['Platform'].value_counts().head(15)  # Top 15 platforms
                sns.barplot(x=platform_counts.index, y=platform_counts.values, ax=ax)
                plt.xticks(rotation=45, ha='right')
                plt.title("Distribution of Top 15 Video Game Platforms")
                plt.ylabel("Number of Games")
                plt.tight_layout()
                st.pyplot(fig)
                
            elif viz_type == "Rating Distribution":
                fig, ax = plt.subplots(figsize=(10, 6))
                rating_counts = video_games_final_df['Rating'].value_counts()
                sns.barplot(x=rating_counts.index, y=rating_counts.values, ax=ax)
                plt.title("Distribution of Video Game Ratings")
                plt.ylabel("Number of Games")
                plt.tight_layout()
                st.pyplot(fig)
                
            elif viz_type == "Critic Score Distribution":
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(video_games_final_df['Critic_Score'].dropna(), bins=20, kde=True, ax=ax)
                plt.xlabel('Critic Score')
                plt.ylabel('Frequency')
                plt.title("Distribution of Critic Scores")
                st.pyplot(fig)
                
            elif viz_type == "User Score Distribution":
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(video_games_final_df['User_Score'].dropna(), bins=20, kde=True, ax=ax)
                plt.xlabel('User Score')
                plt.ylabel('Frequency')
                plt.title("Distribution of User Scores")
                st.pyplot(fig)
                
            elif viz_type == "Critic vs User Scores":
                fig, ax = plt.subplots(figsize=(10, 8))
                # Drop NaN values for the plot
                plot_data = video_games_final_df.dropna(subset=['User_Score', 'Critic_Score'])
                sns.regplot(x=plot_data['User_Score'], y=plot_data['Critic_Score'], 
                        line_kws={"color": "red"}, scatter_kws={'alpha': 0.3}, ax=ax)
                ax.set(xlabel="User Score", ylabel="Critic Score", title="User Scores vs. Critic Scores")
                st.pyplot(fig)
                
            elif viz_type == "Top Genres by Score":
                fig, ax = plt.subplots(figsize=(12, 8))
                genre_avg = video_games_final_df.groupby('Genre')[['Critic_Score', 'User_Score']].mean().sort_values(by='Critic_Score', ascending=False)
                genre_avg['User_Score_Scaled'] = genre_avg['User_Score'] * 10  # Scale to match critic score range
                
                genre_avg[['Critic_Score', 'User_Score_Scaled']].plot(kind='bar', ax=ax)
                plt.title("Average Scores by Genre")
                plt.ylabel("Score")
                plt.xticks(rotation=45, ha='right')
                plt.legend(['Critic Score', 'User Score (scaled)'])
                plt.tight_layout()
                st.pyplot(fig)
        
        # Add a section for recommendation analysis if recommendations exist
        if st.session_state.recommendations is not None and not st.session_state.recommendations.empty:
            st.markdown("### Analysis of Current Recommendations")
            
            recommendations = st.session_state.recommendations
            
            # Create visualizations for current recommendations
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Genre distribution in recommendations
            genre_counts = recommendations['Genre'].value_counts()
            sns.barplot(x=genre_counts.index, y=genre_counts.values, ax=ax1)
            ax1.set_title("Genres in Recommendations")
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
            
            # Score comparison
            recommendations.dropna(subset=['User_Score', 'Critic_Score'])[['Critic_Score', 'User_Score']].plot(
                kind='scatter', x='User_Score', y='Critic_Score', ax=ax2)
            ax2.set_title("Scores of Recommended Games")
            ax2.set_xlabel("User Score")
            ax2.set_ylabel("Critic Score")
            
            plt.tight_layout()
            st.pyplot(fig)

if __name__ == '__main__':
    main()
