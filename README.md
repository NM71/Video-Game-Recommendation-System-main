# Video Games Recommendation System

The aim of the project is to provide video game suggestions based on user input: game title and platform (optional). This effort will benefit individuals who want to explore and find new games.

The recommendation model is built using "NearestNeighbors", a supervised machine learning algorithm that uses distance computation to measure similarity or dissimilarity between the data points.

1. Calculate the distance between the input and each data point in a dataset
2. Select the data points with the smallest distance as the nearest neighbors

The following are the features that were used to develop the model:

- Genre: Genre of the video game
- Platform: Platform to play the video game
- Rating: The ESRB ratings
- Critic Score: Aggregated score compiled by Metacritic staff
- User Score: Aggregated score by Metacritic's subscribers

The dataset was obtained from the website https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings.

## Tools and Programming Languages Used

### Programming Languages:
- **Python 3.11** - Main programming language

### Libraries and Frameworks:
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation and analysis
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning (K-Nearest Neighbors algorithm)
- **Streamlit** - Web application framework

### Development Tools:
- **Jupyter Notebook** - Interactive development and analysis
- **Kaggle Dataset** - Data source for video game information

### Machine Learning Algorithm:
- **K-Nearest Neighbors (KNN)** - Unsupervised learning for similarity matching
- **Cosine Similarity** - Distance computation method

## Project Structure

This project consists of three main components:

1. **Video Games Recommendation System.ipynb** - Original Jupyter Notebook with data analysis and model development
2. **Video_Games_Recommendation_System.py** - Python script version converted from the notebook
3. **app.py** - Streamlit web application for interactive recommendations

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the following packages:

```bash
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install streamlit
```

Or install all dependencies at once:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn streamlit
```

## Usage

### Running the Jupyter Notebook
Launch the Jupyter Notebook for data exploration and model development:
```bash
jupyter notebook "Video Games Recommendation System.ipynb"
```

### Running the Python Script
Execute the standalone Python script:
```bash
python Video_Games_Recommendation_System.py
```

### Running the Streamlit Web Application
Launch the interactive web application:
```bash
streamlit run app.py
```

The Streamlit app provides an intuitive web interface where users can:
- Select games from a dropdown or enter custom game titles
- Choose specific platforms or search across all platforms
- View recommendations in an easy-to-read format
- Explore database analytics and visualizations

## Dataset

Ensure the dataset file `Video Games Sales.csv` is placed in a `Dataset` folder in the project directory:
```
Video-Game-Recommendation-System-main/
├── Dataset/
│   └── Video Games Sales.csv
├── Video Games Recommendation System.ipynb
├── Video_Games_Recommendation_System.py
├── app.py
└── README.md
```

## Features

- **Intelligent Recommendations**: Uses K-Nearest Neighbors algorithm with cosine similarity
- **Flexible Input**: Supports both exact game matches and fuzzy text matching
- **Platform Filtering**: Optional platform-specific recommendations
- **Data Imputation**: Handles missing values using genre-based averages
- **Interactive Web Interface**: User-friendly Streamlit application
- **Data Visualization**: Comprehensive analytics and charts

## Project Members

- **Nousher Murtaza**
- **Ahmad Basil Awan**
