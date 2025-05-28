# Video Games Recommendation System

The aim of the project is to provide video game suggestions based on user input: game title and platform (optional). This effort will benefit individuals who want to explore and find new games.

The recommendation model is built using "NearestNeighbors", an unsupervised machine learning algorithm that uses distance computation to measure similarity or dissimilarity between the data points.

1. Calculate the distance between the input and each data point in a dataset
2. Select the data points with the smallest distance as the nearest neighbors

The following are the features that were used to develop the model:

- Genre: Genre of the video game
- Platform: Platform to play the video game
- Rating: The ESRB ratings
- Critic Score: Aggregated score compiled by Metacritic staff
- User Score: Aggregated score by Metacritic's subscribers

The dataset was obtained from the website https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the following packages.
```bash
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
```

## Program Execution
Launch the Jupyter Notebook
```bash
Video Games Recommendation System.ipynb
```

## Group 12 Members
- Auradee Castro
- Bhumika Rajendra Babu  
- Miraj Sinya
- Olivia Deguit
- Roger Mais
