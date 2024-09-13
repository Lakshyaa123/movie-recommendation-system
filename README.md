# Building A Movie Recommendation System Using Machine Learning with Python in Google Colab

**Introduction**

This repository provides a guide for building a movie recommendation system using machine learning techniques in Python. The system utilizes various libraries and methods to recommend movies based on user preferences and similarities between movies.

**Dataset**

To build and train the recommendation system, you will need the dataset. You can download it from the following link:

- [Movie Dataset](https://drive.google.com/file/d/1cCkwiVv4mgfl20ntgY3n4yApcWqqZQe6/view)

**Core Libraries Used:**

- `numpy`: A fundamental package for numerical computations in Python.
- `pandas`: A powerful data manipulation and analysis library.
- `difflib`: A module for comparing sequences, which helps in matching movie names.
- `sklearn.feature_extraction.text.TfidfVectorizer`: A tool for converting text data into numerical vectors based on Term Frequency-Inverse Document Frequency (TF-IDF).
- `sklearn.metrics.pairwise.cosine_similarity`: A function to compute the similarity between movie vectors using cosine similarity.

**Overview:**

The recommendation system is designed to suggest movies based on their similarity to other movies. It works by analyzing movie descriptions, extracting features, and calculating the similarity between movies to make recommendations.

**How It Works:**

1. **Data Preparation:** Load and preprocess the movie dataset using `pandas`.
2. **Feature Extraction:** Convert movie descriptions into numerical vectors using `TfidfVectorizer`.
3. **Similarity Calculation:** Measure the similarity between movies using `cosine_similarity`.
4. **Recommendation:** Suggest movies based on their similarity scores.

**Implementation:**

This guide will walk you through building and running the movie recommendation system using Google Colab, which provides an interactive environment for executing Python code.

# Steps to Build the Movie Recommendation System

## Step 1: Importing Dependencies

```python
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```
## Step 2: Data Collection and Pre-Processing

```python
# Loading the data from the csv file to a pandas dataframe
movies_data = pd.read_csv('/content/movies.csv')

# Printing the first 5 rows of the dataframe
movies_data.head()

# Number of rows and columns in the data frame
movies_data.shape
```
## Step 3: Feature Selection and Combination

```python
# Selecting the relevant features for recommendation
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
print(selected_features)

# Replacing the null values with empty strings
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combining all the selected features
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']
print(combined_features)
```
## Step 4: Feature Vectorization and Similarity Calculation

```python
# Converting the text data to feature vectors
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
print(feature_vectors)

# Getting the similarity scores using cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(feature_vectors)
print(similarity)
print(similarity.shape)
```

## Step 5: Getting Recommendations Based on User Input

```python
# Getting the movie name from the user
movie_name = input('Enter your favourite movie name: ')

# Creating a list with all the movie names given in the dataset
list_of_all_titles = movies_data['title'].tolist()
print(list_of_all_titles)

# Finding the close match for the movie name given by the user
import difflib

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
print(find_close_match)

close_match = find_close_match[0]
print(close_match)

# Finding the index of the movie with title
index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
print(index_of_the_movie)

# Getting a list of similar movies
similarity_score = list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)

# Sorting the movies based on their similarity score
sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
print(sorted_similar_movies)

# Print the name of similar movies based on the index
print('Movies suggested for you:\n')

i = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index == index]['title'].values[0]
    if i < 30:
        print(f"{i}. {title_from_index}")
        i += 1
```

## Output

```text
Enter your favourite movie name: Iron Man
['Iron Man', 'Iron Man 3', 'Iron Man 2']
Iron Man
68
[(0, 1.0), (1, 0.0721948683744621), (2, 0.03773299420976186), ...]
Movies suggested for you:
1. Iron Man
2. Iron Man 2
3. Iron Man 3
4. Avengers: Age of Ultron
5. The Avengers
6. Captain America: Civil War
7. Captain America: The Winter Soldier
8. Ant-Man
9. X-Men
10. Made
11. X-Men: Apocalypse
12. X2
13. The Incredible Hulk
14. The Helix... Loaded
15. X-Men: First Class
16. X-Men: Days of Future Past
17. Captain America: The First Avenger
18. Kick-Ass 2
19. Guardians of the Galaxy
20. Deadpool
21. Thor: The Dark World
22. G-Force
23. X-Men: The Last Stand
24. Duets
25. Mortdecai
26. The Last Airbender
27. Southland Tales
28. Zathura: A Space Adventure
29. Sky Captain and the World of Tomorrow
```
