# Movie-Recommendation-System
<img width="359" height="267" alt="image" src="https://github.com/user-attachments/assets/4137d8e5-0284-49fd-808e-6b27cc6908c9" />

* Image is AI Generated
  
## Problem Statement
* Users are exposed to a large number of movies, making it difficult to find content that matches their preferences.
* Traditional browsing does not provide personalized recommendations based on user behavior or movie characteristics.
* This project develops a Movie Recommendation System using the MovieLens dataset to address this challenge.
* It uses **content-based recommendations** (genres) and **collaborative filtering** (user ratings) to identify similarities and suggest relevant movies.
* The system aims to improve movie selection by providing accurate and meaningful recommendations.

## Project Goals
* Build a simple recommendation system based on user ratings and genres.
* Introduce content-based recommendations(Genres)  and collaborative filtering (User ratings).
* Use MovieLens Dataset (Kaggle) to build and test a movie recommendation system using real movies and user data.  

## Dataset
* The MovieLens Dataset was obtained from Kaggle Website. Stored in data folder in the repository.
* MovieLens Dataset contains two files namely movies and ratings.
* Movies file contains 3 columns and 10 329 rows.
* Ratings file contains 4 columns and 105,339 rows.
   
 ### Movies Dataset

| Column Name | Description                          | Data Type |
|-------------|--------------------------------------|-----------|
| movieId     | Unique identifier for each movie     | int64     |
| title       | Name of the movie                    | object    |
| genres      | Genres associated with the movie     | object    |

### Ratings Dataset

| Column Name | Description                                      | Data Type |
|-------------|--------------------------------------------------|-----------|
| userId      | Unique identifier for each user                  | int64     |
| movieId     | Identifier linking rating to a specific movie    | int64     |
| rating      | User’s rating for the movie                     | float64   |
| timestamp   | Time when the rating was given (Unix format)     | int64     |


## Features and Target Variable
* Features:
    * genre - content-based filtering
    * rating - collaborative filtering
* Target Variable - NA
  
### Explanation of Features and Target Variable
* The project primarily uses movie genres and user ratings as features to build the recommendation system.
* For content-based filtering, genres are encoded into numerical format and used to measure similarity between movies.
* For collaborative filtering, a user–movie rating matrix is created to capture user preferences.
* There is no explicit target variable, as the system focuses on identifying similarities rather than predicting a specific output.
* The recommendations are generated based on these similarity patterns between movies and users


## Data Exploration and Cleaning 
* **No Missing** Values were found in Movies file and Ratings file.

### Outliers
* Outliers were identified using IQR Method and handled using Winsorization.
* **In ratings file**, the outliers table shows 4,456 rows where the ratings fall outside the normal range.
* Most of these outliers have very low rating values (0.5 and 1.0), indicating that they are significantly lower than the typical ratings observed in the dataset.
* This suggests that while most users tend to give moderate to high ratings, a smaller number of ratings are unusually low, which are flagged as outliers.
* However, these values are still valid user inputs and represent genuine user opinions rather than errors.
* Therefore, they were retained in the dataset to preserve the integrity of user preferences and ensure meaningful recommendations.
* IQR method only applies to numeric values hence only used in movieId in movies file.
* **In movieId in movies file**, The IQR method flagged some rows as outliers because of large movieId values but these are not real outliers since movieId is just an identifier and does not represent meaningful numerical data. 
* Winsorization is not done in movieId column since it would distort the dataset and affect data integrity.

### Movie Theatre filled with people showing  Elemental (2023) movie by Pixar
<img width="545" height="360" alt="image" src="https://github.com/user-attachments/assets/5b4a4764-f44c-47dd-ba80-239d7123a1a2" />

* Image from Pinterest  https://www.pinterest.com/Lishen83/
  
### Incorrect data type
* No **incorrect data type** in ratings file and movies file.
* timestamp column's data type is (int64).
* There was no need to convert timestamp to datetime format because it was not relevant to the project objective and showed a very weak relationship with the rating variable.
* Therefore, it was not used in the recommendation models.
* It was only included in the correlation matrix for analysis, and its original numeric data type posed no issue for computations.


## Visualizations
### Distribution of Movie Ratings Given by Users
  <img width="536" height="355" alt="image" src="https://github.com/user-attachments/assets/5b4becc9-749f-472d-9850-2f0ddbcbda53" />

* The histogram is used to show the distribution of user ratings and understand how users generally rate movies.
* From the graph, most ratings are concentrated between 3.0 and 4.5, with a clear peak around 4.0, indicating that users tend to give positive ratings.
* Very low ratings (below 2.0) appear, less frequently, suggesting that users rarely rate movies poorly.
* The distribution is therefore skewed towards higher values, showing an overall preference for favorable reviews.
* Additionally, although ratings range from 0.5 to 5.0, the majority cluster within a smaller range, highlighting consistent user behavior in giving moderately high ratings.

### Correlation Analysis of user, movie, rating, and timestamp column
  <img width="385" height="328" alt="image" src="https://github.com/user-attachments/assets/0bd33042-eb4a-4f4a-be34-23bceaef15ee" />

* The correlation matrix heatmap shows the relationships between userId, movieId, rating, and timestamp, where most correlation values are very close to zero, indicating weak or no linear relationships among the variables.
* userId has almost no correlation with rating (−0.045), movieId (0.089), or timestamp (0.04) meaning user identifiers do not influence ratings or timing.
* Similarly, movieId has almost no correlation with rating (−0.025), showing that ratings are independent of movie identifiers.
* Rating also has virtually no correlation with timestamp (0.0027), suggesting that user ratings remain consistent over time without noticeable trends.
* The only notable relationship is a moderate positive correlation between movieId and timestamp (0.52), which suggests that movies with higher IDs tend to appear later in time, likely because newer movies are added sequentially to the dataset.
* Overall, the matrix indicates that the dataset variables are largely independent, except for the time-based progression of movie IDs.


## Feature Engineering and Selection 
* In this step, the ratings dataset is enhanced and prepared for modeling by combining, transforming, and selecting meaningful features:

### Merging datasets:
* The Ratings and Movies datasets are merged on movieId to combine user rating information with movie details such as title and genres.

### Creating new features:
* Additional useful features are extracted to improve analysis:
    * The **year column** is extracted from the movie title and converted into a numeric format for analysis.
    * A **genre count feature** is created to represent how many genres each movie has.

### Encoding categorical data:
* The genres column, which contains multiple categories, is transformed using **one-hot encoding**.
* This creates binary columns for each genre so they can be used in machine learning models.

### Feature selection using correlation:
* Only numerical features are selected, and a correlation analysis is performed to understand relationships with the target variable (rating).
* The results show that:Most features have **very weak correlation** with ratings.
* genre_count has a slight positive influence, while year, userId, and movieId contribute very little.

### Removing Unnecessary columns:
Unnecessary columns like title and genres are dropped after feature extraction to avoid redundancy.

## Model Building
## Content-Based Recommendations  
* The model uses cosine similarity to measure how similar movies are based on their one-hot encoded genre features.
* Each movie is represented as a vector of genres, and similarity is computed between every pair of movies.
  
### Movie Similarity Matrix
<img width="442" height="329" alt="image" src="https://github.com/user-attachments/assets/fb1a02b3-0da1-473d-aad7-681b4700d0d2" />

* The output shows a similarity matrix in table form, which represents how similar each movie is to every other movie based on their genres.
* Each row and column corresponds to a movie, and each cell contains a cosine similarity score between the two movies.
* The values range from 0 to 1, where:
    * 1.0 indicates identical genre composition (perfect similarity).
    * Values close to 1 indicate strong similarity (many shared genres).
    * Values around 0 indicate little or no similarity (no shared genres).
* The diagonal values are all 1.0 because each movie is perfectly similar to itself.

### Recommendation Function:
* A function recommend_movies() is created to generate recommendations:
    * It takes a movie title as input.
    * Finds its index in the dataset.
    * Retrieves similarity scores from the matrix.
    * Sorts movies by similarity in descending order.
    * Returns the top N most similar movies, excluding the input movie.
* Example Output:When the model is applied to “Toy Story (1995)”, it returns:
    * Antz (1998)
    * Toy Story 2 (1999)
    * Adventures of Rocky and Bullwinkle, The (2000)
    * Emperor's New Groove, The (2000)
    * Monsters, Inc. (2001)
* These recommendations are generated because these movies share similar genres (e.g., animation, family, comedy) with the input movie.

## Collaborative Filtering 
* This section implements a user-based collaborative filtering model, which recommends items based on similarities between users.
* User–Movie Matrix Creation:
    * A matrix is created where:
        * Each row represents a user (userId).
        * Each column represents a movie (movieId).
        * Each cell contains the rating given by a user to a movie.
        * Missing ratings are filled with 0, indicating no interaction.
* User Similarity Computation:
    * Cosine similarity is applied to this matrix to measure how similar users are based on their rating patterns.
    * Users who rate movies similarly will have higher similarity scores.

### User Similarity Matrix
<img width="484" height="333" alt="image" src="https://github.com/user-attachments/assets/0aede7bb-62b5-499e-b882-4d2f3e852790" />

* The displayed table is the output of the collaborative filtering model, showing similarity scores between users:
    * Each row and column represents a user.
    * Each value indicates the similarity between two users.
    * Values range from:
        * 1.0 → Users have identical rating behavior.
        * Close to 0 → Users have very different preferences.
        * The diagonal values are 1.0, since each user is perfectly similar to themselves.
* For example:
    * A value like 0.47 indicates moderately similar users.
    * A value like 0.05 indicates very weak similarity.
    * A value close to 0 means no meaningful similarity
      
## Limitations
* The content-based recommender system uses only genres and ignores other movie features.
* The recommendation system struggles with new users or new movies (cold start problem).
* The collaborative filtering model treats missing ratings as 0, which may reduce accuracy.
* The content-based model does not provide highly personalized recommendations for each user.

## Future Improvements
* Combine content-based and collaborative filtering (hybrid model).
* Add more features like cast, director, and movie description.
* Handle cold start using popularity or hybrid methods.
* Use advanced techniques like matrix factorization or deep learning.
* Build a web app for real-time recommendations.

## Summary
| Model | Type | Purpose |
|------|------|--------|
| Content-Based | Similarity | Recommend similar movies |
| Collaborative | User-based | Recommend based on users |

## Access and Local Usage

### Access the Deployed App
You can use the movie recommendation system online here:

https://movierecommendationsystemgit-gqwg73ifsffnfr7bke3rxo.streamlit.app/

### Run the App Locally
To run the system on your own machine:

1. Clone or download this project.
2. Open a terminal in the project folder.
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Start the Streamlit app:

```bash
streamlit run app.py
```

5. Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

In the app, choose between content-based recommendations by movie/genre or collaborative recommendations based on user rating histories.
