# Movie-Recommendation-System
## Problem Statement
* Users are exposed to a large number of movies, making it difficult to find content that matches their preferences.
* Traditional browsing does not provide personalized recommendations based on user behavior or movie characteristics.
* This project develops a Movie Recommendation System using the MovieLens dataset to address this challenge.
* It uses **content-based filtering** (genres) and **collaborative filtering** (user ratings) to identify similarities and suggest relevant movies.
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

### Incorrect data type
* No **incorrect data type** in ratings file and movies file.
* timestamp column's data type is (int64).
* There was no need to convert timestamp to datetime format because it was not relevant to the project objective and showed a very weak relationship with the rating variable.
* Therefore, it was not used in the recommendation models.
* It was only included in the correlation matrix for analysis, and its original numeric data type posed no issue for computations.


## Visualizations
  <img width="536" height="355" alt="image" src="https://github.com/user-attachments/assets/5b4becc9-749f-472d-9850-2f0ddbcbda53" />

* The histogram is used to show the distribution of user ratings and understand how users generally rate movies.
* From the graph, most ratings are concentrated between 3.0 and 4.5, with a clear peak around 4.0, indicating that users tend to give positive ratings.
* Very low ratings (below 2.0) appear, less frequently, suggesting that users rarely rate movies poorly.
* The distribution is therefore skewed towards higher values, showing an overall preference for favorable reviews.
* Additionally, although ratings range from 0.5 to 5.0, the majority cluster within a smaller range, highlighting consistent user behavior in giving moderately high ratings.

  
