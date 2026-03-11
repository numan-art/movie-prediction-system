**MOVIE_RECOMMENDATION_SYSTEM:**

**1. Data Aggregation**
The system takes raw user ratings and calculates two main "traits" for every movie:

Average Rating: The quality score of the movie.

Popularity: The total number of people who watched/rated it.

**2. Feature Scaling**
Since popularity scores (e.g., 500) are much larger than ratings (e.g., 4.5), we use a StandardScaler to make sure the computer treats both traits as equally important.

**3. Clustering Logic**
We use two different ways to group the movies:

K-Means: Best for finding clearly defined groups.

Hierarchical: Best for seeing the relationships and sub-genres through a Dendrogram.

**4. Dimensionality Reduction**
To visualize the results, we reduce the high-dimensional data (Genres + Rating + Popularity) into 2D coordinates using PCA and t-SNE.
