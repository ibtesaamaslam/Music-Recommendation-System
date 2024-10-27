# 🎶 Music Recommendation System Using Machine Learning

This project demonstrates a Music Recommendation System inspired by Spotify, using collaborative filtering techniques to recommend songs based on user interactions. By leveraging **Surprise library’s SVD algorithm** and **Kaggle’s Million Song Dataset**, this model suggests personalized music options, offering an immersive experience for users.

---

## 📁 Project Structure

- **Data Loading**: Utilizes Kaggle’s Million Song Dataset, including user-song interaction data and metadata.
- **Collaborative Filtering Model**: Built with Singular Value Decomposition (SVD) for personalized song recommendations.
- **User Interaction Features**: Trained on user-specific listening patterns to predict song preferences.

---

## 🚀 Getting Started

1. **Download the Dataset**: Ensure Kaggle API is set up on your system. Run the Kaggle API command to download the dataset to your local environment.
2. **Install Dependencies**: Make sure to install the required packages:
   ```bash
   pip install pandas numpy surprise kagglehub
   ```
3. **Run the Notebook**: Use the provided Jupyter Notebook file for an end-to-end implementation.

---

## 📊 Data and Model

- **Dataset**: The Million Song Dataset is an open dataset that provides user interaction history with songs, allowing us to identify and recommend songs based on similar patterns.
- **Model**: We use Surprise’s SVD algorithm for collaborative filtering. This model works by identifying similarities in user interaction patterns, predicting how much a user might like a song based on their previous ratings.
- **Features**: Each song recommendation includes:
  - **Predicted Rating**: Estimated user preference for the song.
  - **Song Title and Artist**: Mapped using song metadata for a more user-friendly experience.

---

## 🛠️ Functions and Usage

1. **Data Preprocessing**: We clean and structure data to prepare it for the model.
2. **Training the Model**: The `train_test_split` method from Surprise’s model selection helps split data for training and testing. 
3. **Getting Recommendations**: Use the `recommend_songs()` function to get personalized song recommendations:
   ```python
   recommendations = recommend_songs(user_id, model, trainset, songs_metadata)
   ```

---

## 🔍 Example Usage

- **Top 5 Recommendations**: Run `recommend_songs()` for a given user ID to receive a list of top-rated songs they might enjoy, along with artist details and predicted rating scores.

```python
# Example call for user 196
recommendations = recommend_songs('196', model, trainset, songs_metadata)
for title, artist, score in recommendations:
    print(f'Title: {title}, Artist: {artist}, Predicted rating: {score}')
```

---

## 🔧 Technologies Used

- **Python Libraries**: `pandas`, `numpy`, `surprise`, and `kagglehub`
- **Dataset**: Million Song Dataset from Kaggle
- **Algorithm**: Collaborative Filtering with SVD from the Surprise library

---

## 📑 Future Enhancements

- **Hybrid Recommendation Models**: Combine content-based features (like genre or artist popularity) with collaborative filtering for better results.
- **Enhanced Metadata**: Add additional attributes like genre, release year, or album to improve recommendations.
- **Real-Time Recommendations**: Incorporate a streaming feature to recommend songs based on live interaction data.
