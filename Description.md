Music Recommendation System
This repository contains a Python-based music recommendation system built using the Surprise library. It leverages the Singular Value Decomposition (SVD) algorithm to predict user preferences and generate personalized music recommendations. The system uses the MovieLens 100k dataset as a proxy for demonstration, but it can be adapted for music-specific datasets.
Features

Data Loading: Automatically downloads and loads the MovieLens 100k dataset (or custom datasets with minor modifications).
Model Training: Trains an SVD model with configurable parameters (epochs, learning rate, regularization).
Evaluation: Performs 3-fold cross-validation and test set evaluation using RMSE and MAE metrics.
Recommendations: Generates top-N recommendations for a given user by predicting ratings for unrated items.
Logging: Comprehensive logging to track execution, errors, and performance metrics.

Prerequisites

Python 3.6+
Required libraries:pip install scikit-surprise numpy



Installation

Clone the repository:git clone https://github.com/your-username/music-recommendation-system.git
cd music-recommendation-system


Install dependencies:pip install -r requirements.txt


(Optional) Create a requirements.txt file:scikit-surprise
numpy



Usage

Run the main script:
python music.py


The script will:

Download the MovieLens 100k dataset (if not already present).
Split the data into training and test sets (80-20 split).
Train an SVD model.
Evaluate the model using cross-validation and test set metrics.
Generate top-5 recommendations for user ID 196 (configurable).


Check the music_recommender.log file for detailed execution logs.


Code Structure

music.py: Main script containing the recommendation system logic.
load_data(): Loads the dataset.
split_data(): Splits data into training and test sets.
train_model(): Trains the SVD model.
evaluate_model(): Performs cross-validation.
evaluate_testset(): Evaluates predictions on the test set.
recommend_items(): Generates top-N recommendations for a user.
main(): Orchestrates the workflow.


music_recommender.log: Log file for execution details and errors.

Example Output
2025-04-20 00:51:25,658 - INFO - Splitting data with test_size=0.2
2025-04-20 00:51:27,398 - INFO - Model trained successfully
2025-04-20 00:51:35,879 - INFO - Mean RMSE: Test = 0.9456 ± 0.0026, Train = 0.6926 ± 0.0029
2025-04-20 00:51:36,150 - INFO - Test RMSE: 0.7848
2025-04-20 00:51:36,151 - INFO - Generating 5 recommendations for user 196
2025-04-20 00:51:36,155 - INFO - Top 5 recommendations for user 196:
2025-04-20 00:51:36,155 - INFO - Item 318: Predicted rating 4.85
2025-04-20 00:51:36,155 - INFO - Item 483: Predicted rating 4.78
...

Customization

Dataset: To use a music-specific dataset, modify the load_data() function in music.py to load your dataset with the appropriate Reader configuration (e.g., user-item-rating format).
Model Parameters: Adjust the params dictionary in train_model() to tune the SVD model (e.g., n_epochs, lr_all, reg_all).
User ID: Change the user_id in main() to generate recommendations for a different user.
Number of Recommendations: Modify the n parameter in recommend_items() to change the number of recommendations.

Future Improvements

Integrate a music-specific dataset (e.g., Last.fm, Spotify).
Add support for other recommendation algorithms (e.g., KNN, NMF).
Implement a user interface (e.g., Flask or Streamlit) for interactive recommendations.
Optimize sampling for large datasets using more advanced techniques.

Contributing
Contributions are welcome! Please:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit your changes (git commit -m 'Add your feature').
Push to the branch (git push origin feature/your-feature).
Open a pull request.
