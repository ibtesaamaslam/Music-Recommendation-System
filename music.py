# Installing required libraries in Terminal

# !pip install scikit-surprise numpy
# python music.py

# importing Libraries 
import logging
import numpy as np
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split, cross_validate
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('music_recommender.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

def load_data():
    """Load MovieLens dataset."""
    try:
        logger.info("Using sample MovieLens dataset")
        data = Dataset.load_builtin('ml-100k')
        return data
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def split_data(data, test_size=0.2):
    """Split data into training and test sets."""
    try:
        logger.info(f"Splitting data with test_size={test_size}")
        trainset, testset = train_test_split(data, test_size=test_size, random_state=42)
        return trainset, testset
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        raise

def train_model(trainset, params=None):
    """Train SVD model with given parameters."""
    try:
        if params is None:
            params = {'n_epochs': 20, 'lr_all': 0.005, 'reg_all': 0.02}
        logger.info(f"Training svd model with params: {params}")
        model = SVD(**params)
        model.fit(trainset)
        logger.info("Model trained successfully")
        return model
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

def evaluate_model(model, data):
    """Evaluate model performance with cross-validation."""
    try:
        logger.info("Evaluating model with 3-fold cross-validation")
        cv_results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=3, verbose=True, return_train_measures=True)
        
        # Log mean and std of metrics
        for metric in ['rmse', 'mae']:
            test_mean = np.mean(cv_results[f'test_{metric}'])
            test_std = np.std(cv_results[f'test_{metric}'])
            train_mean = np.mean(cv_results[f'train_{metric}'])
            train_std = np.std(cv_results[f'train_{metric}'])
            logger.info(f"Mean {metric.upper()}: Test = {test_mean:.4f} ± {test_std:.4f}, Train = {train_mean:.4f} ± {train_std:.4f}")
        
        return cv_results
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise

def evaluate_testset(model, testset):
    """Evaluate model on test set."""
    try:
        logger.info("Making predictions on test set")
        predictions = model.test(testset)
        rmse = accuracy.rmse(predictions)
        mae = accuracy.mae(predictions)
        logger.info(f"Test RMSE: {rmse:.4f}")
        logger.info(f"Test MAE: {mae:.4f}")
        return predictions
    except Exception as e:
        logger.error(f"Error evaluating test set: {e}")
        raise

def recommend_items(model, trainset, user_id, n=5, sample_size=1000):
    """Generate top-N recommendations for a user."""
    try:
        logger.info(f"Generating {n} recommendations for user {user_id}")
        
        # Get items the user hasn't rated
        user_rated_items = set()
        for uid, iid, _ in trainset.all_ratings():
            if trainset.to_raw_uid(uid) == user_id:
                user_rated_items.add(trainset.to_raw_iid(iid))
        
        all_items = set([trainset.to_raw_iid(i) for i in range(trainset.n_items)])
        user_anti_testset = list(all_items - user_rated_items)
        
        if not user_anti_testset:
            logger.warning(f"No unrated items available for user {user_id}")
            return []
        
        # Sample candidates for efficiency
        sample_size = min(sample_size, len(user_anti_testset))
        logger.info(f"Sampling from {len(user_anti_testset)} candidates for efficiency")
        
        if sample_size > 0:
            sampled_items = np.random.choice(user_anti_testset, size=sample_size, replace=False)
        else:
            sampled_items = user_anti_testset
        
        # Predict ratings for sampled items
        predictions = [model.predict(user_id, item) for item in sampled_items]
        
        # Sort predictions by estimated rating
        top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
        
        # Extract item IDs and estimated ratings
        recommendations = [(pred.iid, pred.est) for pred in top_n]
        return recommendations
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        logger.error(f"Detailed traceback:", exc_info=True)
        return []

def main():
    """Main function to run the recommendation system."""
    try:
        # Load and split data
        data = load_data()
        trainset, testset = split_data(data)
        
        # Train model
        model = train_model(trainset)
        
        # Evaluate model
        evaluate_model(model, data)
        evaluate_testset(model, testset)
        
        # Generate recommendations for a sample user
        user_id = 196
        recommendations = recommend_items(model, trainset, user_id, n=5)
        
        if recommendations:
            logger.info(f"Top 5 recommendations for user {user_id}:")
            for item_id, rating in recommendations:
                logger.info(f"Item {item_id}: Predicted rating {rating:.2f}")
        else:
            logger.info(f"No recommendations available for user {user_id}")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        logger.error(f"Detailed traceback:", exc_info=True)

if __name__ == "__main__":
    main()