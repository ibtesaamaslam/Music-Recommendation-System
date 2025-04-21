
## 🎵 Music Recommender System using SVD (Collaborative Filtering)

This project implements a **collaborative filtering-based recommender system** using the **SVD (Singular Value Decomposition)** algorithm from the [Surprise](https://surprise.readthedocs.io/en/stable/) library. The model is trained on the **MovieLens 100K dataset**, simulating a music or movie recommendation environment. The system includes robust logging, model evaluation, and a custom recommendation engine for individual users.

---

### 🚀 Features

- Trains an **SVD-based collaborative filtering model** for rating prediction.
- Utilizes **MovieLens 100K** dataset (built-in with `surprise`).
- Clean **logging system** for debugging and model traceability.
- **Model evaluation** using RMSE & MAE with both cross-validation and hold-out test sets.
- Top-N **personalized recommendations** for a given user.
- Handles cold-start scenarios gracefully and includes safety fallbacks.

---

### 🧠 Algorithm Overview

- **SVD (Matrix Factorization)**:
  - Learns latent features for users and items based on past rating patterns.
  - Makes predictions by reconstructing user-item matrices.

- **Surprise Library**:
  - Offers flexible tools for loading datasets, splitting, training, evaluating, and predicting.
  - Highly optimized and easy to use for collaborative filtering.

---

### 📁 Project Structure

```bash
Music-Recommender-System/
│
├── music.py                        # Main Python script (fully functional)
├── music_recommender.log          # Logs execution details
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation (this file)
```

---

### 📊 Workflow Summary

#### 1. **Data Loading**
- Uses `Dataset.load_builtin('ml-100k')` from `surprise`.

#### 2. **Data Splitting**
- 80/20 split into training and test sets.

#### 3. **Model Training**
- Trains an `SVD` model with hyperparameters:
  - `n_epochs=20`
  - `lr_all=0.005`
  - `reg_all=0.02`

#### 4. **Cross-Validation Evaluation**
- 3-fold cross-validation with metrics:
  - RMSE
  - MAE

#### 5. **Test Set Evaluation**
- Predicts ratings on the held-out test set.
- Reports RMSE & MAE.

#### 6. **Top-N Recommendation**
- Generates top-N unrated item predictions for a specific user.
- Filters already-rated items.
- Samples from remaining candidates for speed.
- Returns sorted predictions by estimated rating.

---

### 📌 Example Output

```
Top 5 recommendations for user 196:
Item 50: Predicted rating 4.83
Item 318: Predicted rating 4.76
Item 64: Predicted rating 4.68
Item 408: Predicted rating 4.61
Item 12: Predicted rating 4.59
```

---

### ✅ How to Run

1. **Install dependencies**:
   ```bash
   pip install scikit-surprise numpy
   ```

2. **Run the script**:
   ```bash
   python music.py
   ```

3. **View logs**:
   - Console output for real-time insights.
   - `music_recommender.log` for saved logs.

---

### 🔧 Customization

- 🔄 Replace `ml-100k` with a custom user-item dataset using `Reader()` and `Dataset.load_from_df()`.
- 🎯 Modify `user_id` in `main()` to generate personalized recommendations for different users.
- 📊 Adjust model hyperparameters (`n_epochs`, `lr_all`, etc.) to tune performance.

---

### 🧪 Requirements

```txt
scikit-surprise
numpy
```
