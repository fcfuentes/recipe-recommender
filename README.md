# 🍽️ Recipe Recommender System (NCF + Content-Based Hybrid)

This project is a neural collaborative filtering recommender system designed to suggest recipes based on user preferences and recipe metadata. It was built as part of an ongoing food blog project focused on helping users explore cost-effective, nutritious, and practical food choices.

##  Project Goals

- Predict how much a user will like a given recipe
- Start from a basic NCF model using user-recipe interaction data
- Progressively improve the model through:
  - Regularization and hyperparameter tuning
  - A hybrid model that includes recipe content features (like dietary tags and behavior metrics)


## Tech Stack

- Python 3.11
- TensorFlow 2.16+
- Pandas, NumPy
- Scikit-learn
- Matplotlib


## Workflow Overview

1. **Data Setup**:  
   This project uses the [Food.com Recipes and User Interactions Dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) available on Kaggle.

   ### Steps to Set Up the Data

    1. Download the dataset from Kaggle:  
       https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions

    2. Unzip the archive and create a folder named `data/` in the root of the project.

    ### Preprocessing

        Once the data is in place, run the following notebooks or scripts in order to generate the processed datasets used for modeling:

        1. `data_preprocessing.ipynb`  
            - Cleans and merges the raw datasets  
            - Outputs `recipes_clean.pkl` and `interactions_clean.pkl`

        2. `feature_preprocessing.ipynb`  
            - Adds derived features, tag encodings, behavioral profiles, etc.  
            - Outputs `recipes_extended.pkl` and `interactions_extended.pkl`

2. **Baseline Model**:  
   A basic NCF model using embedding layers and dense layers to predict scaled ratings.

3. **Tuning**:  
   Grid search over key hyperparameters like:
   - Embedding dimension
   - Hidden layer size
   - Dropout rate
   - Learning rate

   Evaluation is done using MAE and RMSE on a 5% data sample for efficiency.

4. **Hybrid Model** (WIP):  
   Adds content-based features from the recipe metadata to enrich the recommendation process.


## How to Run

### Local

1. Clone the repo and set up a virtual environment:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2. Setup data files in `./data/`:
    - `interactions_extended.pkl`
    - `recipes_extended.pkl`

3. Run your notebook or script:

    ```bash
    jupyter notebook
    ```

###  In Google Colab

1. Open the notebook in Colab  
2. Upload the `.pkl` files using the file uploader, or mount Google Drive
3. Change the file paths to match the Colab environment


## Notes

- The project is still a work in progress — especially the hybrid model integration.

## Model Performance Summary

The following table shows the performance (on the test set) of each model after full training:

| Model                  | MAE    | RMSE   | Notes                                        |
|------------------------|--------|--------|----------------------------------------------|
| Basic NCF              | 0.170  | 0.255  | Fixed hyperparameters, no tuning             |
| Tuned NCF              | 0.120  | 0.190  | Tuned using 5% sample and retrained          |
| Hybrid NCF + Content   | 0.117  | 0.186  | Uses recipe/user features + tuned hyperparams|


## Contact

If you'd like to reach out or collaborate, feel free to connect with me!

