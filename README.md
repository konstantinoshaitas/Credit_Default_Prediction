# Credit Default Prediction Project

This project aims to predict whether an individual will experience a 90-day past due delinquency or worse using various machine learning models. The models utilized include Logistic Regression, Decision Tree, Bagging Classifier, and Random Forest Classifier. The project follows a structured approach of data cleaning, preprocessing, model building, evaluation, and final prediction.

## Project Structure

1. **Data Loading and Preprocessing**
    - Load data from CSV files.
    - Drop unnecessary columns and standardize column names.
    - Handle missing values using two imputation methods:
        - Single Value Imputation (SVI)
        - Iterative Model-Based Imputation (IMBI)

2. **Model Building and Training**
    - Logistic Regression:
        - Train models using both SVI and IMBI datasets.
    - Decision Tree Classifier:
        - Train a single decision tree model.
    - Bagging Classifier:
        - Train a Bagging Classifier with varying numbers of base estimators to find the optimal number.
    - Random Forest Classifier:
        - Train a Random Forest Classifier with the optimal number of base estimators.

3. **Model Evaluation**
    - Evaluate models using accuracy scores and AUC scores.
    - Plot ROC curves to visualize the performance of each model.

4. **Final Model and Predictions**
    - Use the Random Forest Classifier, which performed the best, for final predictions on the test dataset.
    - Save the final predictions to a CSV file for submission.

## Key Findings

- The Random Forest Classifier with 400 base estimators achieved the highest accuracy (0.9376) and AUC score (0.8541).
- Iterative Model-Based Imputation (IMBI) provided better model performance compared to Single Value Imputation (SVI).
- Ensemble methods like Bagging and Random Forest significantly outperformed individual models like Logistic Regression and Decision Tree.

## How to Run the Project

1. **Clone the Repository**
    ```bash
    git clone https://github.com/konstantinoshaitas/Credit_24.git
    cd Credit_24
    ```

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Script**
    - Ensure the dataset files are in the `data` directory.
    ```bash
    python main.py
    ```

## Report Summary

The full report provides an in-depth analysis of the methodologies used, the reasoning behind model choices, detailed results, and final insights. It is available in the repository as `credit_score_report.pdf`.

## Authors

- Marco Maluf
- Konstantino Haitas

## Acknowledgements

This project was part of the Bocconi Machine Learning course (30412). We used various academic resources and lectures to guide our methodology and approach.

For more details, refer to the [report](credit_score_report.pdf) included in this repository.

---

Feel free to reach out for any questions or collaboration opportunities!

---

