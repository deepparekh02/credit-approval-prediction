# Credit Approval Prediction
This project uses a logistic regression model to predict whether a loan application will be approved or not. The model is trained on a dataset of past loan applications, which includes various details about the applicant and the loan.

### Performance
The logistic regression model exhibits strong performance with the following metrics:

- AUC-ROC: 0.974096
- Precision: 0.918943
- Recall: 0.919142
- Accuracy: 0.919142

### Implementation Details
The script first loads the data and preprocesses it by removing unnecessary columns and recoding categorical variables. It then one-hot encodes the categorical features and assembles the quantitative variables into a single vector.

The logistic regression model is trained using a pipeline, which includes stages for assembling the features into a single vector and standardizing them. The pipeline also includes a cross-validation stage for hyperparameter tuning and model selection.

The performance of the model is evaluated using multiple metrics, including AUC-ROC, precision, recall, and accuracy.

### Requirements
- Python
- PySpark: Although the dataset is not large enough to benefit from a distributed computing environment, PySpark was  chosen for its ability to efficiently process and analyze large-scale data (and for me to gain experience using it).
- Numpy
- Dataset: The dataset used in this project is credit_approval_dataset.csv (taken from Kaggle at https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset). Each row in the dataset represents a loan application. The columns in the dataset are:
    - loan_id: The ID of the loan application.
    - education: The education level of the applicant.
    - self_employed: Whether the applicant is self-employed or not.
    - loan_status: Whether the loan was approved or not.
    - Quantitative variables: These are numerical columns such as 'income_annum', 'loan_amount', 'loan_term', etc. that are directly used as features in the model.




