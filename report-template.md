# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
The purpose of this analysis is to evaluate the performance of a machine learning model for classifying loans as either "Healthy loan" or "High-risk loan." The model's precision, recall, and accuracy scores are assessed to determine its effectiveness in predicting loan classifications.

* Explain what financial information the data was on, and what you needed to predict.
The goal was to predict the classification of loans as either "Healthy loan" or "High-risk loan." The model aimed to assess the risk associated with each loan and provide insights to aid in decision-making processes.

* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
To gain insight into the distribution of loan classifications, the value_counts() function can be used. 

* Describe the stages of the machine learning process you went through as part of this analysis.
Data Loading and Preprocessing: The loan data was loaded into a suitable data structure, and any necessary preprocessing steps, such as handling missing values or encoding categorical variables, were performed.
Feature Selection and Engineering: Relevant features were selected or engineered to enhance the predictive power of the model.
Model Selection: A suitable machine learning algorithm, such as logistic regression have been chosen based on the nature of the problem and data.
Model Training and Evaluation: The selected model was trained on the prepared data and evaluated using appropriate evaluation metrics.
Fine-tuning and Validation: Hyperparameter tuning and model validation were conducted to optimize the model's performance and ensure its generalization ability.
Final Model Deployment: Once satisfied with the model's performance, it could be deployed to make predictions on new loan data.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).
The analysis mentioned using a machine learning model like Logistic Regression to predict loan classifications. Additionally, the analysis mentions resampling, specifically the RandomOverSampler module from the imbalanced-learn library. Resampling methods are commonly employed to address class imbalance in classification tasks and can help improve the model's ability to capture minority class samples.

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
The results:

Accuracy Score: The model achieves an accuracy of 0.99, indicating that it correctly predicts the loan classification for 99% of the samples.

Precision Score: For the "Healthy loan" class, the precision is 1.00, which means that when the model predicts a loan as "Healthy loan," it is correct 100% of the time. For the "High-risk loan" class, the precision is 0.87, indicating that when the model predicts a loan as "High-risk loan," it is correct 87% of the time.

Recall Score: The recall for the "Healthy loan" class is 1.00, suggesting that the model identifies all "Healthy loan" samples correctly. For the "High-risk loan" class, the recall is 0.91, indicating that the model captures 91% of the actual "High-risk loan" samples.


* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
Accuracy: The model achieves an accuracy score of 1.00, indicating that it correctly predicts the loan classification for all samples in the resampled dataset.

Precision: For the "Healthy loan" class, the precision is 1.00, which means that when the model predicts a loan as "Healthy loan," it is correct 100% of the time. For the "High-risk loan" class, the precision is 0.87, indicating that when the model predicts a loan as "High-risk loan," it is correct 87% of the time.

Recall: The recall score for the "Healthy loan" class is 0.99, suggesting that the model identifies 99% of the actual "Healthy loan" samples correctly. For the "High-risk loan" class, the recall is 1.00, indicating that the model captures all the actual "High-risk loan" samples.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:

* Which one seems to perform best? How do you know it performs best?
Comparing the two models:
Model 1:
Accuracy: 99%
Precision (High-risk loan): 87%
Recall (High-risk loan): 91%

Model 2:
Accuracy: 100%
Precision (High-risk loan): 87%
Recall (High-risk loan): 100%

Model 2 performs the best

* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )
If you do not recommend any of the models, please justify your reasoning.
Factors to consider when choosing the best model in-between Model 1 and Model 2:

Accuracy: Both models demonstrate high accuracy, but Model 2 achieves a perfect accuracy score of 100%, indicating that it predicts loan classifications correctly for all samples in the resampled dataset.

Precision: The precision for the "High-risk loan" class is the same for both models, at 87%. This means that when the models predict a loan as "High-risk loan," they are correct 87% of the time.

Recall: Model 2 achieves a perfect recall score of 100% for the "High-risk loan" class, capturing all actual high-risk loans. In contrast, Model 1 has a recall score of 91% for the "High-risk loan" class, indicating that it captures 91% of the actual high-risk loans.

Considering these factors, Model 2 appears to be the better choice for loan classification. It achieves a higher overall accuracy, maintains the same precision for high-risk loans, and has a perfect recall for high-risk loans, indicating its ability to identify all high-risk loan samples. This makes Model 2 more robust and reliable for assessing loan risk and making informed loan decisions.

However, it's important to note that the final decision should consider other factors such as the specific business requirements, the potential consequences of false positives or false negatives, and the availability of resources for implementing and maintaining the chosen model.
