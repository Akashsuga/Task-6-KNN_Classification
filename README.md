# Task-6-KNN_Classification
# K-Nearest Neighbors (KNN) Classification on Iris Dataset
## Overview

This project implements the K-Nearest Neighbors (KNN) algorithm to classify Iris species using the classic Iris dataset. The workflow includes data preprocessing, model training, hyperparameter tuning, evaluation, and visualization of results including decision boundaries using PCA.

---
## Repository Structure

- **Iris.csv** — Original dataset
- **task_6_knn_classifier_iris.py** — Python script for K-Nearest Neighbors (KNN) Classification on Iris Dataset
- **Visual Outputs** — Folder containing all visual plots
- **task 6.pdf** — Given task
- **README.md** — This documentation

---

## Project Content  
- Data Loading and Preprocessing  
- Feature Scaling  
- Train-Test Split  
- Hyperparameter Tuning (finding best K)  
- Model Training and Evaluation  
- Confusion Matrix Visualization  
- Decision Boundary Visualization using PCA  

## Steps  
1. Load the Iris dataset and remove unnecessary columns.  
2. Scale features using StandardScaler for uniformity.  
3. Split the data into training and test sets.  
4. Train KNN models with different K values and evaluate accuracy.  
5. Select the best K based on accuracy and train the final model.  
6. Generate confusion matrix and classification report for evaluation.  
7. Use PCA to reduce features to 2D and plot decision boundaries.  

---

## Visuals & Descriptions

### 1. K Value vs Accuracy Plot  ![K value vs Accuracy](https://github.com/user-attachments/assets/882c13df-dd3b-4b46-9126-10f293f11e04)

*Description:*  
This plot shows the classification accuracy on the test set for different values of K (from 1 to 20). It helps identify the optimal number of neighbors for the KNN classifier.  
*Insight:*  
We observe that accuracy varies with K, typically peaking around a certain value (in this case, best K found at **K=5**). Too small or too large values of K can reduce accuracy due to overfitting or underfitting respectively.

---

### 2. Confusion Matrix Heatmap  ![Confusion Matrix](https://github.com/user-attachments/assets/e30f652f-7555-4ce3-9ff7-1d0b9fc7032e)

*Description:*  
This heatmap shows the confusion matrix of the final KNN model on the test data. It visualizes correct and incorrect classifications for each Iris species.  
*Insight:*  
High values along the diagonal indicate good classification performance. Some misclassifications may occur between similar species (e.g., *Iris versicolor* and *Iris virginica*), which is common due to their feature overlap.

---

### 3. Decision Boundary Plot (PCA Transformed Features)  
![KNN Decision Boundary (PCA)](https://github.com/user-attachments/assets/0daf5d6d-ebe8-4fe1-85b6-665f8cd5d5b9)

*Description:*  
This plot displays the decision boundaries learned by the KNN classifier in the 2D PCA-transformed feature space. Points are colored by their true species label.  
*Insight:*  
The decision boundaries clearly separate the three classes in this reduced dimensional space. Overlaps or fuzzy boundaries may appear between classes that share similar characteristics, highlighting the challenge of class separation in feature space.

---

## Insights and Patterns

- **Optimal K:** The model performs best around K=5, balancing bias and variance.
- **Feature Scaling:** Standardization was critical to ensure distance metrics in KNN worked properly.
- **Class Overlap:** Misclassifications mostly occur between *Iris versicolor* and *Iris virginica*, indicating feature similarity.
- **Dimensionality Reduction:** PCA reveals that much of the variance can be captured in two components, allowing for effective visualization.
- **Model Robustness:** KNN shows stable performance on this dataset, benefiting from its simplicity and non-parametric nature.

---

## Anomalies or Challenges

- Some points close to class boundaries are misclassified, reflecting natural ambiguity or noise in measurements.
- PCA transformation reduces feature space but may discard some information, slightly affecting classification when done before modeling.
- Choosing K requires tuning; arbitrary selection can lead to suboptimal results.

---

## Conclusion

The K-Nearest Neighbors algorithm effectively classifies Iris species with high accuracy when combined with proper data preprocessing and hyperparameter tuning. Visualizations demonstrate the model's performance and class separability in feature space. Despite minor misclassifications due to overlapping features, KNN remains a simple yet powerful tool for supervised classification problems.
