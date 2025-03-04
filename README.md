![banner](https://github.com/user-attachments/assets/4b744be3-dda8-430e-8484-2953c8936662)

# Customer Segmentation with K-Means and Random Forest Classifier

This project applies **K-Means Clustering** and **Random Forest Classification** to segment bank customers based on behavioral data. The goal is to group similar customers and build a predictive model to classify new customers into these segments.

## Why Machine Learning is Effective for Customer Segmentation

Customer segmentation is crucial for marketing, product development, and business strategy. It involves grouping customers based on behaviors, preferences, and demographics, allowing for personalized strategies and better decision-making. Traditional segmentation methods are often manual, inefficient, and time-consuming, especially with large datasets. 

Leveraging machine learning:

- **Saves time** by automating segmentation.
- **Uncovers subtle patterns** in customer behavior.
- **Improves performance** over time as models learn.
- **Scales easily** for large datasets.
- **Increases accuracy**, improving predictions and strategies.
---

![streamlit](https://github.com/user-attachments/assets/64ccb1bf-3842-46cd-99bf-df22839300ce)

---

# Table of Contents

1. [Data Exploration & Preprocessing](#data-exploration--preprocessing)  
   - [Dataset](#dataset)  
   - [Preprocessing Steps](#preprocessing-steps)  
   - [EDA (Exploratory Data Analysis)](#eda-exploratory-data-analysis)  
   - [Feature Engineering](#feature-engineering)  

2. [Customer Segmentation with K-Means](#customer-segmentation-with-k-means)  
   - [Choosing Optimal K](#choosing-optimal-k)  
   - [Applying K-Means](#applying-k-means)  
   - [Assigning Clusters](#assigning-clusters)  
   - [Analyzing & Visualizing Clusters](#analyzing--visualizing-clusters)  
   - [Profiling Customers](#profiling-customers)  
   - [Saving K-Means Model & Clustered Customer Data](#saving-k-means-model--clustered-customer-data)  

3. [Unsupervised to Supervised: Building a Random Forest Classifier](#unsupervised-to-supervised-building-a-random-forest-classifier)  
   - [Model Selection & Comparison](#model-selection--comparison)  
   - [Hyperparameter Tuning (Grid Search)](#hyperparameter-tuning-grid-search)  
   - [Key Takeaways](#key-takeaways)  
   - [Fitting on Full Data & Saving Random Forest for Predictions](#fitting-on-full-data--saving-random-forest-for-predictions)  

4. [Model Deployment with Streamlit](#model-deployment-with-streamlit)  

5. [Results and Insights](#results-and-insights)  

6. [Conclusion](#conclusion)  

7. [How to Run](#how-to-run)  

8. [Technologies Used](#technologies-used)  

9. [License](#license)
    
---

## 1. Data Exploration & Preprocessing

### Dataset
The [dataset](https://www.kaggle.com/datasets/alifarahmandfar/customer-segmentation) contains **8,950 customer records** with 17 numerical features related to credit card usage, such as balance, purchase frequency, cash advances, and credit limits.

### Key Features

| Feature                     | Description                              |
|-----------------------------|------------------------------------------|
| **BALANCE**                  | Average balance per customer             |
| **PURCHASES**                | Total purchase amount                    |
| **ONEOFF_PURCHASES**         | Purchases made in a single transaction   |
| **INSTALLMENTS_PURCHASES**   | Purchases made in installments           |
| **CREDIT_LIMIT**             | Credit limit assigned to the customer    |
| **PAYMENTS**                 | Total amount paid by the customer        |
| **PRC_FULL_PAYMENT**         | Percentage of full payments made         |
| **TENURE**                   | Number of months the account has been active |
  
### Preprocessing Steps
- **Dropping Unnecessary Columns** and **Standardizing Names**
- **Handling Missing Values**: Imputation of missing data.
- **Outlier Detection & Treatment**: Identified and managed extreme values.
  
![outliers](https://github.com/user-attachments/assets/753313a3-c4ae-4656-9863-e5397e10410d)

### EDA (Exploratory Data Analysis)
A detailed analysis was performed to understand the data distribution, trends, and correlations. **Visualizations include:**

1. **Distribution of Key Features**: Histograms and boxplots were used to analyze feature distributions.
2. **Spending Patterns**: Scatter plots and pair plots provided insights into spending behavior.
3. **Correlation Matrix**: A heatmap was generated to explore relationships between features.

![corr_matrix](https://github.com/user-attachments/assets/14a20038-3f66-48a0-bf1b-abb929eb2ee1)

### Feature Engineering
- **Checked highly correlated or irrelevant features.**
- **Feature Scaling**: Applied `StandardScaler` to normalize numerical features for better clustering performance.
- **Principal Component Analysis (PCA):** Reduced dimensionality to **2 principal components** for better clustering visualization.
  
![pca_heatmap](https://github.com/user-attachments/assets/2ad6c284-7ab2-450e-a29d-efe21ea47366)

---

## 2. Customer Segmentation with K-Means

### Choosing Optimal Number of Clusters (K)
- **Elbow Method**: Determines the optimal number of clusters (k = 4)

![elbow](https://github.com/user-attachments/assets/880f4f93-f437-41ec-8078-2e1228256f66)

- **Silhouette Score**: Validates clustering quality (k=4, Silhouette Score: 0.3981)

### Applying K-Means to Group Customers Based on Similarities
```python
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(pca_df)
```

### Assigning Clusters 
```python
pca_df["cluster"] = kmeans_labels
df["cluster"] = kmeans_labels
```

### Analyzing & Visualizing Clusters
- **Cluster Visualization using PCA components**.
  
![clusters](https://github.com/user-attachments/assets/17b7fd83-7940-4577-9954-15d0d8636004)

- **Cluster Distribution**.

![cluster_distribution](https://github.com/user-attachments/assets/89e41a2c-aeff-4621-8af5-537dd0d6c7a9)

### Profiling Customers

| **Cluster**  | **Description**                                                      | **Possible Profile**                                                                                                               |
|--------------|----------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| **0: Cash Advance Users** | High balance and cash advance usage, moderate purchases.    | Likely credit-dependent users relying on cash advances. <br><br> May struggle with payments or prefer using credit over their own money. |
| **1: Low Spenders**       | Low balance, low purchases, and low overall engagement.          | Casual credit card users who don’t rely heavily on credit, potentially inactive or low-value customers for the bank. <br><br> Likely low-income or budget-conscious customers. |
| **2: High Spenders**      | High balance, large purchases, frequent use, and high credit limit. | High-value customers who spend a lot but pay on time (strong credit scores). <br><br> Good for the bank as they generate high transaction volume. |
| **3: Installment Users**  | Moderate balance with frequent installment purchases.             | Customers who prefer installment payments over one-off large transactions. Likely middle-class managing finances carefully. <br><br> Potentially good long-term customers if nurtured well. |

![cluster_feature_comparison](https://github.com/user-attachments/assets/76e0c5b0-e68d-40f8-b133-793135528918)

### Saving K-Means Model & Clustered Customer Data:

```python
joblib.dump(kmeans, "kmeans.pkl")
df.to_csv("clustered_customer_data.csv")
```
---

## 3. Unsupervised to Supervised: Building a Random Forest Classifier  

### Model Selection & Comparison  
Several models were tested and evaluated based on accuracy, precision, recall, and F1-score. 

| Model                | Accuracy  |
|----------------------|----------|
| **Random Forest**    | **95.14%** |
| Decision Tree       | 93.07% |
| SVM                 | 82.23% |
| KNN                 | 80.89% |
| Logistic Regression | 90.89% |
| Naïve Bayes         | 90.11% |

### Hyperparameter Tuning (Grid Search)
```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
```
![rf_feature_importance](https://github.com/user-attachments/assets/0f4e085a-fd49-48c6-8f99-3c7e64a00ecf)

### Key Takeaways:
- Random Forest was the best performer with the highest accuracy (95.14%) and balanced precision/recall.
- Hyperparameter tuning did not significantly change accuracy, indicating the model was already well-optimized.
- Logistic Regression performed surprisingly well for a simpler model, reaching 90.89% accuracy.
- Decision Tree performed slightly worse than Random Forest but still achieved high accuracy.
- SVM & KNN had lower accuracy, suggesting they may not be the best fit for this dataset.
 
### Fitting on Full Data & Saving Random Forest for Predictions:

```python
best_rf.fit(X, y)
with open('best_rf.pkl', 'wb') as f:
    pickle.dump(best_rf, f))
```
---

## 4. Model Deployment with Streamlit
The model was deployed using Streamlit for interactive customer segmentation. The app allows users to:
- Input customer attributes.
- View predicted customer segments.
- Explore visual segmentation insights interactively.

To run the Streamlit app:
```bash
streamlit run app.py
```
---

## 5. Results and Insights
- Successfully segmented customers into 4 distinct behavioral groups.
- PCA visualization provided insights into the clustering structure.
- The Random Forest classifier achieved high accuracy in predicting customer segments.
- Streamlit enabled interactive exploration of customer insights.

---

## 6. Conclusion
This project demonstrates the power of machine learning for customer segmentation, offering a scalable and automated approach over traditional methods.

---

## 7. How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Airam-DA/ml-customer-segmentation-with-k-means-and-random-forest.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## 8. Technologies Used
- **Python**
- **Pandas, NumPy** (Data Processing)
- **Matplotlib, Seaborn** (Visualization)
- **Scikit-Learn** (ML Models)
- **Streamlit** (Deployment)

## 9. License
This project is open-source and available under the [MIT License](LICENSE).
