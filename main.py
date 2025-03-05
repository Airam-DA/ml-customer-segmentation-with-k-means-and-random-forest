# ===================== Handling =====================
import pandas as pd
import numpy as np
from scipy.stats import zscore

# ===================== Visualization =====================
import matplotlib.pyplot as plt
import seaborn as sns

# ===================== Preprocessing =====================
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ===================== Dimensionality Reduction & Clustering =====================
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# ===================== Train-Test Split & Model Selection =====================
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# ===================== ML Models =====================
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

import xgboost as xgb 

# ===================== Evaluation Metrics =====================
from sklearn import tree, metrics
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, 
    average_precision_score, f1_score, fbeta_score, classification_report, 
    confusion_matrix
)


# ------------------------------
# Data Preparation Functions
# ------------------------------

def load_data(filepath):
    """Loads customer dataset from a CSV file."""
    return pd.read_csv(filepath)

def explore_data(df):
    """Initial EDA"""
    print(df.info())
    display(df.head())
    print("\nSummary Statistics:")
    display(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum()/len(df)*100)
    print("\nDuplicated Values:")
    print(df.duplicated().sum())
    duplicates = df.duplicated().sum()
    percentage = df.duplicated().sum() / df.shape[0] * 100
    print(f'{duplicates} rows contain duplicates amounting to {percentage.round(2)}% of the total data.')
    print (f"\nShape: {df.shape}")
    
def drop_column(df, column_name):
    """Drops a specified column from the DataFrame."""
    df = df.drop(columns=[column_name], axis=1)
    return df

def clean_column_names(df):
    """Cleans column names by stripping spaces, converting to lowercase, 
    and replacing spaces with underscores. Then prints the updated columns."""
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=True)
    print("Updated Columns:", df.columns.tolist())
    return df

def fill_na_with_mean(df, columns):
    """Fills NaN values in the specified columns with their mean."""
    df[columns] = df[columns].fillna(df[columns].mean())
    return df

def normalize_and_plot_outliers(df, filename="outliers.png"):
    """Normalizes numeric columns using MinMaxScaler and plots a boxplot to visualize outliers.
    Saves the figure as a PNG file."""
    df_numeric = df.select_dtypes(include=['number'])
    
    normalized_data = MinMaxScaler().fit_transform(df_numeric)
    df_normalized = pd.DataFrame(normalized_data, columns=df_numeric.columns)
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_normalized, palette="coolwarm")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename, format='png', dpi=300)
    plt.show()

    return df_normalized 

def outlier_percentage_iqr(df):   
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))
    return (outliers.sum() / len(df)) * 100

def outlier_percentage_zscore(df, threshold=3):
    """Calculates the percentage of outliers in numeric columns using the Z-score method.
    Args:
        df (pd.DataFrame): DataFrame with numerical columns.
        threshold (float): Z-score threshold (default is 3).
    
    Returns:
        pd.Series: Percentage of outliers per column.
    """
    z_scores = df.select_dtypes(include=[np.number]).apply(zscore)
    outliers = (z_scores.abs() > threshold)
    return (outliers.sum() / len(df)) * 100  


# ------------------------------
# EDA Functions
# ------------------------------

def plot_kde_all_columns(df, figsize=(30, 45), cols_per_row=3):
    """Plots KDE (Kernel Density Estimate) for each column in the DataFrame."""
    num_columns = len(df.columns)
    num_rows = (num_columns // cols_per_row) + (num_columns % cols_per_row > 0)
    
    plt.figure(figsize=figsize)
    for i, col in enumerate(df.columns):
        ax = plt.subplot(num_rows, cols_per_row, i + 1) 
        sns.kdeplot(df[col], ax=ax, palette="coolwarm") 
        ax.set_xlabel(col)  
    plt.tight_layout()  
    plt.show()
    
def plot_pairplot(df, palette="coolwarm"):
    """Generates a pairplot for the numeric columns in the DataFrame."""
    df_numeric = df.select_dtypes(include=[np.number])

    sns.pairplot(df_numeric, palette=palette)
    plt.show()
    
def plot_corr_matrix(df, filename="corr_matrix.png", figsize=(15,6), dpi=80):
    """Generates a correlation matrix heatmap for the numeric columns in the DataFrame."""
    corr_matrix = df.corr()
    mask = np.triu(np.ones_like(corr_matrix))
    
    plt.figure(figsize=figsize, dpi=dpi)
    sns.heatmap(corr_matrix, annot=True, mask=mask, fmt=".2f", cmap="coolwarm")
    plt.tight_layout()
    plt.savefig(filename, format='png', dpi=300)
    plt.show()
    

# ------------------------------
# PCA Functions
# ------------------------------

def apply_pca_with_scaling(df, n_components=2):
    """Scales the features and applies PCA to reduce the dimensionality of the DataFrame."""
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_df, columns=df.columns)
    
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_df)
    pca_df = pd.DataFrame(data=pca_data, columns=[f"PCA{i+1}" for i in range(n_components)])
    
    print("\nExplained variance per component:", pca.explained_variance_ratio_)
    print("Total explained variance:", sum(pca.explained_variance_ratio_))
    print("\nPCA Components (as rows, original features as columns):")
    display(pd.DataFrame(pca.components_, columns=df.columns))
    
    plt.matshow(pca.components_, cmap='coolwarm')
    plt.yticks([i for i in range(n_components)], [f"{i+1}st component" for i in range(n_components)])
    plt.colorbar()
    plt.xticks(range(len(df.columns)), df.columns, rotation=90)
    plt.tight_layout()
    plt.savefig('pca_heatmap.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

    return pca_df


# ------------------------------
# Optimal K Functions
# ------------------------------

def scale_and_elbow_method(df, range_val=range(1, 15)):
    """Scales the features of the DataFrame and evaluates inertia for different values of K using the Elbow Method."""
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_df, columns=df.columns)
    inertia = []
    for i in range_val:
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(scaled_df)
        inertia.append(kmeans.inertia_)
    
    plt.figure(figsize=(7, 4))
    plt.plot(range_val, inertia, 'k-o')
    plt.xlabel('K Values')
    plt.ylabel('Inertia')
    plt.title('Elbow Method using Inertia')
    plt.tight_layout()
    plt.savefig('elbow.png', format='png', dpi=300)
    plt.show()

    return scaled_df
    
def evaluate_silhouette(df, K_range=range(4, 10)):
    """Evaluates silhouette score for different K values and returns the optimal K."""
    silhouette_scores = []
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(df)
        score = silhouette_score(df, kmeans.labels_)
        silhouette_scores.append(score)
        print(f"k={k}, Silhouette Score: {score:.4f}")
    
    best_k = K_range[silhouette_scores.index(max(silhouette_scores))]
    print(f"\nOptimal k based on Silhouette Score: {best_k}")
    return best_k, silhouette_scores


# ------------------------------
# Clustering Functions
# ------------------------------

def apply_kmeans_to_pca(pca_df, df, n_clusters=4):
    """Applies K-Means clustering to PCA-reduced data and adds cluster labels to the original and PCA dataframes."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(pca_df) 
    
    pca_df["cluster"] = kmeans_labels
    df["cluster"] = kmeans_labels
    
    display(pca_df)
    display(df)
    
    return kmeans, pca_df, df


# ------------------------------
# Classification Functions
# ------------------------------

def evaluate_models(X_train, y_train, X_test, y_test):
    """
    Trains and evaluates multiple classification models, printing accuracy and classification report.
    Args:
    - X_train: Features for training
    - y_train: Target variable for training
    - X_test: Features for testing
    - y_test: Target variable for testing
    """
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": GaussianNB(),   
    }

    for name, model in models.items():
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)  
        acc = accuracy_score(y_test, y_pred) 
        
        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        print("-" * 58)

def tune_and_evaluate_rf(X_train, y_train, X_test, y_test):
    """
    Perform hyperparameter tuning using GridSearchCV for RF and evaluate performance.
    Args:
    - X_train: Features for training
    - y_train: Target variable for training
    - X_test: Features for testing
    - y_test: Target variable for testing
    """
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
    print("Best Parameters:", best_params)

    best_rf = RandomForestClassifier(**best_params, random_state=42)
    best_rf.fit(X_train, y_train)

    y_pred = best_rf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Tuned Random Forest Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))

    feature_importances = best_rf.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    print(importance_df)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=importance_df['Importance'], y=importance_df['Feature'], palette="coolwarm")
    plt.title("Feature Importance in Random Forest")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig('rf_feature_importance.png', format='png', dpi=300)
    plt.show()
    
    return best_rf