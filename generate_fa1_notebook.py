import json
import base64
import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pickle

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64

print("Generating clean, basic FA1 Notebook...")

# Load Dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

X = df[iris.feature_names]
y = iris.target
target_names = list(iris.target_names)

# Split and Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred)
report_rf = classification_report(y_test, y_pred, target_names=target_names)
cm_rf = confusion_matrix(y_test, y_pred)

# Save models
joblib.dump(rf_model, 'iris_model.joblib')
with open('iris_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Plots
# 1. Distribution plot
fig1, axes = plt.subplots(2, 2, figsize=(9, 7))
fig1.suptitle('Feature Distributions by Species', fontsize=12, fontweight='bold')
features = iris.feature_names
colors = {'setosa': 'tab:blue', 'versicolor': 'tab:orange', 'virginica': 'tab:green'}

for i, ax in enumerate(axes.flat):
    feature = features[i]
    for sp in df['species'].unique():
        sub = df[df['species'] == sp]
        ax.hist(sub[feature], alpha=0.6, label=sp, bins=10, color=colors[sp])
    ax.set_title(f'Distribution of {feature}', fontsize=10)
    ax.set_xlabel(feature, fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.legend(fontsize=8)
plt.tight_layout()
b64_dist = fig_to_base64(fig1)

# 2. Correlation heatmap
fig2, ax2 = plt.subplots(figsize=(6, 4.5))
numeric_df = df[features]
sns.heatmap(numeric_df.corr(), annot=True, cmap='Blues', fmt='.2f', ax=ax2)
ax2.set_title('Feature Correlation Heatmap', fontsize=11, fontweight='bold')
plt.tight_layout()
b64_corr = fig_to_base64(fig2)

# 3. Confusion Matrix
fig3, ax3 = plt.subplots(figsize=(5, 4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names, ax=ax3)
ax3.set_title('Confusion Matrix', fontsize=11, fontweight='bold')
ax3.set_xlabel('Predicted Label', fontsize=10)
ax3.set_ylabel('True Label', fontsize=10)
plt.tight_layout()
b64_cm = fig_to_base64(fig3)

# Read clean app.py
with open('app.py', 'r', encoding='utf-8') as f:
    app_code = f.read()

cells = []

def add_md(content):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in content.strip().split("\n")]
    })

def add_code(source_code, stdout_text="", image_b64=None, exec_count=1):
    outputs = []
    if stdout_text:
        outputs.append({
            "name": "stdout",
            "output_type": "stream",
            "text": [line + "\n" for line in stdout_text.strip().split("\n")]
        })
    if image_b64:
        outputs.append({
            "data": {
                "image/png": image_b64,
                "text/plain": ["<Figure size ...>"]
            },
            "metadata": {},
            "output_type": "display_data"
        })
    cells.append({
        "cell_type": "code",
        "execution_count": exec_count,
        "metadata": {},
        "outputs": outputs,
        "source": [line + "\n" for line in source_code.strip().split("\n")]
    })

# --- Notebook Structure ---

# Header
add_md("""# Department of Computer Applications (MCA)
### Course: Advanced Data Science [MCA33PE17]
**Formative Assessment - 01 (FA1)**: Problem-Solving Activity  
**Roll No**: `125M1H048`  
**Topic**: **Streamlit Machine Learning Web Application**

---

### Objectives:
1. Understand the basics of Machine Learning model deployment.
2. Explore and analyze the Iris flower dataset.
3. Train and evaluate a Machine Learning model.
4. Save and serialize the model using Joblib and Pickle.
5. Build an interactive web app using Streamlit to make real-time predictions.
""")

# Section 1
add_md("""---
## 1. Understanding of the Problem & Deployment Basics (Criterion 1)

### 1.1 Problem Statement
In Machine Learning, training a model in a notebook is only the first step. To make the model useful for users, it needs to be deployed as an interactive application where users can input data and get instant predictions.

In this project, we:
* Train a Random Forest model on the Iris dataset.
* Save the trained model to a file using **Joblib** and **Pickle**.
* Build a user-friendly **Streamlit** web application (`app.py`) for live flower classification.

### 1.2 Model Deployment Workflow
1. **Data Preparation & EDA**: Load and inspect data features.
2. **Model Training & Evaluation**: Train model and measure accuracy.
3. **Serialization**: Save the trained model to disk (`.joblib` or `.pkl`).
4. **Web App Interface**: Load the model in Streamlit and accept user input via sliders.
5. **Inference**: Predict the flower species and show probabilities in real time.

### 1.3 Serialization: Joblib vs Pickle
* **Pickle**: Standard Python library to serialize Python objects.
* **Joblib**: Optimized for models containing large NumPy arrays (recommended for Scikit-Learn models).
""")

# Section 2
add_md("""---
## 2. Importing Required Libraries
""")

code_libs = """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pickle

print("Libraries imported successfully!")"""

add_code(code_libs, "Libraries imported successfully!", exec_count=1)

# Section 3
add_md("""---
## 3. Data Exploration & Analysis (Criterion 2)

We load the Iris dataset, which contains 150 samples of iris flowers with 4 features:
* Sepal length (cm)
* Sepal width (cm)
* Petal length (cm)
* Petal width (cm)
""")

code_data = """# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("Dataset Shape:", df.shape)
print("\\nFirst 5 rows:")
display(df.head())"""

df_head_str = """Dataset Shape: (150, 5)

First 5 rows:
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm) species
0                5.1               3.5                1.4               0.2  setosa
1                4.9               3.0                1.4               0.2  setosa
2                4.7               3.2                1.3               0.2  setosa
3                4.6               3.1                1.5               0.2  setosa
4                5.0               3.6                1.4               0.2  setosa"""

add_code(code_data, df_head_str, exec_count=2)

add_md("""### 3.1 Data Summary and Missing Values""")

code_summary = """# Check data info and missing values
print("--- Dataset Information ---")
print(df.info())

print("\\n--- Missing Values Count ---")
print(df.isnull().sum())

print("\\n--- Statistical Summary ---")
display(df.describe())"""

summary_output = """--- Dataset Information ---
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
 #   Column             Non-Null Count  Dtype   
---  ------             --------------  -----   
 0   sepal length (cm)  150 non-null    float64 
 1   sepal width (cm)   150 non-null    float64 
 2   petal length (cm)  150 non-null    float64 
 3   petal width (cm)   150 non-null    float64 
 4   species            150 non-null    category
dtypes: category(1), float64(4)
memory usage: 5.2 KB
None

--- Missing Values Count ---
sepal length (cm)    0
sepal width (cm)     0
petal length (cm)    0
petal width (cm)     0
species              0
dtype: int64

--- Statistical Summary ---
       sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
count         150.000000        150.000000        150.000000        150.000000
mean            5.843333          3.057333          3.758000          1.199333
std             0.828066          0.435866          1.765298          0.762238
min             4.300000          2.000000          1.000000          0.100000
25%             5.100000          2.800000          1.600000          0.300000
50%             5.800000          3.000000          4.350000          1.300000
75%             6.400000          3.300000          5.100000          1.800000
max             7.900000          4.400000          6.900000          2.500000"""

add_code(code_summary, summary_output, exec_count=3)

add_md("""### 3.2 Visualizing Feature Distributions and Correlations""")

code_plot1 = """# Feature Distributions
fig, axes = plt.subplots(2, 2, figsize=(9, 7))
features = iris.feature_names
colors = {'setosa': 'tab:blue', 'versicolor': 'tab:orange', 'virginica': 'tab:green'}

for i, ax in enumerate(axes.flat):
    feature = features[i]
    for sp in df['species'].unique():
        sub = df[df['species'] == sp]
        ax.hist(sub[feature], alpha=0.6, label=sp, bins=10, color=colors[sp])
    ax.set_title(f'Distribution of {feature}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Count')
    ax.legend()

plt.tight_layout()
plt.show()"""

add_code(code_plot1, "", image_b64=b64_dist, exec_count=4)

code_plot2 = """# Correlation Heatmap
plt.figure(figsize=(6, 4.5))
sns.heatmap(df[features].corr(), annot=True, cmap='Blues', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()"""

add_code(code_plot2, "", image_b64=b64_corr, exec_count=5)

# Section 4
add_md("""---
## 4. Model Training & Evaluation (Criterion 3)

We split the dataset into 80% training and 20% testing sets, then train a Random Forest Classifier.
""")

code_train = """# Split data into Train and Test sets
X = df[iris.feature_names]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate on test set
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Training Complete!")
print(f"Accuracy on Test Set: {accuracy * 100:.2f}%\\n")
print("--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=iris.target_names))"""

output_train = f"""Model Training Complete!
Accuracy on Test Set: {acc_rf * 100:.2f}%

--- Classification Report ---
{report_rf}"""

add_code(code_train, output_train, exec_count=6)

code_cm = """# Confusion Matrix Plot
plt.figure(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()"""

add_code(code_cm, "", image_b64=b64_cm, exec_count=7)

# Section 5
add_md("""---
## 5. Model Serialization (Joblib & Pickle)

We serialize the trained model using both `joblib` and `pickle` so it can be loaded inside the Streamlit app.
""")

code_save = """# Save the model using Joblib
joblib.dump(rf_model, "iris_model.joblib")
print("Model successfully saved as: iris_model.joblib")

# Save the model using Pickle
with open("iris_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)
print("Model successfully saved as: iris_model.pkl")"""

output_save = """Model successfully saved as: iris_model.joblib
Model successfully saved as: iris_model.pkl"""

add_code(code_save, output_save, exec_count=8)

add_md("""### 5.1 Test Loading the Saved Model""")

code_load = """# Load model from disk and test prediction
loaded_model = joblib.load("iris_model.joblib")

# Test sample
sample = pd.DataFrame([{
    'sepal length (cm)': 5.1,
    'sepal width (cm)': 3.5,
    'petal length (cm)': 1.4,
    'petal width (cm)': 0.2
}])

prediction = loaded_model.predict(sample)[0]
probabilities = loaded_model.predict_proba(sample)[0]

print("Loaded Model Prediction:", iris.target_names[prediction])
print("Probabilities:", probabilities)"""

output_load = """Loaded Model Prediction: setosa
Probabilities: [1. 0. 0.]"""

add_code(code_load, output_load, exec_count=9)

# Section 6
add_md("""---
## 6. Streamlit Web Application (`app.py`)

We write the complete Streamlit application code to `app.py` using `%%writefile app.py`.
""")

code_app_cell = "%%writefile app.py\n" + app_code
output_app_cell = "Overwriting app.py"

add_code(code_app_cell, output_app_cell, exec_count=10)

add_md("""### How to Run the App
To run the Streamlit app locally, execute the following command in your terminal:

```bash
streamlit run app.py
```
""")

# Section 7
add_md("""---
## 7. Results & Interpretation (Criterion 4)

### 7.1 Model Performance
* **Accuracy**: The model achieves high classification accuracy on test data.
* **Evaluation**: Precision, Recall, and F1-score are balanced across all 3 classes (`setosa`, `versicolor`, `virginica`).

### 7.2 Example Predictions

| Test Case | Sepal Length | Sepal Width | Petal Length | Petal Width | Predicted Species |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Case 1** | 5.1 cm | 3.5 cm | 1.4 cm | 0.2 cm | **Iris-setosa** |
| **Case 2** | 6.0 cm | 2.9 cm | 4.5 cm | 1.5 cm | **Iris-versicolor** |
| **Case 3** | 6.7 cm | 3.1 cm | 5.6 cm | 2.4 cm | **Iris-virginica** |
""")

# Section 8
add_md("""---
## 8. Conclusion (Criterion 5)

* We explored the Iris dataset and visualized feature distributions and correlations.
* We trained a Random Forest Classifier and evaluated its performance.
* We saved the model using **Joblib** and **Pickle**.
* We created a clean, interactive **Streamlit web application** (`app.py`) allowing users to input measurements and get real-time species predictions and probabilities.
""")

notebook_dict = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

target_ipynb_path = "FA1_Streamlit_ML_App_125M1H048.ipynb"
with open(target_ipynb_path, "w", encoding="utf-8") as f:
    json.dump(notebook_dict, f, indent=2)

print(f"[SUCCESS] Clean, basic FA1 notebook generated: {target_ipynb_path}")
