# Chapter 01 - Perceptron Learning Rule

This chapter covers the fundamental concepts of the Perceptron learning algorithm, one of the earliest machine learning algorithms for binary classification.

## 📥 Download Notebook

| Notebook | Description | Download |
|----------|-------------|----------|
| **PerceptronLearningRule.ipynb** | Complete implementation of the Perceptron learning algorithm with step-by-step explanations and visualizations | [📥 Download Notebook](https://github.com/MohElshamy1994/machine-learning-course-2025/raw/main/scripts/chapter01/PerceptronLearningRule.ipynb) |

## 📋 What You'll Learn

- Understanding the Perceptron algorithm
- Implementation from scratch in Python
- Visualization of the learning process
- Practical examples and applications
- Mathematical foundations and intuition

---

## 📓 Complete Notebook Content

### 🔸 Cell 1: Introduction
```markdown
# Perceptron Classification

Using this perceptron implementation, we can now initialize new Perceptron objects with a given learning rate, eta (𝜂), and the number of epochs, n_iter (passes over the training dataset).
```

### 🔸 Cell 2: Dataset Import
```markdown
### Import the dataset
- We will use the pandas library to load the Iris dataset directly from the UCI Machine Learning Repository into a Data Frame object and print the last five lines via the tail method to check that the data was loaded correctly:
- Go to https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data to check the details
```

### 🔸 Cell 3: Loading Data
```python
import os
import pandas as pd
s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
print('From URL:', s)
# utf-8 encoding is 
df = pd.read_csv(s,header=None,encoding='utf-8')
df.tail()
```

**Output:**
```
From URL: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
```

### 🔸 Cell 4: Class Labels Extraction
```markdown
### Extracting the class labels

- Next, we extract the first 100 class labels that correspond to the 50 Iris-setosa and 50 Iris-versicolor flowers
- convert the class labels into the two integer class labels, 1 (versicolor) and 0 (setosa), that we assign to a vector, y, 
- where the values method of a pandas DataFrame yields the corresponding NumPy representation.
```

### 🔸 Cell 5: Data Preprocessing and Visualization
```python
import matplotlib.pyplot as plt
import numpy as np
# select setosa and versicolor
y = df.iloc[0:100, 4].values

y = np.where(y == 'Iris-setosa', 0, 1)
print(y)
# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values
print(X)
# plot data
plt.scatter(X[:50, 0], X[:50, 1],color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],color='blue', marker='s', label='Versicolor')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()
```

**Output:**
```
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]

First few rows of feature matrix X:
[[5.1 1.4]
 [4.9 1.4]
 [4.7 1.3]
 [4.6 1.5]
 [5.  1.4]
 ...
```

This creates a scatter plot showing the separation between Iris-setosa (red circles) and Iris-versicolor (blue squares) based on sepal length and petal length features.

---

## 🚀 Key Concepts Covered

1. **Data Loading**: Import Iris dataset from UCI repository
2. **Data Preprocessing**: Extract features and convert class labels
3. **Visualization**: Create scatter plots to understand data distribution
4. **Perceptron Algorithm**: Implementation of the learning rule
5. **Classification**: Binary classification between two iris species

---

**💡 Tip**: Click the download link above to get the complete notebook with all cells, outputs, and visualizations. Run it in Jupyter Lab/Notebook to interact with the code!
