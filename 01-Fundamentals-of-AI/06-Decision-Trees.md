# Decision Trees

## Overview

![Decision Tree Classifier](decision_tree.png)
*Decision tree classifier diagram with nodes for petal length and width, classifying samples into setosa, versicolor, and virginica*

**Decision trees** are a popular supervised learning algorithm for classification and regression tasks. They are known for their intuitive tree-like structure, which makes them easy to understand and interpret. In essence, a decision tree creates a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

### Real-World Example

Imagine you're trying to decide whether to play tennis based on the weather. A decision tree would break down this decision into a series of simple questions:
- Is it sunny?
- Is it windy?
- Is it humid?

Based on the answers to these questions, the tree would lead you to a final decision: **play tennis** or **don't play tennis**.

---

## Components of a Decision Tree

A decision tree comprises three main components:

### 1. Root Node
This represents the starting point of the tree and contains the entire dataset.

### 2. Internal Nodes
These nodes represent features or attributes of the data. Each internal node branches into two or more child nodes based on different decision rules.

### 3. Leaf Nodes
These are the terminal nodes of the tree, representing the final outcome or prediction.

---

## Building a Decision Tree

Building a decision tree involves **selecting the best feature to split the data at each node**. This selection is based on measures like:
- Gini impurity
- Entropy
- Information gain

These measures quantify the homogeneity of the subsets resulting from the split. The goal is to create splits that result in increasingly pure subsets, where the data points within each subset belong predominantly to the same class.

---

## Gini Impurity

**Gini impurity** measures the probability of misclassifying a randomly chosen element from a set. A lower Gini impurity indicates a more pure set.

### Formula

```
Gini(S) = 1 - Σ (pi)²
```

**Where:**
- **S** is the dataset
- **pi** is the proportion of elements belonging to class i in the set

### Example Calculation

Consider a dataset S with two classes: A and B. Suppose there are 30 instances of class A and 20 instances of class B.

**Step 1: Calculate proportions**
- Proportion of class A: `pA = 30 / (30 + 20) = 0.6`
- Proportion of class B: `pB = 20 / (30 + 20) = 0.4`

**Step 2: Calculate Gini impurity**
```
Gini(S) = 1 - (0.6² + 0.4²)
        = 1 - (0.36 + 0.16)
        = 1 - 0.52
        = 0.48
```

---

## Entropy

**Entropy** measures the disorder or uncertainty in a set. A lower entropy indicates a more homogeneous set.

### Formula

```
Entropy(S) = - Σ pi * log₂(pi)
```

**Where:**
- **S** is the dataset
- **pi** is the proportion of elements belonging to class i in the set

### Example Calculation

Using the same dataset S with 30 instances of class A and 20 instances of class B:

**Step 1: Calculate proportions**
- Proportion of class A: `pA = 0.6`
- Proportion of class B: `pB = 0.4`

**Step 2: Calculate entropy**
```
Entropy(S) = - (0.6 * log₂(0.6) + 0.4 * log₂(0.4))
           = - (0.6 * (-0.73697) + 0.4 * (-1.32193))
           = - (-0.442182 - 0.528772)
           = 0.970954
```

---

## Information Gain

**Information gain** measures the reduction in entropy achieved by splitting a set based on a particular feature. The feature with the highest information gain is chosen for the split.

### Formula

```
Information Gain(S, A) = Entropy(S) - Σ ((|Sv| / |S|) * Entropy(Sv))
```

**Where:**
- **S** is the dataset
- **A** is the feature used for splitting
- **Sv** is the subset of S for which feature A has value v

### Example Calculation

Consider a dataset S with 50 instances and two classes: A and B. Suppose we consider a feature F that can take on two values: 1 and 2.

**Distribution:**
- For F = 1: 30 instances (20 class A, 10 class B)
- For F = 2: 20 instances (10 class A, 10 class B)

**Step 1: Calculate entropy of entire dataset S**
```
Entropy(S) = - (30/50 * log₂(30/50) + 20/50 * log₂(20/50))
           = - (0.6 * log₂(0.6) + 0.4 * log₂(0.4))
           = - (0.6 * (-0.73697) + 0.4 * (-1.32193))
           = 0.970954
```

**Step 2: Calculate entropy for each subset**

For F = 1:
```
Proportion of class A: pA = 20/30 = 0.6667
Proportion of class B: pB = 10/30 = 0.3333
Entropy(S1) = - (0.6667 * log₂(0.6667) + 0.3333 * log₂(0.3333))
            = 0.9183
```

For F = 2:
```
Proportion of class A: pA = 10/20 = 0.5
Proportion of class B: pB = 10/20 = 0.5
Entropy(S2) = - (0.5 * log₂(0.5) + 0.5 * log₂(0.5))
            = 1.0
```

**Step 3: Calculate weighted average entropy**
```
Weighted Entropy = (|S1| / |S|) * Entropy(S1) + (|S2| / |S|) * Entropy(S2)
                 = (30/50) * 0.9183 + (20/50) * 1.0
                 = 0.55098 + 0.4
                 = 0.95098
```

**Step 4: Calculate information gain**
```
Information Gain(S, F) = Entropy(S) - Weighted Entropy
                       = 0.970954 - 0.95098
                       = 0.019974
```

---

## Building the Tree

The tree-building process follows these steps:

1. **Start with the root node** containing all data
2. **Select the best feature** based on Gini impurity, entropy, or information gain
3. **Create branches** for each possible value of that feature
4. **Divide the data** into subsets based on these branches
5. **Repeat recursively** for each subset until stopping criteria are met

### Stopping Criteria

The tree stops growing when one of the following conditions is satisfied:

#### Maximum Depth
The tree reaches a specified maximum depth, preventing it from becoming overly complex and potentially overfitting the data.

#### Minimum Number of Data Points
The number of data points in a node falls below a specified threshold, ensuring that the splits are meaningful and not based on very small subsets.

#### Pure Nodes
All data points in a node belong to the same class, indicating that further splits would not improve the purity of the subsets.

---

## Playing Tennis Example

![Playing Tennis Decision Tree](decision_tree_tennis.png)
*Decision tree diagram with nodes for Outlook, Temperature, Humidity, and Wind, classifying samples into Yes or No*

Let's examine the "Playing Tennis" example more closely to illustrate how a decision tree works in practice.

### Dataset

Imagine you have a dataset of historical weather conditions and whether you played tennis on those days:

| PlayTennis | Outlook_Overcast | Outlook_Rainy | Outlook_Sunny | Temperature_Cool | Temperature_Hot | Temperature_Mild | Humidity_High | Humidity_Normal | Wind_Strong | Wind_Weak |
|------------|------------------|---------------|---------------|------------------|-----------------|------------------|---------------|-----------------|-------------|-----------|
| No | False | True | False | True | False | False | False | True | False | True |
| Yes | False | False | True | False | True | False | False | True | False | True |
| No | False | True | False | True | False | False | True | False | True | False |
| No | False | True | False | False | True | False | True | False | False | True |
| Yes | False | False | True | False | False | True | False | True | False | True |
| Yes | False | False | True | False | True | False | False | True | False | True |
| No | False | True | False | False | True | False | True | False | True | False |
| Yes | True | False | False | True | False | False | True | False | False | True |
| No | False | True | False | False | True | False | False | True | True | False |
| No | False | True | False | False | True | False | True | False | True | False |

### Features

The dataset includes the following features:
- **Outlook**: Sunny, Overcast, Rainy
- **Temperature**: Hot, Mild, Cool
- **Humidity**: High, Normal
- **Wind**: Weak, Strong

The target variable is **Play Tennis**: Yes or No.

### Building Process

1. **Analyze the dataset** to identify features that best separate "Yes" from "No" instances

2. **Calculate information gain or Gini impurity** for each feature

3. **Select the best feature** - For instance, the algorithm might find that the **Outlook** feature provides the highest information gain

4. **Create the root node** with the Outlook feature, with three branches: Sunny, Overcast, and Rainy

5. **Divide the dataset** into three subsets based on these branches

6. **Repeat for each subset** - For example, in the "Sunny" subset, **Humidity** might provide the highest information gain, leading to another internal node with High and Normal branches

7. **Continue recursively** until stopping criteria are met

The final result is a tree-like structure with decision rules at each internal node and predictions (Play Tennis: Yes or No) at the leaf nodes.

---

## Data Assumptions

One of the advantages of decision trees is that they have **minimal assumptions about the data**:

### 1. No Linearity Assumption
Decision trees can handle both linear and non-linear relationships between features and the target variable. This makes them more flexible than algorithms like linear regression, which assume a linear relationship.

### 2. No Normality Assumption
The data does not need to be normally distributed. This contrasts with some statistical methods that require normality for valid inferences.

### 3. Handles Outliers
Decision trees are relatively robust to outliers. Since they partition the data based on feature values rather than relying on distance-based calculations, outliers are less likely to have a significant impact on the tree structure.

---

## Advantages and Limitations

### ✅ Advantages
- **Easy to understand and interpret** - Visual representation is intuitive
- **Requires little data preparation** - No normalization or scaling needed
- **Handles both numerical and categorical data**
- **Can model non-linear relationships**
- **Minimal data assumptions**
- **Robust to outliers**

### ⚠️ Limitations
- **Prone to overfitting** - Especially with deep trees
- **Unstable** - Small variations in data can result in different trees
- **Biased towards features with more levels** - In categorical variables
- **Not optimal for regression** - Better suited for classification

---

## Summary

Decision trees are versatile supervised learning algorithms that:

- Create **tree-like structures** for making predictions
- Use **splitting criteria** (Gini impurity, entropy, information gain) to build optimal trees
- Have **three main components**: root node, internal nodes, and leaf nodes
- Follow **stopping criteria** to prevent overfitting
- Have **minimal data assumptions**, making them flexible and easy to use

Key concepts:
- **Gini Impurity**: Measures probability of misclassification
- **Entropy**: Measures disorder or uncertainty
- **Information Gain**: Measures reduction in entropy from a split
- **Decision Boundary**: Created by splits at each internal node

These minimal assumptions contribute to decision trees' versatility, allowing them to be applied to a wide range of datasets and problems without extensive preprocessing or transformations.
