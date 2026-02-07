# Anomaly Detection

## Overview

![Anomaly Detection](images/anomaly_detection.png)
*Scatter plot showing blue normal data points and red anomaly data points*

**Anomaly detection**, also known as **outlier detection**, is a crucial task in unsupervised learning. It identifies data points that deviate significantly from normal behavior within a dataset.

### What Are Anomalies?

These anomalous data points, often called **outliers**, can indicate critical events such as:
- Fraudulent activities
- System failures
- Medical emergencies
- Security breaches
- Equipment malfunctions

---

## Security System Analogy

Think of it like a **security system that monitors a building**:

1. **Learning Phase**: The system learns the normal activity patterns, such as people entering and exiting during business hours

2. **Detection Phase**: It raises an alarm if it detects something unusual, like someone trying to break in at night

Similarly, anomaly detection algorithms:
- Learn the normal patterns in data
- Flag any deviations as potential anomalies

---

## Types of Anomalies

Anomalies can be broadly categorized into three types:

### 1. Point Anomalies

**Definition:** Individual data points that significantly differ from the rest of the data.

**Examples:**
- A sudden spike in network traffic
- An unusually high credit card transaction amount
- A single temperature reading far outside the normal range
- An unexpected system error log

**Characteristic:** Stand out individually from the normal data distribution.

---

### 2. Contextual Anomalies

**Definition:** Data points considered anomalous within a specific context but not necessarily in isolation.

**Examples:**
- A temperature reading of 30°C might be:
  - **Expected** in summer
  - **Anomalous** in winter
- High website traffic might be:
  - **Normal** during a sale event
  - **Anomalous** at 3 AM on a weekday

**Characteristic:** Depend on the context (time, location, season, etc.) for interpretation.

---

### 3. Collective Anomalies

**Definition:** A group of data points that collectively deviate from normal behavior, even though individual data points might not be considered anomalous.

**Examples:**
- A sudden surge in login attempts from multiple unknown IP addresses could indicate a coordinated attack
- A sequence of small transactions that together indicate money laundering
- A pattern of system calls that individually seem normal but collectively indicate malware

**Characteristic:** The anomaly is in the pattern or collection, not individual points.

---

## Anomaly Detection Techniques

Various techniques are employed for anomaly detection:

### 1. Statistical Methods

**Approach:** Assume that normal data points follow a specific statistical distribution (e.g., Gaussian distribution).

**Identification:** Outliers are data points that deviate significantly from this distribution.

**Examples:**
- **Z-score**: Measures how many standard deviations a point is from the mean
- **Modified Z-score**: Uses median absolute deviation for robustness
- **Boxplots**: Uses interquartile range to identify outliers

**Pros:**
- Simple to implement
- Well-understood theoretical foundation
- Fast for simple distributions

**Cons:**
- Assumes specific data distribution
- May not work well with complex, multi-modal data

---

### 2. Clustering-Based Methods

**Approach:** Group similar data points together and identify outliers.

**Identification:** Outliers are data points that:
- Do not belong to any cluster
- Belong to small, sparse clusters
- Are far from cluster centers

**Examples:**
- K-means clustering with distance threshold
- DBSCAN (Density-Based Spatial Clustering)
- Hierarchical clustering

**Pros:**
- No assumption about data distribution
- Can detect clusters of anomalies
- Works well for spatial data

**Cons:**
- Sensitive to parameter selection (e.g., number of clusters)
- Computational complexity for large datasets

---

### 3. Machine Learning-Based Methods

**Approach:** Utilize machine learning algorithms to learn patterns from normal data.

**Identification:** Outliers are data points that do not conform to learned patterns.

**Examples:**
- One-Class SVM
- Isolation Forest
- Local Outlier Factor (LOF)
- Autoencoders
- Deep learning approaches

**Pros:**
- Can capture complex patterns
- Adapt to data characteristics
- High accuracy with sufficient data

**Cons:**
- Require training data
- Can be computationally expensive
- May require parameter tuning

---

## One-Class SVM

![One-Class SVM](images/one_class_svm.png)
*Scatter plot showing anomaly detection with One-Class SVM, highlighting normal data in green and anomalies in red, enclosed by a decision boundary*

**One-Class SVM** is a machine learning algorithm specifically designed for anomaly detection.

### How It Works

1. **Learn a Boundary**: It learns a boundary that encloses the normal data points

2. **Identify Outliers**: Any data point falling outside this boundary is identified as an outlier

### Analogy

It's like **drawing a fence around a sheep pen**:
- Any sheep found outside the fence is likely an anomaly
- The fence represents the decision boundary
- Normal sheep stay within the fence

### Key Features

**Kernel Functions:** One-Class SVM can handle **non-linear relationships** using kernel functions, similar to SVMs used for classification.

**Common Kernels:**
- **RBF (Radial Basis Function)**: Most popular for One-Class SVM
- **Polynomial**: For polynomial relationships
- **Linear**: For linearly separable data

### Parameters

**ν (nu):** Controls the trade-off between:
- Maximizing the distance from the origin (in feature space)
- The fraction of support vectors

**Typical values:** Between 0 and 1 (e.g., 0.1 means ~10% of data expected to be outliers)

### Advantages

- Handles high-dimensional data well
- Can model complex, non-linear boundaries
- Based on solid mathematical foundation

### Limitations

- Computationally expensive for large datasets
- Sensitive to kernel and parameter selection
- Requires feature scaling

---

## Isolation Forest

![Isolation Forest](images/isolation_forest.png)
*Scatter plot showing Isolation Forest Anomaly Detection with blue normal data points and red anomalies*

**Isolation Forest** is a popular anomaly detection algorithm that isolates anomalies by randomly partitioning the data and constructing isolation trees.

### Core Principle

**Key Insight:** Anomalies, being **"few and different,"** are easier to isolate from the rest of the data and tend to have shorter paths in isolation trees.

### Analogy

It's like playing a game of **"20 questions"**:
- If you can identify an object with very few questions, it's likely unusual (an anomaly)
- Common objects require more questions to identify (normal data)

---

## How Isolation Forest Works

### Step 1: Random Partitioning

The algorithm works by **recursively partitioning the data** until each data point is isolated in its own leaf node.

**At each step:**
1. A **random feature** is selected
2. A **random split value** is chosen (between min and max values of that feature)
3. Data is divided into two subsets based on this split

### Step 2: Build Multiple Trees

This process is repeated to create multiple isolation trees (ensemble approach).

**Why multiple trees?**
- Reduces randomness effects
- Provides more robust anomaly scores
- Improves accuracy

### Step 3: Calculate Path Length

For each data point, measure the **path length** (number of edges from root to leaf) in each tree.

**Key observation:**
- **Anomalies**: Shorter average path lengths (easier to isolate)
- **Normal points**: Longer average path lengths (require more partitions to isolate)

---

## Anomaly Score Calculation

The anomaly score for a data point **x** is calculated as:

```python
score(x) = 2^(-E(h(x)) / c(n))
```

### Components

**Where:**

**E(h(x))**: Average path length of data point x in a collection of isolation trees

**c(n)**: Average path length of unsuccessful search in a Binary Search Tree (BST) with n nodes. This serves as a normalization factor

**n**: Number of data points

### Normalization Factor c(n)

```python
c(n) = 2 * H(n-1) - (2 * (n-1) / n)
```

Where **H(n)** is the harmonic number, approximately equal to ln(n) + 0.5772

### Score Interpretation

| Score Range | Interpretation |
|-------------|----------------|
| **Close to 1** | High likelihood of being an anomaly |
| **Around 0.5** | Data point is likely normal |
| **Below 0.5** | Very likely normal |

**Rule of thumb:**
- Scores > 0.6: Consider as anomalies
- Scores between 0.5-0.6: Ambiguous
- Scores < 0.5: Normal data

---

## Advantages of Isolation Forest

- ✅ **Efficient**: Linear time complexity O(n)
- ✅ **Scalable**: Works well with large datasets
- ✅ **No assumptions**: Doesn't require data distribution assumptions
- ✅ **Few parameters**: Easy to use with minimal tuning
- ✅ **Handles high dimensions**: Works well in high-dimensional spaces

---

## Local Outlier Factor (LOF)

![LOF Anomaly Detection](images/local_outlier_factor.png)
*Scatter plot showing LOF Anomaly Detection with blue normal data points and red anomalies*

**Local Outlier Factor (LOF)** is a density-based algorithm designed to identify outliers in datasets by comparing the **local density** of a data point to that of its neighbors.

### Key Strength

It is particularly effective in detecting anomalies in regions where the **density of points varies significantly**.

---

## LOF Analogy

Think of it like **identifying a house in a sparsely populated area** compared to a densely populated neighborhood:

- The **isolated house** in a region with fewer houses is more likely to be an anomaly
- A house in a dense neighborhood is considered normal

Similarly, in data terms:
- A point with a **lower local density** than its neighbors is considered an outlier
- A point with **similar density** to neighbors is normal

---

## LOF Score Calculation

The LOF score for a data point **p** is calculated using the following formula:

```python
LOF(p) = (Σ lrd(o) / k) / lrd(p)
```

### Components

**Where:**

**lrd(p)**: The **local reachability density** of data point p

**lrd(o)**: The local reachability density of data point o, one of the k nearest neighbors of p

**k**: The number of nearest neighbors (typically 5-20)

### Score Interpretation

| LOF Score | Interpretation |
|-----------|----------------|
| **~1.0** | Similar density to neighbors (normal) |
| **< 1.0** | Higher density than neighbors (very normal) |
| **> 1.0** | Lower density than neighbors (potential outlier) |
| **>> 1.0** | Much lower density (likely outlier) |

**Threshold:** Typically, LOF > 1.5 or 2.0 indicates an anomaly.

---

## Local Reachability Density

The **local reachability density (lrd(p))** for a data point p is defined as:

```python
lrd(p) = 1 / (Σ reach_dist(p, o) / k)
```

### Components

**Where:**

**reach_dist(p, o)**: The **reachability distance** from p to o

**Definition of reachability distance:**
```python
reach_dist(p, o) = max(k_distance(o), actual_distance(p, o))
```

**k_distance(o)**: The distance to the kth nearest neighbor of o

### Intuition

**Why use reachability distance?**

- **Dense regions**: Points have lower reachability distances (many close neighbors)
- **Sparse regions**: Points have higher reachability distances (fewer close neighbors)

This ensures that:
- Points in dense regions have **higher local reachability density**
- Points in sparse regions have **lower local reachability density**

---

## LOF Algorithm Steps

### Step 1: Find k-Nearest Neighbors
For each point, find its k nearest neighbors.

### Step 2: Calculate k-distance
For each point, determine the distance to its kth nearest neighbor.

### Step 3: Calculate Reachability Distance
For each point and its neighbors, compute the reachability distance.

### Step 4: Calculate Local Reachability Density
For each point, compute its local reachability density.

### Step 5: Calculate LOF Score
For each point, compute the LOF score by comparing its density to its neighbors' densities.

---

## Advantages of LOF

- ✅ **Local adaptation**: Adapts to varying densities in different regions
- ✅ **No global threshold**: Works well with clusters of different densities
- ✅ **Interpretable scores**: Easy to understand what LOF values mean
- ✅ **Robust**: Less sensitive to parameter selection than some methods

## Limitations of LOF

- ⚠️ **Computational cost**: O(n²) complexity for large datasets
- ⚠️ **Parameter selection**: Requires choosing k (number of neighbors)
- ⚠️ **Not suitable for streaming**: Requires all data to compute densities
- ⚠️ **Memory intensive**: Needs to store distances

---

## Data Assumptions

Anomaly detection techniques often make certain assumptions about the data:

### 1. Normal Data Distribution

**Assumption:** Some methods assume that normal data points follow a specific distribution, such as Gaussian distribution.

**Applies to:** Statistical methods (z-score, Gaussian mixture models)

**Implication:** If the assumption is violated, these methods may produce false positives/negatives.

**Mitigation:** Use non-parametric methods or validate distribution assumptions.

---

### 2. Feature Relevance

**Consideration:** The choice of features can significantly impact the performance of anomaly detection algorithms.

**Best Practices:**
- Select features that capture relevant aspects of normality
- Remove irrelevant or noisy features
- Apply feature engineering when necessary
- Scale features appropriately

**Impact:** Irrelevant features can mask true anomalies or create false alarms.

---

### 3. Labeled Data (for some methods)

**Requirement:** Some machine learning-based methods require labeled data to train the model.

**Semi-supervised approaches:** Use a small amount of labeled anomalies with mostly normal data

**Fully unsupervised:** Methods like Isolation Forest and LOF don't require labels

**Challenge:** Obtaining labeled anomaly data can be difficult and expensive.

---

## Comparison of Methods

| Method | Type | Complexity | Handles Varying Density | Interpretability | Best For |
|--------|------|------------|------------------------|------------------|----------|
| **One-Class SVM** | ML-based | O(n²-n³) | No | Medium | High-dimensional data with clear boundaries |
| **Isolation Forest** | ML-based | O(n log n) | Yes | High | Large datasets, fast detection |
| **LOF** | Density-based | O(n²) | Yes | High | Varying density clusters |
| **Statistical** | Statistical | O(n) | No | Very High | Simple, normally distributed data |

---

## Choosing the Right Method

### Use One-Class SVM when:
- You have high-dimensional data
- You need to model complex boundaries
- Computational resources are available
- You want strong theoretical guarantees

### Use Isolation Forest when:
- You have large datasets
- Speed is important
- Data has varying densities
- You want minimal parameter tuning

### Use LOF when:
- Data has clusters of varying densities
- Local context is important
- Interpretability is crucial
- Dataset size is moderate

### Use Statistical Methods when:
- Data follows known distributions
- Simplicity is preferred
- Fast computation is needed
- You need highly interpretable results

---

## Summary

Anomaly detection is a critical task in data analysis and machine learning, enabling the identification of unusual patterns and events:

**Types of Anomalies:**
- **Point Anomalies**: Individual unusual data points
- **Contextual Anomalies**: Context-dependent outliers
- **Collective Anomalies**: Groups of points forming unusual patterns

**Major Approaches:**
1. **Statistical Methods**: Assume specific distributions
2. **Clustering-Based**: Group data and identify isolated points
3. **Machine Learning**: Learn patterns from normal data

**Popular Algorithms:**
- **One-Class SVM**: Learns boundary around normal data
- **Isolation Forest**: Isolates anomalies with random partitioning
- **LOF**: Compares local densities

**Key Considerations:**
- Choose method based on data characteristics
- Consider computational resources
- Validate assumptions about data distribution
- Select relevant features carefully
- Tune parameters appropriately

**Applications:**
- Fraud detection
- Network intrusion detection
- System health monitoring
- Quality control
- Medical diagnosis

By leveraging various techniques and algorithms, anomaly detection systems can effectively identify outliers and provide valuable insights for decision-making and proactive intervention.
