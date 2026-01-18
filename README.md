# Linear Regression from Scratch using Batch Gradient Descent

## Overview

This project focuses on understanding the **fundamentals of Linear Regression** by implementing **Batch Gradient Descent from scratch**, without relying on machine learning libraries for optimization.

Instead of prioritizing prediction accuracy, the goal of this project is to:

* understand how gradient descent works internally,
* observe convergence and divergence behavior,
* and compare a manual implementation with a standard library solution.

---

## Problem Statement

The objective is to model **student performance** based on the following factors:

* Study hours per day
* Sleep hours per day
* Break time in minutes

The target variable is:

* Overall performance score

This problem was chosen because it is simple, interpretable, and suitable for studying optimization behavior in linear regression.

---

## Dataset

* The dataset used in this project is **synthetically generated**.
* Values were created using realistic assumptions about student behavior and academic performance.
* Noise was intentionally added to make the problem closer to real-world data.

**Note:**
The dataset is not meant to be a benchmark, but a controlled setup to study gradient descent behavior.

---

## Approach

### 1. Data Visualization

Each input feature is visualized independently against the target variable to:

* verify assumptions about linear relationships,
* ensure the synthetic data behaves realistically.

### 2. Reference Model (scikit-learn)

A linear regression model using `scikit-learn` is trained to:

* obtain a stable reference solution,
* compare coefficients and intercept values.

### 3. Batch Gradient Descent (From Scratch)

A custom class is implemented to train linear regression using:

* Mean Squared Error (MSE) loss
* Batch Gradient Descent updates
* Manual parameter initialization and updates

This implementation highlights:

* sensitivity to learning rate,
* the effect of unscaled features,
* divergence and convergence behavior.

---

## Key Observations

* Large learning rates caused divergence and exploding coefficients.
* Very small learning rates resulted in slow but stable convergence.
* Without feature scaling, gradient descent converges much more slowly.
* The manually trained model learns the **correct direction of relationships**, even if coefficients do not exactly match the closed-form solution.

---

## Results Summary

| Model                           | Stability               | Coefficient Behavior              |
| ------------------------------- | ----------------------- | --------------------------------- |
| scikit-learn LinearRegression   | Stable                  | Optimal (closed-form solution)    |
| Batch Gradient Descent (custom) | Learning-rate dependent | Correct signs, slower convergence |

The focus of this project is **learning dynamics**, not metric optimization.

---

## Limitations

* Features are not scaled.
* Only Batch Gradient Descent is implemented.
* Dataset size is intentionally small.
* Model does not capture non-linear relationships (e.g. diminishing returns).

These limitations are intentional at this stage to emphasize fundamentals.

---

## Future Improvements

* Feature scaling and normalization
* Mini-batch and Stochastic Gradient Descent
* Learning rate scheduling
* Polynomial features for non-linear behavior
* Visualization of loss curves

---

## What This Project Demonstrates

* Understanding of linear regression mathematics
* Hands-on implementation of gradient descent
* Debugging of real numerical and shape issues
* Awareness of optimization limitations

---

## Author

Soumita Das
First-year CSE (AI) student
Interested in Machine Learning fundamentals and applied AI
