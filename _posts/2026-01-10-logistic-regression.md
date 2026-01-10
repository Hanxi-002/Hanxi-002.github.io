---
layout: post
title: "Logistic Regression"
subtitle: "A Mathematical Introduction to Binary Classification"
tags: [machine-learning, statistics]
mathjax: true
---

<style>
:root {
  --ocre: #34B1C9;
  --bg-black5: #F2F2F2;
  --bg-ocre10: #EBF7FA;
  --ocre60: #85D0DF;
}

/* Headings */
h1, h2, h3, h4 {
  font-family: "Avant Garde", "TeX Gyre Adventor", "Helvetica", Arial, sans-serif;
}
h1, h2 {
  color: var(--ocre);
}

/* Box styles inspired by mdframed settings in structure.tex */
.box {
  background: var(--bg-black5);
  border-left: 4px solid var(--ocre);
  padding: 0.9rem 1rem;
  margin: 1rem 0;
}
.box.exercise {
  background: var(--bg-ocre10);
}
.box.corollary {
  background: var(--bg-black5);
  border-left-color: #808080;
}
.box-title {
  font-family: "Avant Garde", "TeX Gyre Adventor", "Helvetica", Arial, sans-serif;
  font-weight: 700;
  color: var(--ocre);
  margin-bottom: 0.4rem;
}
</style>

# Logistic Regression

## Motivation

In linear regression, we model the relationship between an input \(x\) and a target \(y\) with a linear function such as

$$
y = ax + b.
$$

For binary classification, we want model outputs that can be interpreted as probabilities.
Logistic regression does this by mapping a real valued score to the interval \([0,1]\) using the logistic sigmoid.

## Sigmoid function

The logistic sigmoid maps a scalar \(z \in \mathbb{R}\) to a value strictly between 0 and 1:

$$
\sigma(z) = \frac{1}{1 + \exp(-z)}.
$$

If we wrote \(y = \sigma(x)\) directly, this would only make sense when the input is a scalar.
In practice, each sample has many features, so we first compress the feature vector into a single scalar score and then apply the sigmoid.

<div class="box">
  <div class="box-title">Goal of logistic regression</div>

Before we build the full multivariate model, it helps to look at what the sigmoid gives us.

- The output never goes below 0 and never goes above 1, which matches the range of a probability.
- Logistic regression is used when we want to learn probabilities for classification.
- The goal is to model the probability that a sample belongs to a particular class:

$$
P(y_i = y \mid x_i), \quad \text{for each class label } y.
$$

In the binary case, we often write \(p_i = P(y_i = 1 \mid x_i)\) and \(1 - p_i = P(y_i = 0 \mid x_i)\).
For more than two classes, the analogous mapping from scores to probabilities is typically done with softmax.
</div>

## From features to one score

Let each sample have \(d\) features.
For sample \(i\),

$$
x_i \in \mathbb{R}^d, \quad w \in \mathbb{R}^d, \quad b \in \mathbb{R}.
$$

We form a scalar score

$$
z_i = w^\top x_i + b \in \mathbb{R},
$$

and map it to a probability

$$
p_i = P(y_i = 1 \mid x_i) = \sigma(z_i).
$$

In matrix form, for \(n\) samples,

$$
X \in \mathbb{R}^{n \times d},\;
w \in \mathbb{R}^{d \times 1},\;
z \in \mathbb{R}^{n \times 1},\;
y \in \mathbb{R}^{n \times 1}.
$$

$$
z = Xw + b\mathbf{1},
\qquad
p = \sigma(z),
$$

where \(\mathbf{1} \in \mathbb{R}^{n \times 1}\) is an all ones vector and \(\sigma\) is applied elementwise.

## Likelihood and loss

Assume \(y_i \in \{0,1}\) and \(p_i = P(y_i = 1 \mid x_i)\).
Then

$$
P(y_i = 1 \mid x_i) = p_i,
\qquad
P(y_i = 0 \mid x_i) = 1 - p_i.
$$

This can be written as a Bernoulli likelihood:

$$
P(y_i \mid x_i) = p_i^{y_i} (1-p_i)^{1-y_i}.
$$

Maximizing likelihood is equivalent to minimizing the negative log likelihood.
The per sample loss is

$$
\ell_i
= -\log P(y_i \mid x_i)
= -\Bigl[y_i \log(p_i) + (1-y_i)\log(1-p_i)\Bigr].
$$

The average loss over \(n\) samples is

$$
L = \frac{1}{n}\sum_{i=1}^n \ell_i.
$$

This is also called binary cross entropy.

## Gradients

We want the gradients with respect to \(w\) and \(b\) so we can apply gradient based optimization.
The dependencies are

$$
w,b \to z_i = w^\top x_i + b \to p_i = \sigma(z_i) \to L.
$$

We use the chain rule:

$$
\frac{\partial L}{\partial w}
= \sum_{i=1}^n \frac{\partial L}{\partial p_i}\frac{\partial p_i}{\partial z_i}\frac{\partial z_i}{\partial w},
\qquad
\frac{\partial L}{\partial b}
= \sum_{i=1}^n \frac{\partial L}{\partial p_i}\frac{\partial p_i}{\partial z_i}\frac{\partial z_i}{\partial b}.
$$

### Derivative of the sigmoid

For \(p_i = \sigma(z_i)\),

$$
\frac{\partial p_i}{\partial z_i} = p_i(1-p_i).
$$

### Derivative of the loss with respect to probability

From

$$
\ell_i = -\Bigl[y_i \log(p_i) + (1-y_i)\log(1-p_i)\Bigr],
$$

$$
\frac{\partial \ell_i}{\partial p_i}
= \frac{1-y_i}{1-p_i} - \frac{y_i}{p_i}.
$$

Since \(L = \frac{1}{n}\sum_i \ell_i\), we have

$$
\frac{\partial L}{\partial p_i} = \frac{1}{n}\left(\frac{1-y_i}{1-p_i} - \frac{y_i}{p_i}\right).
$$

### Derivative with respect to the score

Combining the previous results yields the standard simplification:

$$
\frac{\partial L}{\partial z_i} = \frac{1}{n}(p_i - y_i).
$$

### Gradients for parameters

Since \(z_i = w^\top x_i + b\),

$$
\frac{\partial z_i}{\partial w} = x_i,
\qquad
\frac{\partial z_i}{\partial b} = 1.
$$

Therefore,

$$
\frac{\partial L}{\partial w}
= \frac{1}{n}\sum_{i=1}^n (p_i - y_i)x_i
= \frac{1}{n}X^\top(p-y),
$$

and

$$
\frac{\partial L}{\partial b}
= \frac{1}{n}\sum_{i=1}^n (p_i - y_i).
$$

## Gradient descent updates

With learning rate \(\eta > 0\), gradient descent updates are

$$
w \leftarrow w - \eta \cdot \frac{1}{n}X^\top(p-y),
\qquad
b \leftarrow b - \eta \cdot \frac{1}{n}\sum_{i=1}^n (p_i - y_i).
$$

## Softmax and logits

Sigmoid maps a scalar score to a single probability.
For multiclass classification with \(k\) classes, softmax maps a vector of scores to a probability distribution that sums to 1.
Given scores \(s \in \mathbb{R}^k\),

$$
\mathrm{softmax}(s)_j = \frac{\exp(s_j)}{\sum_{t=1}^k \exp(s_t)}.
$$

The term logits refers to the unnormalized scores before applying sigmoid or softmax.

## Bias as an extra feature

It is common to fold the bias into the weight vector by augmenting the data with a constant feature.
Define

$$
\tilde{x}_i = \begin{bmatrix} 1 \\ x_i \end{bmatrix} \in \mathbb{R}^{d+1},
\qquad
\tilde{w} = \begin{bmatrix} b \\ w \end{bmatrix} \in \mathbb{R}^{d+1}.
$$

Then

$$
z_i = \tilde{w}^\top \tilde{x}_i,
$$

so the model can be written without a separate bias term.

## Practical notes

- This note describes the binary case where \(y \in \{0,1}\).
- For numerical stability, implementations often compute the loss using stable transformations rather than directly evaluating \(\log(p)\) and \(\log(1-p)\) when \(p\) is extremely close to 0 or 1.
