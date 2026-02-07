---
layout: post
title: "Logistic Regression"
subtitle: "A Mathematical Introduction to Binary Classification"
tags: [ML, statistics, classification]
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

/* Collapsible details box styling */
details {
  background: var(--bg-ocre10);
  border-left: 4px solid var(--ocre);
  padding: 0.9rem 1rem 0.9rem 1.5rem;
  margin: 1rem 0;
  border-radius: 4px;
}
details summary {
  font-family: "Avant Garde", "TeX Gyre Adventor", "Helvetica", Arial, sans-serif;
  font-weight: 700;
  color: var(--ocre);
  cursor: pointer;
  margin-bottom: 0.5rem;
  margin-left: -0.3rem;
  list-style-position: inside;
}
details[open] summary {
  margin-bottom: 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--ocre60);
}
</style>

# Defining the question we want to answer.

We are all familiar with a linear function. In linear regression, we model the relationship between input $x$ and target $y$ with a linear function such as $y = ax + b$. To put it more bluntly, when we utilize linear regression as the model of choice, we are making the assumption that there exist a linear relationship between the data and the outcome we would like to predict. Even one has little knowledge about linear regression, just by observing the visualization of a lienar funciton below, we can gain the intuition of what type of quesitons linear regression can answer. For example, we can make conclusions that the outcome of this linearly predicted model has to be continuous and may take on any value on the equation line. 

<div style="text-align: center; margin: 1.5rem 0;">
  <img src="/assets/img/posts/logistic_regression/linear_function.png" alt="Linear function visualization" style="max-width: 80%;">
</div>

So, what do we do if the relaionship we want to model is not linear? More specifically, what if we want to build a model that we know the outcome we want to predict only takes on 2 values, 0 and 1? 

# Building Intuitition for Binary Classification
If the outcome we want to predict only takes on 2 values, then these outcomes inherently exist in 2 camps. We can then also frame this question as what is called a binary classification question. 

In linear regression, we saw that if we want to model a linear relationship between data and outcome, we used a linear function. So the most intuitive solution is to perhaps search for a function that somehow let us predict 2 values of outcome of interest given the data?

Turns out, there is a function allows us to model this, may seem somewhat strange, relationship. And that is a logistic function. 
<!-- For binary classification, we want model outputs that can be interpreted as probabilities.
Logistic regression does this by mapping a real valued score to the interval  $$[0,1]$$  using the logistic sigmoid. -->

## Logistic Function
### Visualization
<div style="text-align: center; margin: 1.5rem 0;">
  <img src="/assets/img/posts/logistic_regression/sigmoid_function.png" alt="Sigmoid function visualization" style="max-width: 80%;">
</div>

Let's not worry about what a logistic function is, but just looking at what it looks like. By visualizing the logistic function, we see that the the y is mostly concentrated in 2 places, close to 0 and close to 1. Not only this function solved our above problem, but also provides a nice addition that the outcome is always bounded by 0 and 1. And this should sound really familiar as it is also one of the important definition of what a "probability" is. 

So logistic function allows us to do two things: (i) be able to model the relationship of the input to the outcome that has 2 classes and (ii) be able to model the probability of the outcome as well. So in addition to asking, does the input belong to class 1 or class 2? We can ask what is the probability of the input belong to class 1. 

### Understanding the Logistic Function
Now we are getting closer to the case we want to crack, let's look into more of the logistic function. 

The logistic function maps a scalar  $z \in \mathbb{R}$ to a value strictly between 0 and 1:

$$
\sigma(z) = \frac{1}{1 + \exp(-z)}.
$$

Yay, now when we input a sample $x$, we can use the logistic function to get a y right? Not quite, you may notice 2 small details that doesn't quite make sense here. 1 is that we specifically said that "the logistic function maps a **scalar** to a ...";  and 2 is that you notice that we wrote the input to the above logistic function as z instead of x. 

The answer to these 2 questions are actually very much related. Let's start with problem 1: if logistic funciton only takes in a scalar, then what do we do when we have multi-variate input data?

Well, how about we come up with a way to describe the vector of features only using 1 scalar, and define that descriptive sclar z. Here, both of the prbolems have been solved!


### Big Picture of The Model

At this point, we started with the problem we want to solve: find a way to predict the outcome $y$ given some data $x$ knowing that it is a binary classification. We first identified that the logistic function can help us model this relationship because of its unique S shape. We then realized, by visualizin the fuction, that not only the shape of the function means that most of the y value will sit in the extreme 2 sides of the funciton, but the y is also bounded between 0 and 1, which is one of the exact definitions of probabilities. So not only can we know if one datapoint belong to class 1 or class 2, we can also learn that if this data is 90% of class1 or 70% chance of beloning to class 1.  

After that, by investigating the mathematical form of the logistic function, we realized that if we want a scalar output form the funciton, which we do because if datapoint should only have one number represernting the probability of it belonging to a class, we can only input one scalar as the input. So, when we deal with multi-variate datasets, we have to find some work around to summarize the informaion of many features into one number. 

Before we build the full multivariate model, it helps to look at what the sigmoid gives us:

- The output never goes below 0 and never goes above 1, which matches the range of a probability.
- Logistic regression is used when we want to learn probabilities for classification.
- It takes a scalar as an input and outputs a scalar that we interpret as probability.
- The goal is to model the probability that a sample belongs to a particular class:

$$
P(y_i = y \mid x_i), \quad \text{for each class label } y.
$$

In the binary case, we often write  $$p_i = P(y_i = 1 \mid x_i)$$  and  $$1 - p_i = P(y_i = 0 \mid x_i)$$ .


# Building Logistic Regression Mathmatically
## Defining the score, $z$.
We know that the logistic function is perfect to model the relationship we are interested in, given input data $x$, predict the probability of $x$ beining in class0 and class1. In linear regression, we are able to plug in the data directly into the linear equation. But we can't do that in logistic regression when we have multi-variate data. We have to first figure out how to calcualte that one scalar, also called the score, for each datapoint $x_i$. 

Turns out, the score is fairly simple to calcualte, we can use the linear equation here.  

Let each $x_i$ have $d$ features,
$$
z_i = w^\top x_i + b
$$

$$
x_i \in \mathbb{R}^d, \quad w \in \mathbb{R}^d, \quad b \in \mathbb{R}.
$$

Logistic regression defines this score to be calcualted using a linear function and the model defines the score, $z$, to be interpreted as the log-odds ratio. Log-odds is defined as the following: for any probability p of class 1, we get
$$
log-odds = log (\frac{p(x)}{1-p(x)}).
$$
If $p = 0.5$, then log-odds $= 0$.<br>
If $p > 0.5$, then log-odds $> 0$.<br>
If $p < 0.5$, then log-odds $< 0$.<br>

In other words, we can view the log-odds as the "confidence scale" of if the data $x$ belong to class 1. Now, in logistic regression, we know we cauclate the score using the linear funciton, so we can combine the expressions and get:
$$
\text{log-odds} = log (\frac{p(x)}{1-p(x)}) = w^\top x_i + b
$$

Intuitively, we are not combining the features in a "magical" way, we are asking how can we find the best parameters $w$ to build the most logistic regression model that understand how much each of the $d$ features contribute to the "confidence scale". We then simply use the logistic function to re-scale the log-odds score in terms of probability:

$$
p_i = P(y_i = 1 \mid x_i) = \sigma(z_i).
$$

(Bridge section here for non-linear binary classification?)
## Establish Logstic Regression Model
We can now establish the full model in a multivariate form. For  $n$  samples,
$$
p = \sigma(z)
$$

$$
z = Xw + b
$$

$$
X \in \mathbb{R}^{n \times d},\;
w \in \mathbb{R}^{d \times 1},\;
z \in \mathbb{R}^{n \times 1},\;
y \in \mathbb{R}^{n \times 1},\;
p \in \mathbb{R}^{n \times 1}.
$$

<!-- ###################################### Wrapping Bias in Features ################################################## -->
<details markdown="1">
<summary>Bridge: Warpping bias in input matrix</summary>
(Remember, the "Bridge" section is not meant as a full deep dive or tutorial. It is meant for pattern recognition to organize and connect all the loose terminology and concepts that are closely connected and/or similar.)

Sometimes, you will see a simple mathmeticle trick to make calculations easier and that is wrapping the bias inside the input matrix. 
In above section, 
where  $b_1 \in \mathbb{R}^{n \times 1}$  is an all ones vector and  $\sigma$  is applied elementwise.

</details>
<!-- ###################################### Softmax Bridge ################################################## -->


## Likelihood and loss

Assume  $$y_i \in \\{0,1\\}$$  and  $$p_i = P(y_i = 1 \mid x_i)$$ .
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

The average loss over  $$n$$  samples is

$$
L = \frac{1}{n}\sum_{i=1}^n \ell_i.
$$

This is also called binary cross entropy.

## Gradients

We want the gradients with respect to  $$w$$  and  $$b$$  so we can apply gradient based optimization.
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

For  $$p_i = \sigma(z_i)$$ ,

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

Since  $$L = \frac{1}{n}\sum_i \ell_i$$ , we have

$$
\frac{\partial L}{\partial p_i} = \frac{1}{n}\left(\frac{1-y_i}{1-p_i} - \frac{y_i}{p_i}\right).
$$

### Derivative with respect to the score

Combining the previous results yields the standard simplification:

$$
\frac{\partial L}{\partial z_i} = \frac{1}{n}(p_i - y_i).
$$

### Gradients for parameters

Since  $$z_i = w^\top x_i + b$$ ,

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

With learning rate  $$\eta > 0$$ , gradient descent updates are

$$
w \leftarrow w - \eta \cdot \frac{1}{n}X^\top(p-y),
\qquad
b \leftarrow b - \eta \cdot \frac{1}{n}\sum_{i=1}^n (p_i - y_i).
$$

## Softmax and logits

Sigmoid maps a scalar score to a single probability.
For multiclass classification with  $$k$$  classes, softmax maps a vector of scores to a probability distribution that sums to 1.
Given scores  $$s \in \mathbb{R}^k$$ ,

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

- This note describes the binary case where  $$y \in \\{0,1\\}$$ .
- For numerical stability, implementations often compute the loss using stable transformations rather than directly evaluating  $$\log(p)$$  and  $$\log(1-p)$$  when  $$p$$  is extremely close to 0 or 1.


<!-- ###################################### Softmax Bridge ################################################## -->
<details markdown="1">
<summary>Bridge: Softmax as the multiclass sigmoid</summary>
(Remember, the "Bridge" section is not meant as a full deep dive or tutorial. It is meant for pattern recognition to organize and connect all the loose terminology and concepts that are closely connected and/or similar.)

If you have seen softmax functions before, you may have started to recognize that softmax looks familiar with the logistic function and performs a somewhat related task. Indeed, while the sigmoid function is perfect for binary classification (two classes), real-world problems often involve multiple classes. The softmax function extends this concept to handle $K$ classes.

### Definition
For a vector of scores that describe one sample $$z = [z_1, z_2, \ldots, z_K]$$ , the softmax function converts them into a probability distribution:

$$
\text{softmax}(z_j) = \frac{\exp(z_j)}{\sum_{k=1}^K \exp(z_k)}
$$

In other words, instead of using 1 scalar to "summarize and describe" a multivariate datapoint $x_i$ in logistic regression. We summarize the datapoint by using $K$ number of scores and then learn the probability of $x_i$ belong to each of the $K$ classes. 

### Key Properties
- **Outputs sum to 1**: $$\sum_{j=1}^K \text{softmax}(z_j) = 1$$
- **Each output is between 0 and 1**: Perfect for representing probabilities
- **Differentiable**: Enables gradient-based optimization

### Connection to the Logistic Function

When $K = 2$ (binary case), softmax reduces to the sigmoid function. The two functions are fundamentally related, with sigmoid being a special case of softmax.
We first review the key realization for binary clasification, we only need to know $p(y = 0 | x)$ and we get $p(y = 1 | x)$ for free since the sum of both probabilities has to be 1.

$$
\begin{align*}
\text{softmax}(z_0) &= \frac{\exp(z_0)}{\exp(z_0) + \exp(z_1)} \\[1em]
&= \frac{\frac{\exp(z_0)}{\exp(z_0)}}{\frac{\exp(z_0)}{\exp(z_0)} + \frac{\exp(z_1)}{\exp(z_0)}} \\[1em]
&= \frac{1}{1 + \exp(z_1 - z_0)}
\end{align*}
$$

Intuitively, we should learn 2 scores for logistic regression given the introduction we gave about softmax above. However, remember that we get $p(y = 1 | x)$ for free, so we only need one scalar score for $x_i$ and learn the probability of $p(y = 0 | x)$. Hence, we can se $z_1 = 0$ and get:
$$\frac{1}{1+ exp(-z_0)}$$

</details>
<!-- ###################################### Softmax Bridge ################################################## -->