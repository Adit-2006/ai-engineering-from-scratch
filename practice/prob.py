import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

def uniform_pdf(x,a,b):
    if x <= a <=b:
        return 1.0/(b-a)
    return 0.0
def normal_pdf(sigma, mu, x):
    return 1.0 / (math.sqrt(2 * math.pi * sigma ** 2) * math.exp(((x - mu) ** 2) / (2 * sigma ** 2)))

def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))


def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

def softmax(logits):
    max_logit = max(logits)
    shifted = [x - max_logit for x in logits]
    exp = [math.exp(ele) for ele in shifted]
    total = sum(exp)
    return [e / total for e in exp]

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
sigma = math.sqrt(var)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")


import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)