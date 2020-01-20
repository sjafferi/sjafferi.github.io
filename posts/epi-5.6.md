## Problem Statement
Given an input of prices, compute the maximum profit you can get by buying and selling at different price points.

## Input / Output
[6, 5, 3, 10, 0] => 7

## Brute Force Solution
```python
def max_profit_brute_force(prices):
    max_profit = 0
    for i in range(0, len(prices)):
        for j in range(i + 1, len(prices)):
            if prices[j] - prices[i] > max_profit:
                max_profit = prices[j] - prices[i]
    return max_profit
```

Simple enough...but can we do better than `O(n^2)`? We can go through our list of potential algorithm choices and two stand out. 

## Divide and Conquer

Divide-and-Conquer could work by splitting the prices in half and determining the best result for each half and combining those results. There are a few edge cases to consider here, such as, what if the optimal buy is in a different half than the optimum sell? The following solution implements this approach...

```python
def split(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]

def max_profit_divide_and_conquer(prices):
    if len(prices) == 0 or len(prices) == 1: 
        return 0
    A,B = split(prices)
    return max(
        max_profit_divide_and_conquer(A),
        max_profit_divide_and_conquer(B),
        max(B) - min(A),
        0
     )
```
Worst-case time: `O(nlogn)`

## Greedy 

The Greedy approach actually ends up providing the best time complexity for us. The insight that leads to the key part of this approach can be derived from the Divide and Conquer algorithm. We can see that the maximum profit can be made by selling on any particular day and buying on a specific minimum day that occurred previously. Hence, each element must consider the minimum before it to see if it's optimal, and this can be kept track of as we iterate.

```python
def max_profit_greedy(prices):
    min_price_so_far = float('inf')
    max_profit = 0.0
    for price in prices:
        max_profit_sell_today = price - min_price_so_far
        max_profit = max(max_profit, max_profit_sell_today)
        min_price_so_far = min(min_price_so_far, price)
    return max_profit
```

