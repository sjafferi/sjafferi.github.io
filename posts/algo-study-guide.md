# What's in this guide

This is meant to be a comprehensive guide for mastering technical interviews on algorithms and data structures. It includes a ground up understanding of the theory (relying on [UWaterloo's CS341](https://www.student.cs.uwaterloo.ca/~cs341/) and [Skiena's Algorithm Design Manual](http://www.algorist.com/)) as well as problems aggregated from books (relying heavily on [Elements of Programming Interviews](https://www.amazon.com/gp/product/1537713949?ie=UTF8&tag=sjafferi-20&camp=1789&linkCode=xm2&creativeASIN=1537713949)) and the web. 

This guide assumes you have some background in algorithms and hence won't reiterate common definitions.

By the end of each section, you should have the tools necessary to come up with on the spot solutions to related problems.  

## Study Plan

The plan of attack is as follows:

- Understand the underlying theory behind each data structure
  - Implementation, applications, complexities, traversals
- Study problems representative of the data structure
  - Core problems that can be used to derive solutions to many other problems
- Iterate until perfect
  - Use spaced repetition (linked Anki cards) to commit learnings to long term memory

## Spaced Repetiton with Anki Cards

Knowing algorithms commonly used for specific purposes can greatly boost on-demand problem solving ability by enabling crafting of a solution using some combination of past knowledge. 

It follows that having a vast and diverse knowledge of algorithms correlates with a higher success rate in interviews. The amount of material required to do predictably exceptional in interviews can get staggering.

[Spaced repetition](https://en.wikipedia.org/wiki/Spaced_repetition) is a well researched technique to improve long term recall of information through the use of meaningful spaces in review.  ...research

[Anki cards](https://apps.ankiweb.net/) are flash cards that are shown to you based on an algorithm that uses a rating of how difficult the problem was to calculate when you'll be shown the card next. ...research

This guide includes Anki cards for each major topic (theory + problems) that can be used as supplemental material.


# Algorithms

## Divide and Conquer

Divide and conquer involves dividing the problem into smaller parts, solving those individually, and then combining them back together for a meaningful result.

Formally:

- Divide the problem instance I into smaller subproblems `I1...In`
- Solve `I1... In` recursively to get solutions `S1...Sn`
- Use `S1...Sn` to compute `S`. 


## Greedy
## Dynamic Programming

# Data Structures

Data structures can be split into two categories:

1. **Contiguously-allocated** data structures are a single block of memory used for arrays, matrices, heaps and hash tables.

2. **Linked** data structures are composed of distinct chunks of memory 

## Arrays

Arrays are contiguous blocks of memory representing a list of elements. 

Steven Skiena quotes a great analogy in ADM (Algorithm Design Manual)

>

### Complexities

| Routine | Dynamic Array | Sorted Dynamic Array |
| :------ | :-----------: | -------------------: |
| select  |     O(1)      |                 O(1) |
| search  |     O(n)      |              O(logn) |  |
| insert  |     O(1)      |                 O(n) |  |
| remove  |     O(1)      |                 O(n) |  |

### Common Algorithms

**Search (sorted)**

```
binary search
```

**Search (unsorted)**

```
quick select
```

### Problems


#### Compute max profit from one buy and sell
**Source:** Elements of Programming Interviews 5.6

**Problem Statement**
Given an input of prices, compute the maximum profit you can get by buying and selling at different price points.

**Input / Output**
[6, 5, 3, 10, 0] => 7

**Brute Force Solution**
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

**Divide and Conquer**

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

**Greedy**

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

## Strings

## Stacks & Queues

## Binary Trees

## Heaps

## Hash Tables


