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

Knowing algorithms commonly used for specific purposes can greatly boost on-demand problem solving ability by enabling crafting of a solution using some combination of past knowledge. There are "template problems" that other problems of the same sort can be reduced to that greatly help with this (e.g [Sliding Window Problems](https://medium.com/outco/how-to-solve-sliding-window-problems-28d67601a66)].

The amount of material required to do predictably exceptional in interviews can get staggering. This is where spaced repetition and anki cards come in.

[Spaced repetition](https://en.wikipedia.org/wiki/Spaced_repetition) is a well researched technique to improve long term recall of information through the use of meaningful spaces in review.  ...research

[Anki cards](https://apps.ankiweb.net/) are flash cards that are shown to you based on an algorithm that uses a rating of how difficult the problem was to calculate when you'll be shown the card next. ...research

This guide includes Anki cards for each major topic (theory + problems) that can be used as supplemental material.


# Algorithms

## Divide and Conquer

Divide and conquer involves dividing the problem into smaller parts, solving those individually, and then combining them back together for a meaningful result. Merge sort is a classic example of divide and conquer.

[This set of slides](https://www.ics.uci.edu/~goodrich/teach/cs260P/notes/DivideAndConquer.pdf) goes over merge sort and recurrence relations in divide and conquer problems.

Formally:

- Divide the problem instance I into smaller subproblems `I1...In`
- Solve `I1... In` recursively to get solutions `S1...Sn`
- Use `S1...Sn` to compute `S`. 

Let's explore a couple of problems.

### Merge-sort

A classic divide & conquer algorithm. Merge sort is a recursive algorithm that continually splits a list in half. If the list is empty or has one item, it is sorted by definition (the base case). If the list has more than one item, we split the list and recursively invoke a merge sort on both halves. Once the two halves are sorted, a merge is performed.

The basic routine of merge sort:

1. Divide the array into two halves
2. Recursively sort each half
3. Merge two halves
   
Merging is the process of taking two smaller sorted lists and combining them together into a single, sorted, new list. 

The merge subroutine:
```python
def merge(array, auxillary_array, lo, mid, hi):
  for k in range(lo, hi + 1):
    auxillary_array[k] = array[k]
  
  i, j = lo, mid + 1
  for k in range(lo, hi + 1):
    if i > mid:
      array[k] = auxillary_array[j]
      j += 1
    elif j > hi:
      array[k] = auxillary_array[i]
      i += 1
    elif array[j] < array[i]:
      array[k] = auxillary_array[j]
      j += 1
    else:
      array[k] = auxillary_array[i]
      i += 1
```

```python
def merge_sort(array):
  aux = [0] * len(array)
  sort(array, aux, 0, len(array) - 1)
  return array

def helper(array, auxillary_array, lo, mid, hi):
  if (hi <= lo) return
  mid = lo + (hi - lo) / 2
  sort(a, aux, lo, mid)
  helper(a, aux, mid+1, hi)
  helper(a, aux, lo, mid, hi)
```




### Non-dominated points

We say a point is non-dominated in a set if there is no other point `(x', y')` in the set such that `x <= x'` and `y <= y'` 

![Set of points](https://i.imgur.com/YWPJRpo.png)

The non-dominated point set here is `{A, H, I, G, D}`

We can apply divide and conquer here by:

1. Sort the points lexographically (first by x, then by y if x's are equal). 
2. Split the sorted set of points in half
3. Find non-dominated points in each half
4. Combine: Using the observation that any non-dominated point on the left with have to be as high or higher than all non-dominated points on the right, we can filter the left points by checking them against any right point and return `[filtered_left, right]`.  (we can use the first element in right as the highest because non-dominated point sets non-increasing).

The code looks like this:

```python
def non_dominated(points):
  points = sorted(points, key=lambda p: p.x)
  if len(points) == 1: return {points[0]}
  pivot = floor(len(points) / 2)
  left = non_dominated(points[:i])
  right = non_dominated(points[pivot + 1:])
  i = 0
  while i < len(left) and left[i].y > right[0]:
    i += 1
  return left[:i] + right
```

Complexity...

### Find the Kth largest element

The kth largest element in a set `A` is `sorted(A)[n - k]`. The trivial solution is sorting the set and returning the element with `O(nlogn)` time complexity. Sorting is overkill since we don't need all that computation. Using a heap would reduce the time complexity to `O(nlogk)`. This approach is faster, but still does more than what's required (finds the k largest).

A divide and conquer approach leveraging randomization leads to `quick select`.

1. We pick a random pivot
2. Partition the elements such that the pivot is < the right half and > the left half
3. Check if the element is kth largest 
4. If no, recurse on half containing the kth largest
5. If yes, return left[k - 1]
   
  
Code:

```python
def quick_select(arr, k):
  pivot = random.randint(0, len(arr) - 1)
  partition1, partition2 = [x for x in array if x > pivot], [x for x in array if x < pivot]
  if k - 1 == len(partition1):
    return pivot
  elif k - 1 < len(partition1):
    return quick_select(partition1, k)
  else:
    return quick_select(partition2, k - len(patition1) - 1)
```

Comlexity:

Unlike quicksort, there's only one recursive call in each invocation. On average, the pivot will be good about 50% of the time, where a good pivot is one that is within 25th and 75th percentile inclusive. So on average, every other call would have partitions that are of size at most 75% of the array's size. Therefore, on average the size of the subproblem would be reduced by 25% on every other call, which makes this function `O(n) average case`.

The worst case is the same as quick sort `O(n^2)`, however it can become highly unlikely to hit this worst case as n gets larger.


### Search for an element in a circular sorted array

Given a circular sorted array of distinct integers, search for an element k.

For example,

`search_circular_sorted([8, 9, 10, 2, 5, 6], 10) == 2`
`search_circular_sorted([9, 10, 2, 5, 6, 8], 5) == 3`

There is a non-divide and conquer solution to this that employs a modified binary search with the same time complexity `O(logn)`.

The divide and conquer solution is similar but solves the problem recursively.

```python
def dnc_search_cyclically_sorted(arr, k):
    def helper(left, right):
        mid = (left + right) // 2
        if right < left:
            return -1
        if arr[mid] == k:
            return mid
        if arr[mid] <= arr[right]:
            if arr[mid] < k <= arr[right]:
                return helper(mid + 1, right)
            else:
                return helper(left, mid - 1)
        else:
            if arr[left] <= k < arr[mid]:
                return helper(left, mid - 1)
            else:
                return helper(mid + 1, right)

    return helper(0, len(arr) - 1)
```

## Greedy

### Compute optimum assignment of tasks
Source: EPI 17.1, page 282

Consider a problem of assigning tasks to workers. Each worker must be assigned exactly two tasks. Each task takes a fixed amount of time. Tasks are independent. Any task can be assigned to any worker.

We want to minimize the amount of time it takes for all tasks to be completed. 

Example:
Tasks: [5, 2, 1, 6, 4, 4]
Assignment: [[5,2], [1,6], [4,4]] (total task time = 8 hours)

A simple greedy heuristic would be to pair the longest task with the shortest one in order to minimize total task time for that worker. This intuition ends up leading to an optimal solution.

Code:

```python
def optimum_task_assignment(durations):
  durations.sort()
  return [
    [task_durations[i], task_durations[~i]],
    for i in range(len(task_durations) // 2)
  ]
```


## Dynamic Programming

# Data Structures

Data structures can be split into two categories:

1. **Contiguously-allocated** data structures are a single block of memory used for arrays, matrices, heaps and hash tables.

2. **Linked** data structures are composed of distinct chunks of memory 

## Arrays

Arrays are contiguous blocks of memory representing a list of elements. 

A few tips from Elements of Programming Interviews that are notable:

> When dealing with integers encoded by an array consider processing the digits from the back of the array so the least-significant digit is the first entry.
> When operating on 2D arrays, use parallel logic for rows and columns

We'll explore problems concerning modifying arrays in-place such that some property is satisfied, traversing and multi-dimensional datasets.

### Complexities

| Routine | Dynamic Array | Sorted Dynamic Array |
| :------ | :-----------: | -------------------: |
| select  |     O(1)      |                 O(1) |
| search  |     O(n)      |              O(logn) |
| insert  |     O(1)      |                 O(n) |
| remove  |     O(1)      |                 O(n) |

### 3-way partition
Also known as the dutch national flag problem from EPI page 44.

Given an array A and an index i, rearrange the elements such that all elements less than A[i] appear first, followed by elements equal to the pivot, followed by elements greater than the pivot.

The trivial solution is to form the 3 lists and combine them, but let's try to avoid O(n) space complexity by rearring the list in place.

We can do this by maintaing four subarrays: bottom (elements < pivot), middle (elements == pivot), unclassified, and top (elements > pivot). Initially all elements are unclassified. We iterate through elements in unclassified, and move elements into one of bottom, middle, and top groups according to the relative order between the incoming unclassified element and the pivot. (epi pg. 44)

Code:

```python
def three_way_partition(arr, i):
    pivot_value = arr[i]
    smaller, equal, larger = 0, 0, len(arr)
    while equal < larger:
        if arr[equal] < pivot:
            arr[smaller], arr[equal] = arr[equal], arr[smaller]
            smaller, equal = smaller + 1, equal + 1
        elif arr[equal] == pivot:
            equal += 1
        else: 
            larger -= 1
            arr[equal], arr[larger] = arr[larger], arr[equal]
```


### Compute max profit from one buy and sell
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

### Maximum Sum Circular Subarray
 
Given a circular array of integers, find subarray in it which has the largest sum.

For example,

Input:  {2, 1, -5, 4, -3, 1, -3, 4, -1}
Output: Subarray with the largest sum is {4, -1, 2, 1} with sum 6.
 
Input:  {-3, 1, -3, 4, -1, 2, 1, -5, 4}
Output: Subarray with the largest sum is {4, -1, 2, 1} with sum 6.

Let's first try to find the max subarray of a regular array. The brute-force algorithm would be to have a nested loop and determine the max at each index, return the total max.

The brute force approach leads to the insight that we're trying to find the max index somehow. We can alter this problem slightly and look for the max ending at an index i. This value can be calculated using the previous max:

`max_at_i = max(max_at_i-1, arr[i])`

This leads to a linear solution:

```python
def max_subarray(arr):
    max_sum = 0
    max_at_i = arr[0]
    for i in range(1, len(arr)):
        max_at_i = max(max_at_i + arr[i], arr[i])
        max_sum = max(max_sum, max_at_i)
    
    return max_sum
```

To find the max of a circular subarray, we can just concatenate the array to itself and find use the max_subarray routine.

```python
def max_subarray_circular(arr):
    return max_subarray(arr + arr)
```

### Container with maximum water

Let us suppose we have a two dimensional plane where the the length of lines determine the height of a container. We need to determine the maximum capacity of water this kind of an arrangement can hold. The heights are represented by an array.

Input: [1, 8, 6, 2, 5, 4, 8, 3, 7]

Output: 49

![Max water](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/07/17/question_11.jpg)

The brute force solution is to have a nested for loop find the max water starting at an index i. 

```python
def max_area(height: List[int]) {
    max_area = 0
 
    for i in range(len(height)):
        for j in range(i + 1, len(height)):
          max_area = max(max_area, Math.min(height[i], height[j]) * (j - i))

    return max_area
}
```

The nested loop can be avoided with a heuristic for moving either left or right when considering start and end indices. 

This works because if we move away from a larger tower then we'll definitely decrease our max area since the width decreases and we are limited in area by the height of the shorter towe. Hence, move away from the shorter one.


```python
def max_area(height: List[int]) -> int:
    left, right = 0, len(height) - 1
    max_area = 0
    
    while (left < right):
        area = min(height[left], height[right]) * (right - left)
        max_area = max(max_area,  area)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
        
    return max_area
```

## Strings

### Minimum window problem

## Tries

### Longest common prefix

## Stacks & Queues

## Binary Trees

## Heaps

## Hash Tables


