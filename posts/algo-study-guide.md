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

Greedy algorithms are often used for optimization problems, which are problems where the goal is to find a solution that both satisfies the problem constraints, and maximizes/minimizes an objective/profit/cost function. 

We essentially greedily pick the locally optimal solution at every step in order to determine a globally optimal solution (or approximate).

Greedy algorithms do no backtracking or lookahead - they only consider the elements of the choice set at each step. That means they are often faster than algorithms that do backtracking, but the solution is possibly not as good. When implementing greedy algorithms, it is often efficient to do a preprocessing step to compute the local evaluation criterion, like sorting the elements of an array.

For some greedy algorithms, it is possible to always obtain an optimal solution. However, these proofs are often rather difficult.

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

### The Interval Covering Problem
Source: EPI 17.2, page 285

Consider a foreman responsible for visiting a factory to check on the running tasks. In each visit, he can check on all the tasks taking place at the time of the visit. We want to minimize the number of visits. 

Given a set of intervals containing start and end times for tasks, return the minimum number of visits required to cover each interval.

Input: [[1,2], [2,3], [3,4], [2,3], [3,4], [4,5]]
Output: 2

A greedy approach could be to focous on extreme cases. Consider the interval that ends first, i.e. the interval whose right endpoint is minimum. To cover it, we must pick a number that appears in it. Hence we can eliminate any other intervals covered by it, and repeat for the next right endpoint.

```python
import operator

def find_min_visits(intervals):
    intervals.sort(key=lambda x: x[1])
    last_visit_time = intervals[0][1]
    num_visits = 1
    for interval in intervals:
        if interval[0] > last_visit_time:
            last_visit_time = interval[1]
            num_visits += 1
    
    return num_visits
```

Complexity: O(nlogn) - Dominated by sort

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

### Kruskal's Minimum Spanning Tree 

Kruskal's algorithm is a minimum-spanning-tree algorithm which finds an edge of the least possible weight that connects any two trees in the forest. It is a greedy algorithm in graph theory as it finds a minimum spanning tree for a connected weighted graph adding increasing cost edges at each step.

Steps:

1. Sort all the edges in non-decreasing order of their weight.
2. Pick the smallest edge. Check if it forms a cycle with the spanning tree formed so far using Union Find data-structure. If cycle is not formed, include this edge else, discard it.
3. Repeat Step 2 until there are (V-1) edges in the spanning tree.

Code (from [GeeksForGeeks - Neelam Yadav ](https://www.geeksforgeeks.org/kruskals-minimum-spanning-tree-algorithm-greedy-algo-2/))
```python
# Python program for Kruskal's algorithm to find 
# Minimum Spanning Tree of a given connected,  
# undirected and weighted graph 
  
from collections import defaultdict 
  
#Class to represent a graph 
class Graph: 
  
    def __init__(self,vertices): 
        self.V= vertices #No. of vertices 
        self.graph = [] # default dictionary  
                                # to store graph 
          
   
    # function to add an edge to graph 
    def addEdge(self,u,v,w): 
        self.graph.append([u,v,w]) 
  
    # A utility function to find set of an element i 
    # (uses path compression technique) 
    def find(self, parent, i): 
        if parent[i] == i: 
            return i 
        return self.find(parent, parent[i]) 
  
    # A function that does union of two sets of x and y 
    # (uses union by rank) 
    def union(self, parent, rank, x, y): 
        xroot = self.find(parent, x) 
        yroot = self.find(parent, y) 
  
        # Attach smaller rank tree under root of  
        # high rank tree (Union by Rank) 
        if rank[xroot] < rank[yroot]: 
            parent[xroot] = yroot 
        elif rank[xroot] > rank[yroot]: 
            parent[yroot] = xroot 
  
        # If ranks are same, then make one as root  
        # and increment its rank by one 
        else : 
            parent[yroot] = xroot 
            rank[xroot] += 1
  
    # The main function to construct MST using Kruskal's  
        # algorithm 
    def KruskalMST(self): 
  
        result =[] #This will store the resultant MST 
  
        i = 0 # An index variable, used for sorted edges 
        e = 0 # An index variable, used for result[] 
  
            # Step 1:  Sort all the edges in non-decreasing  
                # order of their 
                # weight.  If we are not allowed to change the  
                # given graph, we can create a copy of graph 
        self.graph =  sorted(self.graph,key=lambda item: item[2]) 
  
        parent = [] ; rank = [] 
  
        # Create V subsets with single elements 
        for node in range(self.V): 
            parent.append(node) 
            rank.append(0) 
      
        # Number of edges to be taken is equal to V-1 
        while e < self.V -1 : 
  
            # Step 2: Pick the smallest edge and increment  
                    # the index for next iteration 
            u,v,w =  self.graph[i] 
            i = i + 1
            x = self.find(parent, u) 
            y = self.find(parent ,v) 
  
            # If including this edge does't cause cycle,  
                        # include it in result and increment the index 
                        # of result for next edge 
            if x != y: 
                e = e + 1     
                result.append([u,v,w]) 
                self.union(parent, rank, x, y)             
            # Else discard the edge 
  
        # print the contents of result[] to display the built MST 
        print "Following are the edges in the constructed MST"
        for u,v,weight  in result: 
            #print str(u) + " -- " + str(v) + " == " + str(weight) 
            print ("%d -- %d == %d" % (u,v,weight)) 
  
# Driver code 
g = Graph(4) 
g.addEdge(0, 1, 10) 
g.addEdge(0, 2, 6) 
g.addEdge(0, 3, 5) 
g.addEdge(1, 3, 15) 
g.addEdge(2, 3, 4) 
  
g.KruskalMST() 
  
#This code is contributed by Neelam Yadav 
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

### Spiral Matrix

Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order.

### Merge meeting times

Given a set of intervals, condense the list so overlapping intervals are merged.

For example:
[(0, 1), (3, 5), (4, 8), (10, 12), (9, 10)] => [(0, 1), (3, 8), (9, 12)]

We first sort our list of intervals by start time so potentially mergeable candidates are adjacent.

Then, we iterate through the list, keeping track of the last merged interval to see if we can continually condense it further or add a new one.

Solution:
```python
def merge_meeting_times(times):
    sorted_times = sorted(times)
    intervals = [sorted_times[0]]

    for curr_start, curr_end in sorted_times[1:]:
        last_start, last_end = intervals[-1]
        
        if curr_start <= last_end:
            intervals[-1] = last_start, max(curr_end, last_end)
        else:
            intervals.append((curr_start, curr_end))
        
        
    return intervals

## tests
test(merge_meeting_times([(0, 1), (3, 5), (4, 8), (10, 12), (9, 10)]), [(0, 1), (3, 8), (9, 12)])
test(merge_meeting_times([(1, 5), (2, 3)]), [(1, 5)])
test(merge_meeting_times([(1, 10), (2, 6), (3, 5), (7, 9)]), [(1, 10)])
```


## Strings

Hopefully we all know about strings so let's just get into problems.

### Reverse a list of characters in place

This is the same problem as reversing any list in place.

```python
def reverse_list(chars):
    for i in range(len(chars) // 2):
        chars[i], chars[~i] = chars[~i], chars[i]
    return chars

## tests
test(reverse_list("a b c d".split(" ")), "d c b a".split(" "))
test(reverse_list("a b c d e".split(" ")), "e d c b a".split(" "))
```

### String to int

We'll go with the [leetcode description](https://leetcode.com/problems/string-to-integer-atoi/).

The function first discards as many whitespace characters as necessary until the first non-whitespace character is found. Then, starting from this character, takes an optional initial plus or minus sign followed by as many numerical digits as possible, and interprets them as a numerical value.

Valid inputs:

`"42" => 42`
`"    -42" => -42`

```python
def stoi(self, str: str) -> int:
    index = 0
    while index < len(str) - 1 and str[index] == ' ':
        index += 1

    is_neg = str[index] == '-'
    num = functools.reduce(lambda running_sum, c: running_sum * 10 + string.digits.index(c), str[index + int(is_neg):], 0) * (-1 if is_neg else 1)
        
    return num
```

### Palindromic Permutation

Given a string determine if any permutation of that string is a palindrome. You can disregard whitespace.

Example: 

`"acta tac"` => `True` (`"taco cat"`)

The definition of a palindrome can be reduced to a string with at most one character that has an odd frequency of occurences. So if all but one letters are even, we can form a palindrome. This leads to the following algorithm:

```python
import functools
import collections

def palindromic_permutation(str):
    letters = collections.defaultdict(lambda: 0)
    for letter in str:
        if letter != ' ':
            letters[letter] += 1
        
    return functools.reduce(lambda odd_count, a: odd_count + 1 if letters[a] % 2 != 0 else odd_count, str, 0) <= 1

assert palindromic_permutation('acto tac') == True
assert palindromic_permutation('acbtc') == False
```

### Replace and remove
Source: EPI 6.4, page 76

Consider the following two rules that are to be applied to an array of characters:

1. Replace each 'a' by two 'd's
2. Delete each entry containing 'b'

The array will always be big enough to contain the final string.

Examples:

`[a, b, a, c, _] => [d, d, d, d, c]`
`[a, c, d, b, b, c, a] => [d, d, c, d, c, d, d]`

The solution should be in-place in O(n) time. This implies we can't shift the characters over to make room for a's or shift back for b's because that would entail O(n^2).

One problem solving technique leads to an algorithm here: solve an easier version of the problem first.

If there are no a's, we can implement the function in-place with one iteration by skipping 'b's and copying the other characters.

If there are no b's, we first compute the final length of the resulting string and then write the result character by character starting from the last character working backwards.

```python
def replace_and_remove(size, s):
    write_idx, a_count = 0, 0
    # Remove b's and count the number of a's
    for i in range(size):
        if s[i] != 'b':
            s[write_idx] = s[i]
            write_idx += 1
        if s[i] == 'a':
            a_count += 1
    
    curr_idx = write_idx - 1
    write_idx += a_count - 1
    final_size = write_idx + 1
    # Replace a's with dd's starting from the end
    while curr_idx >= 0:
        if s[curr_idx] == 'a':
            s[write_idx - 1: write_idx + 1] = 'dd'
            write_idx -= 2
        else:
            s[write_idx] = s[curr_idx]
            write_idx -= 1
        curr_idx -= 1
        
    return s
```

### Convert Base

### Reverse words in a sentence
Source: EPI 6.6

Given an array of characters representign a sentence, reverse the order of the words (space delimeted) in the sentence.

Example: 

Input: `['w', 'h', a', 't', ' ', 'i', 's', ' ', 'u', 'p']`

Output: `['u', 'p', ' ', 'i', 's', ' ', 'w', 'h', a', 't']`

We'll approach this problem with 2 passes. In the first pass, we'll reverse the entire array. In the following pass we'll reverse each word. 

```python
def reverse_words(words):
    def reverse_range(chars, start, end):
        while start < end:
            chars[start], chars[end] = chars[end], chars[start]
            start, end = start + 1, end - 1
    
    reverse_range(words, 0, len(words) - 1)
    start = 0

    while True:
        finish = start
        while finish < len(words) and words[finish] != ' ':
            finish += 1
        
        if finish == len(words):
            break
        
        reverse_range(words, start, finish - 1)
        start = finish + 1

    reverse_range(words, start, len(words) - 1)
    
    return words

assert reverse_words(['w', 'h', 'a', 't', ' ', 'i', 's', ' ', 'u', 'p']) == ['u', 'p', ' ', 'i', 's', ' ', 'w', 'h', 'a', 't']
```

### Minimum window problem

## Tries

### Longest common prefix

## Stacks & Queues

## Binary Trees

## Heaps

## Hash Tables


