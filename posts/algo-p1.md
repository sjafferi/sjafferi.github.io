<!--

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
4. Combine: Using the observation that any non-dominated point on the left with have to be as high or higher than all non-dominated points on the right, we can filter the left points by checking them against any right point and return `[filtered_left, right]`. (we can use the first element in right as the highest because non-dominated point sets non-increasing).

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

Input: {2, 1, -5, 4, -3, 1, -3, 4, -1}
Output: Subarray with the largest sum is {4, -1, 2, 1} with sum 6.

Input: {-3, 1, -3, 4, -1, 2, 1, -5, 4}
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

### Maximum Subarray - Divide & Conquer

### Subarray Sum - Divid & Conquer

```

var Mocha = require('mocha')
var assert = require('assert')
var mocha = new Mocha()

mocha.suite.emit('pre-require', this, 'solution', mocha)

describe('Test rearrange string', function() {
  [
    ['aab','aba'],
    ['aaa', ''],
    ['aaab', ''],
    ['aaabb', 'ababa']
  ].forEach(
    ([input, output]) => it(`input: ${input}`, () => {
      const actual = rearrangeString(input);
      assert(actual === output, `Expected: ${output}, Actual: ${actual}`)
    }
  ))
})

mocha.run()


function rearrangeString(S) {

}
```
-->
