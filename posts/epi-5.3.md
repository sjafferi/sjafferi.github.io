## Problem Statement
This is a fairly straightforward problem: Delete repeated elements from a sorted array.

## Input / Output
`[1, 1, 2, 3, 4, 5, 5, 6]`  =>  `[1, 2, 3, 4, 5, 6]` 

## Insights 
The main lesson I took from this problem is to question assumptions about the nature of our output and solution.  

For example, if you want to have an `O(n)` time and `O(1)` space solution, you must somehow shift or nullify the duplicate numbers such that the remaining array is in sorted order. I got confused on how I would "remove" these duplicate numbers even after shifting and maintaining the sort. 

At this point, it's important to ask, and even make assumptions as to how you would do something like that. The book goes with returning an index such that all elements after that index are considered invalid. The main question is about shifting, so this makes a lot of sense. 


## Solution

1. Iterate through the array, starting at index 1
2. Maintain a `write index` also starting at 1
3. Once the element at `write index - 1` does not equal the current element, set the element at `write index` to the current element and increment the `write index` by 1

Note: This does not have any effect when the `write index` is pointing to the current element

When it's pointing to a different element, it means that the `write index` was not incremented due to a previous element being a duplicate at that index. Hence, writing to this causes the current element to overwrite the duplicate.

We're left with invalid entries past the `write index`, and a sorted non-repeating array prior to that index.

## Code

```python
def delete_duplicates(A: List[int]) -> int:
     if not A:
          return 0
     
     write_index = 1
     for i in range(1, len(A)):
          if A[write_index - 1] != A[i]:
               A[write_index] = A[i]
               write_index++

     return write_index
```

Another thing I picked up from this solution was to always consider the converse of a solution I come up with. For example, my first intuition was to shift whenever we identify a duplicate. This works, but can get messy quick as you have to deal with the duplicate which is now ahead of your current element being investigated and the next element in question. 


