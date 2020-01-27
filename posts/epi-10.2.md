## Problem Statement

Given a continuous stream of numbers, return the running median at every input.

## Intuition

In order to avoid a full blown search every time we add a new number, we have to somehow leverage the result of previous computations.

We can do this by splitting the running numbers into two roughly equal halves. A max heap for the bottom half and min heap for the top. 

Then querying for the median becomes a straight forward case of either averaging the max of the bottom and min of top if the halves are equal or returning the priority element in the larger half.

A further simplification can be made to reduce some code complexity by adding all of the numbers to the min heap initially and evicting it's min element into the max heap immediately. And then adding a check to ensure that the length of the max heap is not larger than the min heap.

This ensures that the min heap is always the larger half and hence contains the median when the stream length is odd.

## Solution

```python
def stream_median(nums):
    min_heap, max_heap = [], []
    result = []
    
    for x in nums:
        heapq.heappush(max_heap, -heapq.heappushpop(min_heap, x))
        
        if len(max_heap) > len(min_heap):
            heapq.heappush(min_heap, -heapq.heappop(max_heap))
        
        result.append(0.5 * (min_heap[0] + -max_heap[0]) if len(max_heap) == len(min_heap) else min_heap[0])
    
    return result

test_1 = [1, 0, 3, 5, 9, 7]
test(stream_median(test_1), [1, 0.5, 1, 2, 3, 4])
```