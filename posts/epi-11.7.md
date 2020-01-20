## Problem Description

Find the min and max of a list of numbers in the least amount of comparisons

The brute-force approach would take 2n comparisons if we find the 
min and max individually. However, this fails to take advantage of 
the relative comparisons. 

## Insights

We can use the fact that a < b and b < c => a < c

Essentially eliminating a comparison by first determining min and max candidates
We can do this in a pair-wise fashion

```python
import collections

MinMax = collections.namedtuple('MinMax', ('smallest', 'largest'))

def find_min_max(arr):
    def min_max(a, b):
        return MinMax(a, b) if a < b else MinMax(b, a)
    
    if len(arr) <= 1:
        return MinMax(arr[0], arr[0])
    
    global_min_max = min_max(arr[0], arr[1])

    for i in range(2, len(arr) - 1, 2):
        local_min_max = min_max(arr[i], arr[i+1])
        global_min_max = MinMax(
            min_max(local_min_max.smallest, global_min_max.smallest).smallest,
            min_max(local_min_max.largest, global_min_max.largest).largest
        )
    
    if len(arr) % 2 != 0:
        global_min_max = MinMax(
            min_max(global_min_max.smallest, arr[-1]).smallest,
            min_max(global_min_max.largest, arr[-1]).largest
        )

    return global_min_max

test(find_min_max([1, 3, 2, 4, -1]), MinMax(-1, 4))
test(find_min_max([1, 3, 2, -1, 52, 1]), MinMax(-1, 52))
test(find_min_max([1, 3, 2, 3, 1]), MinMax(1, 3))
```