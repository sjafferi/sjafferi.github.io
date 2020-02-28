**Problem statement**

Given a very large set of distances, return the k smallest ones.

**Approach**

As soon as you read "k smallest" or "k largest" alarm bells should go off for usage of a heap data structure.

In this particular problem, the set of distances is very large, hence the trivial solution of sorting the set and returning the first k elements may be too computationally heavy.

Instead, we'll add the first k elements we see into a max heap, and then evict the max from the heap every time we encounter a lower number. Thus ensuring that we'll have the k smallest elements in the heap by the end of the iterations (because all larger elements would have been evicted).

**Solution**

```python
def k_closest_stars(distances, k):
    max_heap = []
    
    for distance in distances:
        heapq.heappush(max_heap, -distance)
        if len(max_heap) == k + 1 and -max_heap[0] >= distance:
            heapq.heappop(max_heap)
    
    return sorted([-x for x in max_heap]) # sort is not needed, doing for testing output

test_1 = [52, 33, 24, 67, 28, 19, 13, 76, 7, 412, 331, 13, 1312, 31, 3, 331, 56, 52, 32]
test(k_closest_stars([52, 33, 24, 67, 28, 19, 13], 4), [13, 19, 24, 28])
test(k_closest_stars(test_1, 10), sorted(test_1)[0:10])
```