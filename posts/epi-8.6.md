## Problem Description

Compute binary tree nodes in increasing depth order


## Input / Output

```
      1
  2       3
4   5   6   7
```
=>

```
[[1], [2, 3], [4, 5, 6, 7]]
```

## Initial Insights

A simple solution could be to store nodes in an array or hashmap with the index or key indicating the depth. This would have the same time complexity as any traversal (`O(n)`) with space complexity `O(m)` where `m` is the maximum number of nodes in at a single depth.

The insight that leads to a simpler solution is that we can add nodes in increasing depth order by keeping track 2 rows of nodes while traversing. Hence eliminating the need to keep a direct mapping of depth to nodes.

That is, we can use row `d` to generate `d + 1`, record the results and swap to the next pair of rows. This approach leads to the same time and space complexities but is a bit easier to implement.

## Solution

```python
from collections import deque

def depth_order_binary_tree(root):
    q1, q2 = deque(), deque()
    q1.append(root)
    result = []
    while len(q1) > 0:
        result.append([node.data for node in q1])
        q2 = []
        for node in q1:
            if node.left:
                q2.append(node.left)
            if node.right:
                q2.append(node.right)
        q1 = q2
    return result
```

