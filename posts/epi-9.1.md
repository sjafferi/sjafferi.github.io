## Problem Description
Test if a binary tree is balanced or not. A tree is balanced iff every one of its subtrees has at most 1 height diff in its children.

## Initial Approach
The initial approach I came up with was to find the height of each subtree, compare and return an answer. You can get this to work, but run into the issue of returning an answer to the balanced question. 

It would be much simpler if you could compute the heights and whether or not it's balanced at the same time. This is where named tuples comes in.

## Insights gained from this problem
The usefulness of returning tuples within helper functions is widely exploited in EPI. This problem perfectly exemplifies how we can simplify our solution by returning multiple values.


## Solution

```python
import collections

def balanced_bin_tree(tree):
    BalancedStatusWithHeight = collections.namedtuple('BalancedStatusWithHeight', ('balanced', 'height'))
    def traverse(root):
        if not root:
            return BalancedStatusWithHeight(True, 0)
        
        left_height, right_height = 0, 0
        left_balanced, right_balanced = True, True

        if root.left:
            left_balanced, left_height = traverse(root.left)
            
        if root.right:
            right_balanced, right_height = traverse(root.right)
        
        is_balanced = abs(left_height - right_height) <= 1 and left_balanced and right_balanced
        
        return BalancedStatusWithHeight(is_balanced, max(left_height, right_height) + 1)

    return traverse(tree).balanced
```