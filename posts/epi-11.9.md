## Problem Description
Say we have a file containing 1 billion ip addresses. Our job is to find a valid ip not within the file, i.e. find a missing ip address.

We should do this keeping memory constraints in mind. So sorting solutions or anything that manipulates the entire data set at a time won't work. 

We'll have to develop an approach that uses megabytes worth of space.

The ip's are stored as 32bit integers

## Test case
```python

ips = [
    0x0,
    0x1,
    0x2,
    0x3,
    0x4,
    0x6
]

test(find_ip(ips), 0x5)
```

## Solution

```python
def find_ip(ips):
    bucket_cap = 1 << 16
    count the frequency of ips having the same first 16 bits
    counts = [0] * bucket_cap
    for ip in ips:
        counts[ip >> 16] += 1
    
    # if any counts are below 2^16 - 1, then we can explore the second half
    candidate = next(
        i for i, c in enumerate(counts) if c < bucket_cap
    )
    
    candidates = [0] * bucket_cap
    for ip in ips:
        upper_part = ip >> 16
        if candidate == upper_part:
            lower_part = ((1 << 16) - 1) & ip
            candidates[lower_part] = 1
    
    for i, v in enumerate(candidates):
        if v == 0:
            return (candidate << 16) | i
```