## Problem Statement

Given a string of digits, determine all the possible ip addresses that can be made out of it (if any).

An IP address is considered valid if it's the form `xxx.xxx.xxx.xxx`, where `0 <= xxx <= 255`

### Input
`"19216811'`

### Possible outputs
`["192.168.1.1", "19.216.8.11", ... 7 more]`

### My initial thoughts

Initially,  I thought backtracking was the way to go for this problem. After wrangling with the problem, I realized backtracking wouldn't be the best approach for this as we know exactly the number of decisions we have to make. It's much simpler to just consider them one by one.  The book takes this approach

### Insights from this problem

The solution that the book goes with is very straight forward. Find the first part and determine it's validity. If the first part is valid, find the second part and determine its validity and so on until we find parts that are all valid and add it to our solution set. 

The insight I gleaned from this solution is to apply Occam's Razor when dealing with seemingly simple problems. Go with the approach that you would logically use to solve this by hand. Iterate and optimize on top of that if necessary. 

### Solution (EPI, pg. 52)
```python
def compute_valid_ip(ip):
    def is_valid_parts(parts):
        return len(parts) == 1 or (parts[0] != '0' and int(parts) <= 255)

    result, parts = [], [''] * 4
    for i in range(1, min(4, len(ip))):
        parts[0] = ip[:i]
        if is_valid_parts(parts[0]):
            for j in range(1, min(len(ip) - i, 4)):
                parts[1] = ip[i: i + j]
                if is_valid_parts(parts[1]):
                    for k in range(1, min(len(ip) - i - j, 4)):
                        parts[2], parts[3] = ip[i + j: i + j + k], ip[i + j + k:]
                        if is_valid_parts(parts[2]) and is_valid_parts(parts[3]):
                            result.append('.'.join(parts))
    return result
```