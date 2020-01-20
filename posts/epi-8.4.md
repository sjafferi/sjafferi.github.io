## Problem Statement

Given a pathname (absolute or relative), return the shortest equivalent pathname.

## Examples

`"/usr/lib/../bin/gcc"` => `"/usr/bin/gcc"`

`"scripts//./../scripts/awkscripts/././"` => `"scripts/awkscripts"`

## Initial Thoughts

My initial thoughts were to somehow string replace with a regex pattern to handle `//, ./` and `../`. But this ends up being tedious when replacing the correct directory names with shorter routes. 

The question that I should've started with is, "What data structure best suits the probelm?". Since each path is conditionally going to be in the shortened pathname or not based on succeeding entries, a stack works well here.

In particular, we can push entries on to a stack if they're actual paths and pop  / or not add if they're `../` or `//, ./`

We can return the final stack joined with `/` as the shortened pathname.

What remains is just addressing edge cases like maintaining absolute pathname structure, throwing errors when paths are invalid.

## Code

```python
def shorten_pathnames(paths):
    if not paths:
        raise ValueError('Paths cannot be an empty string')
    shortened_paths = []
    for path in paths.split("/"):
        if path == '' or path == '.':
            continue
        if path == '..':
            if not shortened_paths or shortened_paths[-1] == '..':
                shortened_paths.append('..')
            else:
                if shortened_paths[-1][0] == '/':
                    raise ValueError('Invalid directory')
                shortened_paths.pop()
        else:
            shortened_paths.append(path)
    if paths[0] == '/':
        shortened_paths[0] = '/' + shortened_paths[0]
    return '/'.join(shortened_paths)
```