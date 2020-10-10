# What's in this guide

This is meant to be a comprehensive guide for mastering technical interviews on algorithms and data structures. It includes a ground up understanding of the theory (relying on [UWaterloo's CS341](https://www.student.cs.uwaterloo.ca/~cs341/) and [Skiena's Algorithm Design Manual](http://www.algorist.com/)) as well as problems aggregated from books (relying heavily on [Elements of Programming Interviews](https://www.amazon.com/gp/product/1537713949?ie=UTF8&tag=sjafferi-20&camp=1789&linkCode=xm2&creativeASIN=1537713949)) and the web.

This guide assumes you have some background in algorithms and hence won't reiterate common definitions.

By the end of each section, you should have the tools necessary to come up with on the spot solutions to related problems.

## Study Plan

The plan of attack is as follows:

1. Understand the underlying theory behind each data structure
   - Implementation, applications, complexities, traversals
2. Study problems representative of the data structure
   - Core problems that can be used to derive solutions to many other problems
3. Iterate until perfect
   - Use spaced repetition (linked Anki cards) to commit learnings to long term memory

## Spaced Repetiton with Anki Cards

Knowing algorithms commonly used for specific purposes can greatly boost on-demand problem solving ability by enabling crafting of a solution using some combination of past knowledge. There are "template problems" that other problems of the same sort can be reduced to that greatly help with this (e.g [Sliding Window Problems](https://medium.com/outco/how-to-solve-sliding-window-problems-28d67601a66)].

The amount of material required to do predictably exceptional in interviews can get staggering. This is where spaced repetition and anki cards come in.

[Spaced repetition](https://en.wikipedia.org/wiki/Spaced_repetition) is a well researched technique to improve long term recall of information through the use of meaningful spaces in review. ...research

[Anki cards](https://apps.ankiweb.net/) are flash cards that are shown to you based on an algorithm that uses a rating of how difficult the problem was to calculate when you'll be shown the card next. ...research

This guide includes Anki cards for each major topic (theory + problems) that can be used as supplemental material.

# Algorithms

## Backtracking / Comprehensive Search

From Algorithm Design Manual, 7.1

> Backtracking is a systematic way to iterate through all the possible configurations of a search space. These configurations may represent all possible arrangements of objects (permutations), or all possible ways of building a collection of them (subsets).

> What these problems have in common is that we must generate each possible configuration exactly once. Avoiding both repitions and missing configurations means that we must define a systematic generation order.

> We will model our combinatorial search solution as a vector a = <a1, ... , an> where each element ai is selected from a finite ordered set Si. Such a vector might represent an arrangement where ai contains the ith element in the permutation or some subset correspnding to the ith element in the search space, or a sequence of moves in a game or a path in a graph...etc

> At each step in the backtracking algorithm, we try to extend a given partial solution a = <a1, ... , ak> by adding another element at the end. Once we extend it, we have to test whether this what we now have is a solution and do something with it. If not, we must continue the search if the partial solution is still potentially extendible to some complete solution.

This creates a tree of partial solutions; where each vertex represents a partial solution. There is an edge from x to y if node y was created by advancing from x. This tree of parital solutions provides an alterantive way to think about backtracking.

> Backtracking ensures correctness by enumerating all possibilities. It ensures efficiency by never visiting a state more than once.

This chapter from Skiena's text was a true mind opener for me. Realizing that the solution space of any search related problem can be traversed through one subroutine that gaurentees correctness is pretty cool. The following realization was that many of these problems involve enumerating permutations or creating subsets of the solution space in a directed manner. The subroutines for these generalized backtracking solutions are presented below and can be modified to _many_ problems.

### Generalized Backtracking Algorithm

Given the following subroutines, we can use this generalized backtracking algorithm to complete the the questions in this chapter (their usual implementations will also be given).

```python
class Backtrack:
    # setting up generalization with required subroutines
    def __init__(self,
                 is_a_solution,
                 process_solution,
                 construct_candidates,
                 make_move = None,
                 unmake_move = None,
                 is_finished = None
                ):
        self.is_a_solution = is_a_solution
        self.process_solution = process_solution
        self.construct_candidates = construct_candidates
        self.make_move = make_move
        self.unmake_move = unmake_move
        self.is_finished = is_finished

    # This is the meat and bones
    def backtrack(self, a, k, input):
        if self.is_a_solution(a, k, input):
            self.process_solution(a, k, input)
        else:
            k += 1
            candidates = self.construct_candidates(a, k, input)
            for c in candidates:
                a[k] = c

                if self.make_move:
                    self.make_move(a, k, input)

                self.backtrack(a, k, input)

                if self.unmake_move:
                    self.unmake_move(a, k, input)

                if self.is_finished and self.is_finished(a, k, input):
                    return
```

### Constructing all subsets

A power set of a set S is the set of all subsets of S.

In order to come up with a way to generate subsets, we can think about how to enumerate all subsets in {1, ..., n}.

Using concrete examples:

```
Let num_subsets(i) represent the number of subsets for the set {1, ..., i}

num_subsets(1) = 2
    {}, {1}

num_subsets(2) = 4
    {}, {1}, {2}, {1, 2}

num_subsets(3) = 8
    {}, {1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3}

...

num_subsets(n) = 2^n
    {}, {1}, {2}, ...  {1, 2, ..., n - 1}, {1, 2, ..., n}
```

We can see that all subsets leading to the original set include an element that was not in it previously. This lends itself to an easy recurisve solution.

Particularly, we examine all subsets that include a given element and all subsets that don't include that element. Then, the power set at this step (k) is the union of these two sets.

For example, if we have {1, 2, 3}

We first generate all subsets that include 1 and union it with all subsets that don't include 1.

The next recursive call computes all subsets that include 2 and all subsets that don't include 2.

And so on until we reach the end of the list.

This is how that looks like in code.

**Without generalization**

```python
def generate_all_subsets(input_set):
    def generate_helper(to_be_selected, selected_so_far):
        if to_be_selected == len(input_set):
            result.append(selected_so_far)
            return

        generate_helper(to_be_selected + 1, selected_so_far)
        generate_helper(to_be_selected + 1, selected_so_far + [input_set[to_be_selected]])


    result = []
    generate_helper(0, [])
    return result
```

Using the generalized algorithm we can define the candidates as either `[]` or `k`. Indicating the inclusion or exclusion of this element from the complete solution.

**With generalization**

```python
def gen_subsets_is_a_solution(a, k, n):
    return n - 1 == k

def gen_subsets_construct_candidates(a, k, n):
    return [None, k]

def gen_subsets_process_solution(a, k, n):
    print([i for i in a if i is not None])
```

Then, we can call the backtrack subroutine

```python
generate_subsets_backtrack = Backtrack(
                                gen_subsets_is_a_solution,
                                gen_subsets_process_solution,
                                gen_subsets_construct_candidates,
                            )
n = 5
soln_space = [0] * n
generate_subsets_backtrack.backtrack(soln_space, -1, n)
```

### Constructing all permutations

A similar enumeration argument can be used to give us an intuition on how to construct all permutations.

To enumerate all permatutions of length 5, we consider the number of options for the first position in the permutation, followed by the next, and so on until the very last element.

```
n = 5
Number of options for:
1st element: 5
2nd element: 4
3rd element: 3
4th element: 2
5th element: 1

Total possibilities = 5x4x3x2x1 = 5!
```

This leads to a natural recursive solution.

1. Generate candidates for the current element. If this is the first element, this will be the entire array. For the rest, it is a subset of size `n - k` (indexed 0)
2. Iterate through the list of candidates, and fix one to the current position.
3. With this element fixed, recurse to the next element `k + 1`
4. Once `k == n - 1`, stop and print the current permutation

Once the recursive calls return and backtrack, the rest of the candidate set will be considered for each step in the recursion. Hence, this covers all possible permutations.

**Without generalization**

```python
def generate_permutations(A):
    def generate(i):
        if i == len(A) - 1:
            result.append(A.copy())
            return

        for j in range(i, len(A)):
            A[i], A[j] = A[j], A[i]
            generate(i + 1)
            A[i], A[j] = A[j], A[i]

    result = []
    generate(0)
    return result
```

**With generalization**

```python
def gen_permutations_is_a_solution(a, k, n):
    return n - 1 == k

def gen_permutations_construct_candidates(a, k, n):
    candidates = []
    in_perm = [False] * n

    for i in range(k):
        in_perm[a[i]] = True

    for i in range(n):
        if not in_perm[i]:
            candidates.append(i)

    return candidates

def gen_permutations_process_solution(a, k, n):
    print([i + 1 for i in a])
```

Then, we can call the backtrack subroutine

```python
gen_permutations_backtrack = Backtrack(
                                gen_permutations_is_a_solution,
                                gen_permutations_process_solution,
                                gen_permutations_construct_candidates,
                            )
n = 5
soln_space = [-1] * n
gen_permutations_backtrack.backtrack(soln_space, -1, n)
```

### Tower of Hanoi

EPI 15.1 - Tower of Hanoi, see [problem description here](https://en.wikipedia.org/wiki/Tower_of_Hanoi)

```python
def solve_tower_of_hanoi(n):
    def helper(curr_n, source, to, intermediary):
        if curr_n <= 0:
            return
        # move n - 1 rings to second peg
        helper(curr_n - 1, source, intermediary, to)
        # move nth ring to third peg
        pegs[to].append(pegs[source].pop())
        # move n - 1 rings from second peg to third using first as intermediary
        helper(curr_n - 1, intermediary, to, source)

    pegs = [list(reversed(range(1, n + 1)))] + [[]] + [[]]
    helper(n, 0, 2, 1)
    return pegs
```

### Attacking queens

Generate all non attacking placements of n-queens in an nxn board. Also known as the [Eight Queens Puzzle](https://en.wikipedia.org/wiki/Eight_queens_puzzle).

This is a classic backtracking problem as there is no other way to compute the results than comprehensive search.

```python
def non_attacking_queens(n):
    def non_attacking_helper(row):
        if row >= n:
            result.append(partial_result.copy())

        for col in range(n):
            if all(
                    abs(c - col) not in (0, row - i)
                    for i, c in enumerate(partial_result[:row])):
                partial_result[row] = col
                non_attacking_helper(row + 1)


    result = []
    partial_result = [0] * n
    non_attacking_helper(0)
    return result

assert non_attacking_queens(4) == [[1, 3, 0, 2], [2, 0, 3, 1]]
```

### Phone Mnemonics

Older phones had numbers mapped to keys, allowing you to create mnemonics to help remember them. Given a phone number, determine all the possible mnemonics that can be generated from it.

Key mappings are given below.

The format follow very closely to the previous problem.

```python
KEYS = {
    "2": "abc",
    "3": "def",
    "4": "ghi",
    "5": "jkl",
    "6": "mno",
    "7": "pqrs",
    "8": "tuv",
    "9": "wxyz"
}

def phone_mnemonic(number):
    def mnemonic_helper(i):
        if i >= len(number):
            results.append("".join(partial_result))
            return

        if number[i] in KEYS:
            for char in KEYS[number[i]]:
                partial_result[i] = char
                mnemonic_helper(i + 1)

    results = []
    partial_result = [0] * len(number)
    mnemonic_helper(0)
    return results

assert "acronym" in phone_mnemonic("2276696")
```

### Generate all subsets of size k

Given n and k, generate all subsets of size k in [1, 2, ..., n].

The brute force approach is to generate all possible subsets and filter for sized k subsets. This obviously performs more work than necessary as it continues subset computation for invalid subsets.

We can make this more efficient by implementing the same heuristic as the previous question. i.e. generate all subsets of size k that include some element and all subsets of size k that don't.

For elements that include the element, the remaining size becomes k - 1. The size remains k for the subset that does not include the element.

Instead of using recursive calls to generate all subsets including and not including an element, we'll iterate through the remaining elements and include / disclude them while fixing the current element.

This will make it easier to add exactly k elements.

```python
def generate_subsets_size_k(n, k):
    def generate_helper(offset, partial_result):
        if len(partial_result) == k:
            result.append(partial_result.copy())
            return

        num_remaining = k - len(partial_result)
        i = offset
        while i <= n and num_remaining <= n - i + 1:
            generate_helper(offset + 1, partial_result + [i])
            i += 1

    result = []
    generate_helper(1, [])
    return result
```

### Generate strings of matched parens

Given an integer k, return all strings with tahat number of matched pair of parens

The brute force approach would be to generate all possible strings with parens of size 2k, and filter for strings with valid parens.

This approach does too much work because it continues computing strings that could never be a valid parens.

We can perform a more directed search by using the heuristic that at every step of the generation, our partial result has the possibility of becoming a valid string.

We can break this down into cases:

1. Can we add a left paren? If so, add and continue.
2. Can we add a right paren? If so, add and continue.

In order to answer these questions, we can keep track of how many left parens are remaining and how many right parens are remaining.

```python
def generate_matched_parens(k):
    def generate_helper(num_left_parens_remaining, num_right_parens_remaining, partial_result, result = []):
        if num_left_parens_remaining > 0:
            generate_helper(num_left_parens_remaining - 1, num_right_parens_remaining, partial_result + '(')

        if num_left_parens_remaining < num_right_parens_remaining:
            generate_helper(num_left_parens_remaining, num_right_parens_remaining - 1, partial_result + ')')

        if not num_right_parens_remaining:
            result.append(partial_result)

        return result

    return generate_helper(k, k, "")
```

### Generate palindromic decompositions

Given a string s, compute all palindromic decompositions of s.

The brute force approach is to compute all possible decompositions of s and filter for palindromic ones (see a pattern here?)

This does too much computation because it continues computing decompositions that aren't palindromic.

We can perform a more directed search by only continuing with decompositions that are palindromic.

So we'll branch off only when we've gotten a valid palindromic decomposition.

```python
def generate_palindromic_decomps(input_str):
    def generate_helper(offset, partial_result):
        if offset == len(input_str):
            result.append(partial_result.copy())
            return

        for i in range(offset + 1, len(input_str) + 1):
            prefix = input_str[offset:i]
            if prefix == prefix[::-1]: # prefix == reverse(prefix)
                generate_helper(i, partial_result + [prefix])


    result = []
    generate_helper(0, [])
    return result
```

### Generate binary trees

Generate all binary trees with n nodes.

We can perform a directed search by computing all left subtrees of size i and right subtrees of n - i for i in 1 -> n. This can be done without a helper function.

```python
def generate_binary_trees(n):
    if n == 0:
        return [None]

    result = []

    for i in range(n):
        left_subtrees = generate_binary_trees(i)
        right_subtrees = generate_binary_trees(n - i - 1)
        result += [
            BinaryTreeNode(0, left, right)
            for left in left_subtrees
            for right in right_subtrees
        ]

    return result
```

### Sudoku solver

Given a partially completed sudoku board, solve the board if possible.

The brute force approach is to try every combination of board completions and filter for the ones that are valid.

This approach does too much work because it does not stop when there is an invalid board.

A more directed search could only continue with valid boards until the board is complete. We can use the fact that our partial result is a valid board to only check new additions for validity instead of the entire board.

### Compute gray code

Write a program that takes n and returns an n-bit [Gray Code](https://en.wikipedia.org/wiki/Gray_code)

The brute force approach would be to enumerate all permuations of 0, ..., 2^n - 1 and stop once we find a gray code.

This approach does too much work because it continues with permutations that cannot be gray codes.

A more directed approach would be to implement a heuristic that creates a partial valid value at every step. We can do this by, starting of with [0000] (if n = 4) and try changing 1 bit to find the next num that is not already in the set. So [0000, 0001, ...].

```python
def gray_code(n):
    num_elems = 2**n

    def directed_search(partial_result):
        def differs_by_one(x, y):
            bit_difference = x ^ y
            return bit_difference and not (bit_difference & (bit_difference - 1))

        if len(partial_result) == num_elems:
            return differs_by_one(result[0], result[-1])

        for i in range(n):
            previous_code = result[-1]
            candidate_next_code = previous_code ^ (1 << i)
            if candidate_next_code not in partial_result:
                partial_result.add(candidate_next_code)
                result.append(candidate_next_code)
                if directed_search(partial_result):
                    return True

                del result[-1]
                partial_result.remove(candidate_next_code)

        return False

    result = [0]
    directed_search(set([0]))
    return result
```

## Dynamic Programming

DP is a generalized technique that enables solving searching, counting and optimization problems by breaking them down into subproblems. It's great for tabulation or memoization when there isn't a clearly efficient solution.

From Algorithm Design Manual, Chapter 8

> Algorithms for optimization problems require proof that they always return the best possiible solution. Greedy algorithms that make the best local decision at each step are typically efficient but usually do not guarentee global optimality. Exhaustive search algorithms that try all possibilities and select the best always produce the optimum result, but usually at a prohibitive cost in terms of time complexity.

> Dynamic programming combines the best of both worlds. It gives us a way to design custom algorithms that systematically search all possibilities while storing results to avoid recomputing.

Skiena recommends starting with the recursive algorithm or definition first, and then optimizing with DP.

> DP is generally the right method for optimization problems on combinatorial objects that have an inherent left to right order among components.

> Left to right objects include character strings, rooted trees, polygons and integer sequences

[Geeks for geeks](https://www.geeksforgeeks.org/optimal-substructure-property-in-dynamic-programming-dp-2/) provides a way of identifying DP problems.

There are two main properties that suggest a problem can be sovled with DP:

1. Overlapping subproblems -> memoization
2. Optimal substructure -> If an optimal solution of the given problem can be determined using optimal solutions of subproblems

From Geeks for Geeks:

> For example, the Shortest Path problem has following optimal substructure property:
> If a node x lies in the shortest path from a source node u to destination node v then the shortest path from u to v is combination of shortest path from u to x and shortest path from x to v.

### Maximum subarray

### Edit Distance

The [Edit Distance](https://en.wikipedia.org/wiki/Edit_distance) between two words is the minimum number of "edits" it would take to transform one word into another. A single edit is either a insertion, deletion, or substitution.

Compute the minimum edit distance for two strings.

**Examples**

Saturday, Sunday => 3
abcd, fgcd => 2

Firstly, let's define some terminology and our subproblems.

For two strings:

- X of length n,
- Y of length m

We define ED(i, j) as:

- The edit distance between X[0:i] and Y[0:j]

Therefore, the solution to our problem is ED(n, m)

Looking at the last characters of the each string leads us to this observation:

- If the last character of X == last character of Y, then ED(n, m) = ED(n - 1, m - 1)
- If the last characters are not equal, then we have the following options:
  - Replace the last character of X -> ED(n, m) = ED(n - 1, m - 1) + 1
  - Delete the last character of X -> ED(n, m) = ED(n - 1, m) + 1
  - Insert the last character of Y at the end of X -> ED(n, m) = ED(n, m - 1) + 1
  - The minimum of these three is equal to ED(n, m)

This leads to the following algorithm:

```python
def edit_distance(str1, str2):
    def compute_distances(str1_idx, str2_idx):
        if str1_idx < 0:
            return str2_idx + 1
        elif str2_idx < 0:
            return str1_idx + 1

        if edit_distances[str1_idx][str2_idx] == -1:
            if str1[str1_idx] == str2[str2_idx]:
                edit_distances[str1_idx][str2_idx] = compute_distances(str1_idx - 1, str2_idx - 1)
            else:
                replace_last = compute_distances(str1_idx - 1, str2_idx - 1)
                insert_last = compute_distances(str1_idx, str2_idx - 1)
                delete_last = compute_distances(str1_idx - 1, str2_idx)
                edit_distances[str1_idx][str2_idx] = 1 + min(replace_last, insert_last, delete_last)

        return edit_distances[str1_idx][str2_idx]

    n = len(str1)
    m = len(str2)
    edit_distances = [[-1] * m for i in range(n)]
    return compute_distances(n - 1, m - 1)
```

### Count number of ways to move down 2d grid

Given an nxm 2d array, determine the number of paths you can take to go from top left to bottom right.

Again, lets start with some definitions.

Let N(i, j) be the number of ways to traverse to A[i][j] from A[0][0]

Then, our answer is N(n - 1, m - 1)

N(i, j) can be defined as:

- N(i - 1, j) + N(i, j - 1)
- i.e. the number of ways to get directly above + number of ways to get directly left of A[i][j]

```python
def num_ways_traverse(A):
    n = len(A)
    m = len(A[0])

    def find_num_ways(i, j):
        if i == j == 0:
            return 1

        if num_ways[i][j] == 0:
            ways_top = 0 if i == 0 else find_num_ways(i - 1, j)
            ways_left = 0 if j == 0 else find_num_ways(i, j - 1)
            num_ways[i][j] = ways_top + ways_left

        return num_ways[i][j]

    num_ways = [[0] * m for _ in range(n)]
    return find_num_ways(n - 1, m - 1)
```

### Search for a sequence in a 2D array

Given a 2D array, A, determine if a pattern (1D array) exists in A.

The pattern exists in A if you can start from some netry in A and traverse adjacent entries in the order of the pattern until all elements of the pattern are found.

**Example**
A = [
[1, 2, 3],
[3, 4, 1],
[5, 6, 7]
]
P = [1, 3, 4, 6]
Pattern exists @ [ (0, 0), (1, 0), (1, 1), (2, 1) ]

We can define the subproblem as:

- Exists(x, y, offset) = True iff P[offset:] is found ending at A[x][y]

Then our answer becomes:
at least 1 of [Exists(i, j, 0) for all i in range(A) and all j in range(A[i])] is True

In order to find Exists(i, j, offset), we have to determine:

- if we have finished pattern -> return True
- if we A[i][j] is out of bounds or A[i][j] != P[offset] -> return False
  - Note: we can return early here by keeping track of unsuccesful combinations of (i, j, offset),
- if for all neighbors of (i, j) at least one Exists(neighbor_x, neighbor_y, offset + 1) is True -> True
- else -> False

Now, this essentially becomes a backtracking search with an added cache for early returns.

```python
def search_for_sequence(A, P):
    def find_pattern(x, y, offset):
        if offset == len(P):
            return True

        if (not (0 <= x < len(A) and 0 <= y < len(A[x]))
            or A[x][y] != P[offset]
            or (x, y, offset) in prev_attempts
        ):
            return False

        if any(
            find_pattern(x + a, y + b, offset + 1)
            for a, b in ((-1, 0), (0, -1), (1, 0), (0, 1))
        ):
            return True

        prev_attempts.add((x, y, offset))
        return False

    prev_attempts = set()
    return any(
        find_pattern(i, j, 0)
        for i in range(len(A))
        for j in range(len(A[i]))
    )
```

### Word break

```python
def word_break(s: str, wordDict: List[str]) -> bool:
    def is_word(j):
        if j < 1:
            return True

        if words[j - 1] == -1:
            words[j - 1] = any(s[i:j] in wordDict and is_word(i) for i in range(j))

        return words[j - 1]

    n = len(s)
    words = [-1] * n

    return is_word(n)
```

### Concatenate words

Source: https://leetcode.com/problems/concatenated-words/

Given a list of words (without duplicates), please write a program that returns all concatenated words in the given list of words.
A concatenated word is defined as a string that is comprised entirely of at least two shorter words in the given array.

```
Example:
Input: ["cat","cats","catsdogcats","dog","dogcatsdog","hippopotamuses","rat","ratcatdogcat"]

Output: ["catsdogcats","dogcatsdog","ratcatdogcat"]

Explanation: "catsdogcats" can be concatenated by "cats", "dog" and "cats";
 "dogcatsdog" can be concatenated by "dog", "cats" and "dog";
"ratcatdogcat" can be concatenated by "rat", "cat", "dog" and "cat".
```

The approach we'll take will be similar to the word break problem. Specifically, we'll determine if a word in the list of words is concatenable by treating the list of words as the dict and re-using a similar subproblem.

Namely, is_concat(word) = word[:i] in dict and (word[i:] in dict or is_concat(word[i:])) for i in range(1, len(words))

Note the differences here:

```python
def concatenated_words(self, words: List[str]) -> List[str]:
    dict_words = set(words)

    if len(words) < 2:
        return []

    def is_concat(word):
        if word in is_concat_table:
            return is_concat_table[word]

        is_concat_table[word] = False

        for i in range(1, len(word)):
            if word[:i] in dict_words and (word[i:] in dict_words or is_concat(word[i:])):
                is_concat_table[word] = True
                break

        return is_concat_table[word]

    is_concat_table = {}

    results = [word for word in words if is_concat(word)]
    return results
```

### Smallest subarry sequentially covering subset

Source: EPI 12.7

Given 2 lists of words P & K, find the length of the smallest subarray in P that _sequentially_ covers all values in P.

```
Example:
P = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "2", "4", "6", "1", "0", "1", "0", "1", "0", "3", "2", "1", "0"]
K = ["0", "2", "9", "4", "6"]

Output: 13

Explanation: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "2", "4", "6"] is the smallest subarray sequentially covering all values
```

At first glance, this seems like a sliding window problem (described below) as we're trying to minimize a subrange in the input set. However, after some trial and error, it becomes clear that the pointer method is not ideal for a linear solution due to the difficulty in jumping back and forth between candidate sets.

One brute force approach we can use here is to find the smallest subarry ending at the current index by iterating through the previously seen values in P.

The intuition that leads to a cleaner algorithm follow from optimizing the brute force approach. This is because the brute force approach fails to take advantage of the fact that we've already seen the previous elements. In order to do this, we need to be able to calculate the subarray length for each candidate index based on some past computation.

This leads to the following DP solution

```
S(i) = Shortest subarray in P[:i+1] sequentially containing all values in K[:j] (let j = last seen keyword idx in K + 1)
        if P[i] not in K[:j] => -1
        else => S(k) + (i - k) where k = last seen keyword idx in P
```

Implementation

```python
def find_smallest_sequentially_covering_subset(paragraph, keywords):
    ans = Subarray(0, len(paragraph))

    keywords_to_idx = {key: idx for idx, key in enumerate(keywords)}
    last_seen = collections.defaultdict(lambda: -1)
    S = [-1] * len(paragraph)

    for idx in range(len(paragraph)):
        if paragraph[idx] in keywords:
            keyword = paragraph[idx]
            keyword_idx = keywords_to_idx[keyword]
            last_seen[keyword] = idx

            if keyword_idx == 0:
                S[idx] = 1
            elif last_seen[keywords[keyword_idx - 1]] != -1:
                last_key_idx = last_seen[keywords[keyword_idx - 1]]
                S[idx] = S[last_key_idx] + (idx - last_key_idx)
            else:
                last_seen[keyword] = -1

            if last_seen[keyword] != -1 and keyword_idx == len(keywords) - 1:
                if ans[1] - ans[0] > S[idx] - 1:
                    ans = Subarray(idx - S[idx] + 1, idx)

    return ans
```

### Find the minimum weight path in a triangle

### Sliding Window Problems

[This blog post](https://medium.com/outco/how-to-solve-sliding-window-problems-28d67601a66) goes over sliding window problems very well.

Essentially, they are a subset of DP problems that have certain similar properties. The advantage to identifying sliding window problems is that they can be reduced a few different traversal routines.

The properties are:

1. You are looking for a subrange in your problem input set
2. This subrange must be **optimal** (like longest, or shortest)

There are 4 types of sliding window problems with different traversal routines involved for solving.

1. Fast / Slow
2. Fast / Catchup
3. Fast / Lag
4. Front / Back

Note: The speed here refers to how the window will shrink / grow

#### Fast / Slow

The window will grow with the fast pointer until your condition is met.

The window will shrink with the slow pointer until you no longer have a valid window.

##### Minimum window substring

Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n).

This falls in the bucket of fast / slow because we need to grow our window until we have a valid substring that contains all the letters we are looking for, and then shrink that window until we no longer have a valid substring.

Now we need to determine when to grow or shrink.

We want to continue growing until we have a valid window (i.e. the window contains all letters of T). Then we want to shrink until the window is no longer valid.

We'll iterate through the characters in S using an index called fast. We'll also initialize a slow counter to 0.

We can use a dict to keep track of characters currently within the window, and a counter for the current number of missing elements in the window.

Once we have configure the dict and counter to update properly, we can use the counter to detect if the window is valid (num_missing == 0).

Then, we can use a while loop to bring the slow pointer up until the window is no longer valid.

We'll end up searching all valid substrings in at most O(2n) time.

```python
def min_window(s: str, t: str) -> str:
    if not t or not s:
        return ""

    chars_in_window = {}
    freq_t = collections.defaultdict(lambda: 0)

    for char in t:
        chars_in_window[char] = 0
        freq_t[char] += 1

    num_missing = len(t)
    result = float('-inf'), float('inf')

    slow = 0
    for fast in range(len(s)):
        if s[fast] in chars_in_window:
            if chars_in_window[s[fast]] < freq_t[s[fast]]:
                num_missing -= 1

            chars_in_window[s[fast]] += 1

        while slow <= fast and num_missing == 0:
            if fast - slow < result[1] - result[0]:
                result = slow, fast

            if s[slow] in chars_in_window:
                if chars_in_window[s[slow]] <= freq_t[s[slow]]:
                    num_missing += 1

                chars_in_window[s[slow]] -= 1

            slow += 1


    return s[result[0]:result[1] + 1] if result[0] > float('-inf') and result[1] < float('inf') else ''
```

#### Repeating Characters

Source: [swecareers](https://www.swecareers.com/problem/maximum-substring-with-non-repeating-characters)

Given a string, find the length of the longest substring without repeating characters.

**Examples**

Example 1:
Input: "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.

Example 2:
Input: "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.

Example 3:
Input: "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.

In this problem, we have to traverse all substrings without repeating characters. Hence our window will increase until we have an invalid substring, and then shrink until it's valid again.

We'll keep a mapping of current characters in the window and their indices in order to determine if there are repeating characters. We'll use the indices to calculate how far we'll bring up the slow pointer to get a valid window.

```python
import collections

def longest_substring(s: str) -> int:
    char_map = collections.defaultdict(lambda: -1)
    slow = 0
    max_len = 0
    for fast in range(len(s)):
        if char_map[s[fast]] != -1:
            while slow <= char_map[s[fast]]:
                char_map[s[slow]] = -1
                slow += 1
            ## slow == char_map[s[fast]] + 1 (1 element above the index we last saw s[fast], hence the window becomes valid again)
        else:
            max_len = max(fast - slow, max_len)

        char_map[s[fast]] = fast

    return max_len + 1
```

##### Consecutive sum subarray

#### Fast / Catchup

The fast pointer works the same way as mentioned above.

The catchup pointer will jump to the fast pointer once a condition is met.

##### Max consecutive sum

##### Bit flip

Source: [Maximize number of 0s by flipping a subarray](https://www.geeksforgeeks.org/maximize-number-0s-flipping-subarray/)

Given a binary array, find the maximum number zeros in an array with one flip of a subarray allowed. A flip operation switches all 0s to 1s and 1s to 0s.

**Examples**
Input: arr[] = {0, 1, 0, 0, 1, 1, 0}
Output: 6
We can get 6 zeros by flipping the subarray {4, 5}

Input: arr[] = {0, 0, 0, 1, 0, 1}
Output: 5

##### Buy / sell stocks

#### Fast / Lag

The lag pointer is referencing a few indices behind the fast.

##### Knapsack problem

#### Front / back

You have one pointer at the back and one at the front. You move either or both according to some condition.

##### Trapped rainwater

##### Sorted two sum

## Graph Theory

From Algorithm Design Manual, 5.1

> Graph theory provides a language for talking about the properties of relationships, and it is amazing how often messy applied problems have simple descriptions and solutions

Creating models using graph analogies allows us to to systematically search the solution space using the relationship modeled edges and nodes. The backtracking model essentially applies a form of depth first search to conduct a comprehensive search. However, if our problem is modeled well enough, searching the entirety of the solution space is not necessary.

This leads to the fundamental problem in graph theory: Visiting every edge and vertex in a systematic way.

### Breadth First Search

BFS conducts a systematic search through the graph by maintaining a queue of undiscovered nodes. This set of nodes is continously dequed and enqueued (when more undiscovered nodes are encountered) until it is empty (i.e. all nodes are discovered).

This is the basic subroutine for BFS

```python
bfs(g, start):
    queue = [start]
    discovered[start] = True

    while queue is not empty:
        v = deque(queue)
        processed[v] = True
        p = g.edges[v]
        while p:
            if not processed[p]:
                process(v, p)
            else:
                enqueue(queue, p)
                discovered[p] = True
            p = p.next
```

Common applications for BFS include

- Path finding: Due to the search order of BFS, it will always find the shortest path from the source to a target vertex. This is because vertices proccessed in a first in, first out order. Hence, the starting node's neighbors will always be processed first, followed by the first neighbor's neighbors, second neighbor's neighbors and so on.

- Connected components: An amazing number of seemingly complicated problems reduce to finding or counting connected components (solving Rubik's cube, Sudoku). This can be done by iterating through all of the graph's vertices and running BFS it the current vertex has not yet been discovered (by previous run of BFS). Every undiscovered vertex in this loop will be a different connected component.

- Two-coloring Graphs: Identifying bipartite graphs (graphs that can be colored with 2 colors such that neighboring vertices do not share the same color) can be done with BFS since it processes vertices the same distance from the root sequentially. Hence if a child vertex with the same color as it's parent is found on the search path, it cannot be bipartite.

### Depth First Search

DFS performs a systematic graph traversal through a stack; hence vertices are processed in a last in, first out fashion. A vertex's neighbor is processed completely before it returns back to process the rest of its siblings. This lends itself it a neat recursive implementation.

```python
dfs(g, v):
    if (is_finished(g, v)):
        return

    discovered[v] = True
    process_vertex_before(v)

    p = g.edges[v]
    while p:
        if not discovered[p]:
            process_edge(v, p)
            dfs(g, p)
        elif not processed[p]:
            process_edge(v, p) # cycle found
        if is_finished(g, v, p):
            return
        p = p.next

    process_vertex_after(v)
    processed[v] = True

```

Common applications of DFS include

- Finding cycles: Cycles can be found with DFS if an already discovered (but not yet processed) vertex is encountered. A discovered but not processed vertex, v0, entails that v0 has begun its DFS but not yet completed it. In fact, it is in the process of that search as another vertex, v1, is found that has an edge leading back to it. Hence, this must be a cycle.

- Articulation point finding: An articulation point is a vertex with the property that if removed, will disconnect a connected component (a graph with connectivity of 1).

DFS can be used to identify articulation points as it partitions edges into tree edges (a connection to a previously undiscovered node), and back edges (a connection to an ancestor). Any vertices that lie between an ancestor connected by a back edge and a chain of tree edges cannot be an articulation point because the back edge guarentees connectivity if they are removed.

A vertex that does not have this property and is not a leaf is then an articulation point.

### Maze traversal (DFS)

Given a 2D array of black and white entries representing a mazy with designated entrance and exit points, find a path from the entrance to the exit, if one exists.

The question does not ask for shortest possible path, so we can use DFS since the implementation is simpler.

Let's see in this in action with a maze traversal.

```python
import collections

Coordinate = collections.namedtuple('Coordinate', ('x', 'y'))

def search_maze(maze, s, t):

    def search_maze_helper(curr):
        if not (0 <= curr.x < len(maze) and 0 <= curr.y < len(maze[curr.x]) and maze[curr.x][curr.y] == 0):
            return False

        maze[curr.x][curr.y] = 1
        path.append((curr.x, curr.y))

        if t.x == curr.x and t.y == curr.y:
            return True

        if any(
            map(search_maze_helper,
                map(Coordinate, (curr.x - 1, curr.x + 1, curr.x, curr.x),
                    (curr.y, curr.y, curr.y - 1, curr.y + 1)))):
            return True

        del path[-1]
        return False

    path = []
    search_maze_helper(s)
    return len(path) > 0

maze = [
    [0, 1, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0]
]

assert search_maze(maze, Coordinate(0, 0), Coordinate(1, 3)) == False
assert search_maze(maze, Coordinate(1, 1), Coordinate(1, 3)) == True
```

#### Flip Color (BFS)

Implement a routine that takes an nxm Boolean array A together with an entry (x, y) and flips the color of the region associated with (x, y). i.e. all elements reachable from x, y that have the same color as it will flip their color.

BFS is natural to use for region finding problems since it explores neighbors in order of distance from starting point. Hence it grows outward from the starting point until all reachable points are found.

BFS follows this pattern:

```python
def bfs(graph, s, t):
    queue = [s] # initialize a queue of vertices to explore
    while len(queue) > 0:
        coords = queue.pop()
        # do something with coords
        # add all valid neighbors of coords to next queue
        next_queue = map(get_valid_neighbors, coords)
        if len(next_queue) > 0:
            queue += next_queue
```

See test cases for examples.

```python
def flip_color(maze, x, y):
    color = maze[x][y]
    queue = collections.deque([(x, y)])
    maze[x][y] = int(not color)

    while queue:
        x, y = queue.popleft()
        for next_x, next_y in ((x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y)):
            if 0 <= next_x < len(maze) and 0 <= next_y < len(maze[next_x]) and maze[next_x][next_y] == color:
                maze[next_x][next_y] = int(not color)
                queue.append((next_x, next_y))

    return maze


maze = [
    [0, 1, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0]
]

assert flip_color(maze, 0, 0) = [
    [1, 1, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0]
]

assert flip_color(maze, 1, 2) = [
    [0, 1, 1, 1],
    [1, 1, 1, 1],
    [0, 1, 1, 1]
]

assert flip_color(maze, 0, 1) = [
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0]
]
```

Let's do the DFS implementation for fun.

```python
def flip_color_dfs(maze, x, y):
    color = maze[x][y]
    maze[x][y] = int(not color)

    for next_x, next_y in ((x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y)):
        if 0 <= next_x < len(maze) and 0 <= next_y < len(maze[next_x]) and maze[next_x][next_y] == color:
            flip_color_dfs(maze, next_x, next_y)

    return maze
```

#### Compute enclosed region

Let A be a 2D binary array. Write a program that takes A, and replaces all 0's that cannot reach the boundary with a 1. i.e. if there does not exist a path from an enclosed 0 to the outer edges of the array, it has to be flipped.

See test cases for examples below.

Computing whether or not all enclosed 0 entries can reach the boundary may be a bit difficult, hence lets consider the converse.

How do we find the 0's that can reach the boundary? These are the elements that won't be flipped.

We can find these 0's by traversing all reachable 0's from boundary 0's.

Then, we can mark these with some flag. These marked entries will become the only 0 entries in the the grid.

Finally, we go back and turn all entries that weren't marked into 1 and the marked ones into 0.

```python
def compute_enclosed_regions(maze):
    n, m = len(maze), len(maze[0])
    queue = collections.deque(
        [(i, j) for k in range(n) for i, j in ((k, 0), (k, m - 1))] +
        [(i, j) for k in range(m) for i, j in ((0, k), (n - 1, k))]
    )

    while queue:
        x, y = queue.popleft()
        if maze[x][y] == 0:
            maze[x][y] = 'T'
            for next_x, next_y in (x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y):
                if 0 <= next_x < len(maze) and 0 <= next_y < len(maze[next_x]) and maze[next_x][next_y] == 0:
                    queue.append((next_x, next_y))


    return [[0 if c == 'T' else 1 for c in row] for row in maze]

assert compute_enclosed_regions([
    [0, 1, 1, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0]
]) == [
    [0, 1, 1, 1],
    [1, 1, 1, 1],
    [0, 1, 1, 0]
]

assert compute_enclosed_regions([
    [0, 1, 1, 1],
    [1, 0, 1, 1],
    [0, 1, 1, 0]
]) == [
    [0, 1, 1, 1],
    [1, 1, 1, 1],
    [0, 1, 1, 0]
]

assert compute_enclosed_regions([
    [0, 1, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0]
]) == [
    [0, 1, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0]
]
```

#### Cycle detection with DFS

DFS can be used to detect cycles in graphs because it maintains a color for each vertex. All vertices start out white. When a vertex is first discovered it's colored grey. After a vertex has been processed it becomes black.

The algorithm is straightforward DFS while check for vertex color.

```python
class GraphVertex:

    WHITE, GREY, BLACK = range(3)

    def __init__(self, edges = []):
        self.color = GraphVertex.WHITE
        self.edges = edges

def has_cycle(graph):

    def has_cycle_helper(curr):
        if curr.color == GraphVertex.GREY:
            return True

        curr.color = GraphVertex.GREY

        if any(next.color != GraphVertex.BLACK and has_cycle_helper(next) for next in curr.edges):
            return True

        curr.color = GraphVertex.BLACK
        return False


    return any(node.color == GraphVertex.WHITE and has_cycle_helper(node) for node in graph)


x = GraphVertex()
y = GraphVertex([x])
x.edges = [y]
z = GraphVertex()

graph_1 = [
    GraphVertex([GraphVertex([GraphVertex()]), z]),
    GraphVertex([z]),
    GraphVertex()
]

assert has_cycle(graph_1) == False

x = GraphVertex()
graph_2 = [
    GraphVertex([GraphVertex([GraphVertex([x, y])]), z]),
    GraphVertex([x]),
    GraphVertex()
]

assert has_cycle(graph_2) == True
```

#### Clone a graph

Given a vertex, create a copy of the graph on the vertices reachable from this vertex. Vertex = { Label, Edges }.

Any standard graph traversal algorithm will work here. As we traverse, we'll add new vertices / edges to our cloned graph. We'll use a hashmap of vertices to do this.

```python

```

#### Transform one string into another

Given a dictionary D, and two string s and t, write a program to determine if s produces t. s can produce t if there exists a sequence of words in the dictionary that consecutively differ in one letter starting at s and ending at t.

Modeling this as a graph problem we can have words be vertices and edges are formed if a word is one away from another word.

```python
import collections, string

def transform_string(dictionary, s, t):
    StringWithDistances = collections.namedtuple('StringWithDistances',\
                                                ('candidate_string', 'distance'))
    queue = collections.deque([StringWithDistances(s, 0)])
    dictionary.remove(s)

    while queue:
        word = queue.popleft()
        if word.candidate_string == t:
            return word.distance
        for i in range(len(word.candidate_string)):
            for c in string.ascii_lowercase:
                candidate = word.candidate_string[:i] + c + word.candidate_string[i+1:]
                if candidate in dictionary:
                    queue.append(StringWithDistances(candidate, word.distance + 1))
                    dictionary.remove(candidate)
    return -1

dictionary = set(['bat', 'cot', 'dog', 'dag', 'dot', 'cat', 'mad', 'sad'])
assert transform_string(dictionary, 'cat', 'dog') > 0
assert transform_string(dictionary, 'cat', 'mad') == -1
```

### Shortest Path

It's imperative to know the algorithm to obtain the shortest path from any given vertex to another. It's a question that's commonly asked in interviews that you should knock out of the park.

Namely, Dijkstra's Algorithm

#### Dikjstra's Shortest Path Algorithm

[The Wikipedia page](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm) has all the info you'd need plus this great illustration.

![Dijjkstra's](https://upload.wikimedia.org/wikipedia/commons/5/57/Dijkstra_Animation.gif)

This is essentially BFS accounting for weights in graphs. Instead of picking the first neighbor we find in BFS, we instead choose the neighbor with the lowest distance from the start vertex. We continue picking the least weighted path until we reach the end vertex.

This is broken down into the following steps:

Given G, s, t

1. Create an array `dist` where dist[u] denotes the distance from s to u
2. Initialize the array with all vertices being infinity and dist[s] to 0
3. Use BFS to traverse the graph, pick the vertex that is minimum distance from s when popping from the queue
4. For each unvisited neighbors from the current node, calculate the distance to that node (if there is already a distance, choose smaller one)
5. Mark the current node as unvisited
6. If the destination node has been reached or if the smallest distance among unvisited nodes is inifinity, then stop

This is a modified version of [Maria Boldyreva's solution on dev.to](https://dev.to/mxl/dijkstras-algorithm-in-python-algorithms-for-beginners-dkc)

```python
# Dijkstra's
import collections, functools, heapq

Edge = collections.namedtuple('Edge', ('start', 'end', 'cost'))

class Graph:
    def __init__(self, edges=[]):
        self.edges = [Edge(*edge) for edge in edges]

    @property
    def vertices(self):
        return set(
            functools.reduce(lambda acc, e: acc + [e.start, e.end], self.edges, [])
        )

    @property
    def neighbours(self):
        neighbours = {vertex: set() for vertex in self.vertices}
        for edge in self.edges:
            neighbours[edge.start].add((edge.end, edge.cost))

        return neighbours

    def dijkstra(self, s, t):
        vertices = self.vertices.copy()
        neighbours = self.neighbours.copy()
        distances = {vertex: float('inf') for vertex in vertices}
        prev = {
            vertex: None for vertex in vertices
        }

        distances[s] = 0
        min_queue = [(distances[vertex], vertex) for vertex in distances.keys()]
        heapq.heapify(min_queue)
        while min_queue:
            curr_dist, curr_vertex = heapq.heappop(min_queue)
            if curr_dist == float('inf'):
                break
            for neighbour, cost in neighbours[curr_vertex]:
                alt_cost = curr_dist + cost
                if alt_cost < distances[neighbour]:
                    distances[neighbour] = alt_cost
                    prev[neighbour] = curr_vertex

            # Rebuild heap based on new distances
            unvisited_vertices = []
            while min_queue:
                _, v = heapq.heappop(min_queue)
                unvisited_vertices.append((distances[v], v))

            heapq.heapify(unvisited_vertices)
            min_queue = unvisited_vertices

        path, current_vertex = collections.deque(), t
        while prev[current_vertex] is not None:
            path.appendleft(current_vertex)
            current_vertex = prev[current_vertex]
        if path:
            path.appendleft(current_vertex)
        return path

graph = Graph([
    ("a", "b", 7),  ("a", "c", 9),  ("a", "f", 14), ("b", "c", 10),
    ("b", "d", 15), ("c", "d", 11), ("c", "f", 2),  ("d", "e", 6),
    ("e", "f", 9), ("y", "z", 1)])
assert graph.dijkstra("a", "e") == collections.deque(['a', 'c', 'd', 'e'])
assert graph.dijkstra("a", "y") == collections.deque([])
```

### Minimum Spanning Tree

### Matching

### Maximum Flow

# Data Structures

Data structures can be split into two categories:

1. **Contiguously-allocated** data structures are a single block of memory used for arrays, matrices, heaps and hash tables.

2. **Linked** data structures are composed of distinct chunks of memory

## Arrays

Arrays are contiguous blocks of memory representing a list of elements.

A few tips from Elements of Programming Interviews that are notable:

> When dealing with integers encoded by an array consider processing the digits from the back of the array so the least-significant digit is the first entry.
> When operating on 2D arrays, use parallel logic for rows and columns

We'll explore problems concerning modifying arrays in-place such that some property is satisfied, traversing and multi-dimensional datasets.

**Complexities**

| Routine | Dynamic Array | Sorted Dynamic Array |
| :------ | :-----------: | -------------------: |
| select  |     O(1)      |                 O(1) |
| search  |     O(n)      |              O(logn) |
| insert  |     O(1)      |                 O(n) |
| remove  |     O(1)      |                 O(n) |

### 3-way partition

Also known as the dutch national flag problem from EPI page 44.

Given an array A and an index i, rearrange the elements such that all elements less than A[i] appear first, followed by elements equal to the pivot, followed by elements greater than the pivot.

The trivial solution is to form the 3 lists and combine them, but let's try to avoid O(n) space complexity by rearring the list in place.

We can do this by maintaing four subarrays: bottom (elements < pivot), middle (elements == pivot), unclassified, and top (elements > pivot). Initially all elements are unclassified. We iterate through elements in unclassified, and move elements into one of bottom, middle, and top groups according to the relative order between the incoming unclassified element and the pivot. (epi pg. 44)

Code:

```python
def three_way_partition(arr, i):
    pivot_value = arr[i]
    smaller, equal, larger = 0, 0, len(arr)
    while equal < larger:
        if arr[equal] < pivot:
            arr[smaller], arr[equal] = arr[equal], arr[smaller]
            smaller, equal = smaller + 1, equal + 1
        elif arr[equal] == pivot:
            equal += 1
        else:
            larger -= 1
            arr[equal], arr[larger] = arr[larger], arr[equal]
```

### Compute max profit from one buy and sell

**Source:** Elements of Programming Interviews 5.6

**Problem Statement**
Given an input of prices, compute the maximum profit you can get by buying and selling at different price points.

**Input / Output**
[6, 5, 3, 10, 0] => 7

**Brute Force Solution**

```python
def max_profit_brute_force(prices):
    max_profit = 0
    for i in range(0, len(prices)):
        for j in range(i + 1, len(prices)):
            if prices[j] - prices[i] > max_profit:
                max_profit = prices[j] - prices[i]
    return max_profit
```

Simple enough...but can we do better than `O(n^2)`? We can go through our list of potential algorithm choices and two stand out.

**Divide and Conquer**

Divide-and-Conquer could work by splitting the prices in half and determining the best result for each half and combining those results. There are a few edge cases to consider here, such as, what if the optimal buy is in a different half than the optimum sell? The following solution implements this approach...

```python
def split(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]

def max_profit_divide_and_conquer(prices):
    if len(prices) == 0 or len(prices) == 1:
        return 0
    A,B = split(prices)
    return max(
        max_profit_divide_and_conquer(A),
        max_profit_divide_and_conquer(B),
        max(B) - min(A),
        0
     )
```

Worst-case time: `O(nlogn)`

**Greedy**

The Greedy approach actually ends up providing the best time complexity for us. The insight that leads to the key part of this approach can be derived from the Divide and Conquer algorithm. We can see that the maximum profit can be made by selling on any particular day and buying on a specific minimum day that occurred previously. Hence, each element must consider the minimum before it to see if it's optimal, and this can be kept track of as we iterate.

```python
def max_profit_greedy(prices):
    min_price_so_far = float('inf')
    max_profit = 0.0
    for price in prices:
        max_profit_sell_today = price - min_price_so_far
        max_profit = max(max_profit, max_profit_sell_today)
        min_price_so_far = min(min_price_so_far, price)
    return max_profit
```

### Spiral Matrix

Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order.

### Merge meeting times

Given a set of intervals, condense the list so overlapping intervals are merged.

For example:
[(0, 1), (3, 5), (4, 8), (10, 12), (9, 10)] => [(0, 1), (3, 8), (9, 12)]

We first sort our list of intervals by start time so potentially mergeable candidates are adjacent.

Then, we iterate through the list, keeping track of the last merged interval to see if we can continually condense it further or add a new one.

Solution:

```python
def merge_meeting_times(times):
    sorted_times = sorted(times)
    intervals = [sorted_times[0]]

    for curr_start, curr_end in sorted_times[1:]:
        last_start, last_end = intervals[-1]

        if curr_start <= last_end:
            intervals[-1] = last_start, max(curr_end, last_end)
        else:
            intervals.append((curr_start, curr_end))


    return intervals

## tests
test(merge_meeting_times([(0, 1), (3, 5), (4, 8), (10, 12), (9, 10)]), [(0, 1), (3, 8), (9, 12)])
test(merge_meeting_times([(1, 5), (2, 3)]), [(1, 5)])
test(merge_meeting_times([(1, 10), (2, 6), (3, 5), (7, 9)]), [(1, 10)])
```

### Two sum

Given an array of integers A, and an integer k, return the indices of two numbers that add up to k

There are a few approaches we can take here.

1. Use a hashmap to see if the sum has been found while iterating through the array
2. Sort the array and run binary search on each element k - arr[i]
3. Use the front / back sliding window method by keeping two pointers at start and end and close the window to check all possible options.
   This can be done because we can positively say that if arr[left] + arr[right] > k, then increasing left will not move us closer to the target. Vice versa.

```python
def two_sum(arr, k):
    if not arr:
        return None

    idx_map = {}

    for i in range(len(arr)):
        target = (arr[i] - k)*-1
        if target in idx_map:
            return [idx_map[target], i]
        idx_map[arr[i]] = i

    return None
```

```python
def sorted_two_sum_1(arr, k):
    if not arr:
        return None

    def bin_search(arr, k):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == k:
                return mid
            if arr[mid] > k:
                right = mid - 1
            else:
                left = mid + 1

        return -1

    for i in range(len(arr)):
        idx = bin_search(arr, k - arr[i])
        if idx > -1:
            return [i, idx]

    return None
```

```python
def sorted_two_sum_2(arr, k):
    if not arr:
        return None

    left, right = 0, len(arr) - 1
    while left < right:
        curr_sum = arr[left] + arr[right]
        if curr_sum == k:
            return [left, right]
        if curr_sum > k:
            right -= 1
        else:
            left += 1

    return None
```

### Delete duplicates in array

Delete repeated elements from a sorted array.

```python
def delete_duplicates(A: List[int]) -> int:
     if not A:
          return 0

     write_index = 1
     for i in range(1, len(A)):
          if A[write_index - 1] != A[i]:
               A[write_index] = A[i]
               write_index += 1

     return write_index
```

## Strings

### Compute Valid IP's

Given a string of digits, determine all the possible ip addresses that can be made out of it (if any).

An IP address is considered valid if it's the form `xxx.xxx.xxx.xxx`, where `0 <= xxx <= 255`

**Input**
`"19216811'`

**Possible outputs**
`["192.168.1.1", "19.216.8.11", ... 7 more]`

The solution that EPI goes with is very straight forward. Find the first part and determine it's validity. If the first part is valid, find the second part and determine its validity and so on until we find parts that are all valid and add it to our solution set.

The insight I gleaned from this solution is to apply Occam's Razor when dealing with seemingly simple problems. Go with the approach that you would logically use to solve this by hand. Iterate and optimize on top of that if necessary.

**Solution (EPI, pg. 52)**

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

### Reverse a list of characters in place

This is the same problem as reversing any list in place.

```python
def reverse_list(chars):
    for i in range(len(chars) // 2):
        chars[i], chars[~i] = chars[~i], chars[i]
    return chars

## tests
test(reverse_list("a b c d".split(" ")), "d c b a".split(" "))
test(reverse_list("a b c d e".split(" ")), "e d c b a".split(" "))
```

### String to int

We'll go with the [leetcode description](https://leetcode.com/problems/string-to-integer-atoi/).

The function first discards as many whitespace characters as necessary until the first non-whitespace character is found. Then, starting from this character, takes an optional initial plus or minus sign followed by as many numerical digits as possible, and interprets them as a numerical value.

Valid inputs:

`"42" => 42`
`" -42" => -42`

```python
def stoi(self, str: str) -> int:
    index = 0
    while index < len(str) - 1 and str[index] == ' ':
        index += 1

    is_neg = str[index] == '-'
    num = functools.reduce(lambda running_sum, c: running_sum * 10 + string.digits.index(c), str[index + int(is_neg):], 0) * (-1 if is_neg else 1)

    return num
```

### Palindromic Permutation

Given a string determine if any permutation of that string is a palindrome. You can disregard whitespace.

Example:

`"acta tac"` => `True` (`"taco cat"`)

The definition of a palindrome can be reduced to a string with at most one character that has an odd frequency of occurences. So if all but one letters are even, we can form a palindrome. This leads to the following algorithm:

```python
import functools
import collections

def palindromic_permutation(str):
    letters = collections.defaultdict(lambda: 0)
    for letter in str:
        if letter != ' ':
            letters[letter] += 1

    return functools.reduce(lambda odd_count, a: odd_count + 1 if letters[a] % 2 != 0 else odd_count, str, 0) <= 1

assert palindromic_permutation('acto tac') == True
assert palindromic_permutation('acbtc') == False
```

### Replace and remove

Source: EPI 6.4, page 76

Consider the following two rules that are to be applied to an array of characters:

1. Replace each 'a' by two 'd's
2. Delete each entry containing 'b'

The array will always be big enough to contain the final string.

Examples:

`[a, b, a, c, _] => [d, d, d, d, c]`
`[a, c, d, b, b, c, a] => [d, d, c, d, c, d, d]`

The solution should be in-place in O(n) time. This implies we can't shift the characters over to make room for a's or shift back for b's because that would entail O(n^2).

One problem solving technique leads to an algorithm here: solve an easier version of the problem first.

If there are no a's, we can implement the function in-place with one iteration by skipping 'b's and copying the other characters.

If there are no b's, we first compute the final length of the resulting string and then write the result character by character starting from the last character working backwards.

```python
def replace_and_remove(size, s):
    write_idx, a_count = 0, 0
    # Remove b's and count the number of a's
    for i in range(size):
        if s[i] != 'b':
            s[write_idx] = s[i]
            write_idx += 1
        if s[i] == 'a':
            a_count += 1

    curr_idx = write_idx - 1
    write_idx += a_count - 1
    final_size = write_idx + 1
    # Replace a's with dd's starting from the end
    while curr_idx >= 0:
        if s[curr_idx] == 'a':
            s[write_idx - 1: write_idx + 1] = 'dd'
            write_idx -= 2
        else:
            s[write_idx] = s[curr_idx]
            write_idx -= 1
        curr_idx -= 1

    return s
```

### Reverse words in a sentence

Source: EPI 6.6

Given an array of characters representign a sentence, reverse the order of the words (space delimeted) in the sentence.

Example:

Input: `['w', 'h', a', 't', ' ', 'i', 's', ' ', 'u', 'p']`

Output: `['u', 'p', ' ', 'i', 's', ' ', 'w', 'h', a', 't']`

We'll approach this problem with 2 passes. In the first pass, we'll reverse the entire array. In the following pass we'll reverse each word.

```python
def reverse_words(words):
    def reverse_range(chars, start, end):
        while start < end:
            chars[start], chars[end] = chars[end], chars[start]
            start, end = start + 1, end - 1

    reverse_range(words, 0, len(words) - 1)
    start = 0

    while True:
        finish = start
        while finish < len(words) and words[finish] != ' ':
            finish += 1

        if finish == len(words):
            break

        reverse_range(words, start, finish - 1)
        start = finish + 1

    reverse_range(words, start, len(words) - 1)

    return words

assert reverse_words(['w', 'h', 'a', 't', ' ', 'i', 's', ' ', 'u', 'p']) == ['u', 'p', ' ', 'i', 's', ' ', 'w', 'h', 'a', 't']
```

### One Away

Source: Crack the Coding Interview, page 199

There are 3 types of edits. Replace, insert and remove. Determine if two strings are one edit away from being equal.

Examples:

```python
assert one_away('abc', 'abd') == True
assert one_away('ab', 'abd') == True
assert one_away('ad', 'abd') == False
assert one_away('abc', 'abde') == False
```

It helps to analyze each case separately.

1. Replace: There must be at most 1 element that is different and length must be the same.
2. Insert: There must be 1 character difference in the strings and the existing characters must match.
3. Remove: Opposite of insert.

Here's the algorithm:

```python
def one_away(str1, str2):
    n1 = len(str1)
    n2 = len(str2)
    if abs(n1 - n2) > 1:
        return False
    elif n1 == n2:
        # replace
        found_diff = False
        for i in range(n1):
            if str1[i] != str2[i] and not found_diff:
                found_diff = True
            elif str1[i] != str2[i]:
                return False
    else:
        # remove or insert
        i, j = 0, 0
        found_diff = False
        while i < n1 and j < n2:
            if str1[i] != str2[j]:
                if found_diff:
                    return False
                found_diff = True
                if n1 < n2:
                    j += 1
                else:
                    i += 1
            else:
                i += 1
                j += 1

    return True
```

## Tries

### Find words

Source: https://leetcode.com/problems/word-search-ii/

Given a 2D board and a list of words from the dictionary, find all words in the board.

Each word must be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once in a word.

Example:

```
Input:
board = [
  ['o','a','a','n'],
  ['e','t','a','e'],
  ['i','h','k','r'],
  ['i','f','l','v']
]
words = ["oath","pea","eat","rain"]

Output: ["eat","oath"]
```

The problem here is essentially a graph search on all vertices, looking for any word (treated as a path).

Simple enough, but we have to come up with some way to efficiently check if a current letter is in the search path. We can do this by adding all words to a trie.

A trie simplifies our path checking logic because we just have to check if the current node contains the current letter.

If it does, then we:

1. Update the path to the next node
2. Update the current word
3. Mark the current point in the board as visited (just an empty string '')
4. Check if we've reached the end of a word, if so, store it.
5. Check all valid neighbours for the remaining path
6. Unmark the current point

```python
def find_words(board, words):
    def search_board(i, j, node, path):
        if board[i][j] in node:
            board[i][j], letter = '', board[i][j]
            node, path = node[letter], path + letter

            if '$' in node:
                result.add(path)

            for x,y in (i + 1, j), (i - 1, j), (i, j - 1), (i, j + 1):
                if 0 <= x < n and 0 <= y < m:
                    search_board(x, y, node, path)

            board[i][j] = letter

    trie = Trie()
    for word in words:
        trie.add_word(word)

    result = set()
    n = len(board)
    m = len(board) and len(board[0])

    [search_board(i, j, trie.root_node, '') for i in range(n) for j in range(m)]

    return list(result)
```

Trie implementation

```python
class Trie:
    def __init__(self):
        self.root_node = {}

    def add_word(self, word):
        is_new_word = False
        current_node = self.root_node

        for char in word:
            if char not in current_node:
                is_new_word = True
                current_node[char] = {}

            current_node = current_node[char]

        if "$" not in current_node:
            is_new_word = True
            current_node["$"] = {}

        return is_new_word
```

### Concatenated Words

```python
        # add all words to a trie
        trie = Trie()
        for word in words:
            trie.add_word(word)

        # for each word
            # traverse through trie till end of word
            # every time a word is found
                # search rest of word using root trie
                # continue search with current trie
                # increment number of found words
            # once end of word is reached
                # if any non-root search has also ended,
                    # add to result
```

### Longest common prefix

## Stacks & Queues

### Stack with min. API

Design a stack data structure such that you can query the minimum element in O(1) time.

This problem is easily solved with an additional stack. Specifically, we can use another stack to store the current min. for the stack.

```python
class Stack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def pop(self):
        if self.stack[-1] == self.min_stack[-1]:
            self.min_stack.pop()

        self.stack.pop()

    def push(self, x):
        if not self.min_stack or x < self.min_stack[-1]:
            self.min_stack.append(x)

        self.stack.append(x)

    def get_min(self):
        return self.min_stack[-1]
```

### Shortest Equivalent Path

Source: EPI 8.4

Given a pathname (absolute or relative), return the shortest equivalent pathname.

**Examples**

`"/usr/lib/../bin/gcc"` => `"/usr/bin/gcc"`

`"scripts//./../scripts/awkscripts/././"` => `"scripts/awkscripts"`

Since each path is going to be conditionally shortened based on succeeding entries, a stack works well here.

In particular, we can push entries on to a stack if they're actual paths and pop if they're `../` or `//, ./`

We can return the final stack joined with `/` as the shortened pathname.

What remains is just addressing edge cases like maintaining absolute pathname structure, throwing errors when paths are invalid.

**Code**

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

assert shorten_pathnames("../bin/../gc/lol") == "../gc/lol"
assert shorten_pathnames("/user/bin/../gc/lol") == "/user/gc/lol"
assert shorten_pathnames("scripts//./../scripts/awkscripts/./.") == "scripts/awkscripts"
```

### Compute Buildings with Sunset View

You are given a series of buildings that face west. The buildings are in a straight line, and any building which is to the east of the building of equal or greater height cannot view the sunset.

Return the buildings with a sunset view given a list of buildings in east-to-west order.

```python
def sunset_view(buildings):
    stack = []

    for building in buildings:
        while stack and stack[-1] <= building:
            stack.pop()

        stack.append(building)

    return stack


assert sunset_view([2, 1, 3, 4, 5, 4]) == [5, 4]
assert sunset_view([0, 1, 1, 3, 2, 3]) == [3]
```

### Compute binary tree nodes in order of increasing depth

```python
def increasing_depth_binary_tree(tree):
    queue = collections.deque([[tree]])

    results = []

    while queue:
        nodes = queue.popleft()
        results.append([node.data for node in nodes])
        q2 = []
        for node in nodes:
            if node.left:
                q2.append(node.left)
            if node.right:
                q2.append(node.right)

        if q2:
            queue.append(q2)

    return results
```

### Implement a circular queue

Implement a queue with O(1) enqueue and dequeue operations using two additional fields (beginning and end indices). You are given the initial size of the queue in the constructor. Resize the inner array dynamically as needed.

We can solve this problem by keeping track of the head and tail as well as current size of the queue.

For enqueues:

If current size exceeds limit, we must resize
to resize, move all elements in order of head to tail to another array
increase this new array size by some factor
change the pointers of head and tail to reflect new array

Then, we can just add the element to the tail

For dequeues:

We just move the head pointer to the next element in the queue

```python
class CircularQueue:
    SCALE_FACTOR = 2

    def __init__(self, size):
        self.entries = [0] * size
        self.head = self.tail = self.num_elems = 0

    def enqueue(self, elem):
        if self.num_elems == len(self.entries):
            self.entries = self.entries[self.head:] + self.entries[:self.head]
            self.head, self.tail = 0, self.num_elems
            self.entries += [0] * (CircularQueue.SCALE_FACTOR * len(self.entries) - len(self.entries))

        self.entries[self.tail] = elem
        self.tail = (self.tail + 1) % len(self.entries)
        self.num_elems += 1

    def dequeue(self):
        result = self.entries[self.head]
        self.head = (self.head + 1) % len(self.entries)
        self.num_elems -= 1
        return result

    def size():
        return self.num_elems
```

### Implement a queue using stacks

How would you implement a queue using stacks?

This can be done with 2 stacks. Namely, a stack for enqueues and a stack for dequeus.

For enqueues:

We just append to the enqueue stack

For dequeues:

If the dequeue stack is empty, fill it by pushing all elements of the enqueue stack onto dequeue.

Then, dequeue[-1] should be deleted.

```python
class QueueWithStacks:
    def __init__(self):
        self.enq, self.deq = [], []

    def enqueue(self, val):
        self.enq.append(val)

    def get_head(self):
        if not self.deq:
            while self.enq:
                self.deq.append(self.enq.pop())

        return self.deq[-1]

    def dequeue(self):
        result = self.get_head()
        del self.deq[-1]
        return result
```

### Queue With Max

Implement a queue with a O(1) get_max method.

The brute force approach here is to keep track of the current max on enqueue and dequeue. Enqueue is fast, dequeue is slow since you must search through the list to find the next max.

The insight that leads to a better algorithm is that once we add an element to queue, any elements previously added that are less than the current element can never be the max. Hence we can iteratively remove these from consideration.

We can use another queue to store our max candidates.

On enqueue:

While the element is larger than the tail of the max queue, remove tail from max queue

Now append the element to the max queue.

Note that it is less than all preceding elements. Hence, we have that the max queue head contains the max element.

On dequeue:

If the max element is equal to current element to be dequeued. Dequeue max queue.

```python
class QueueWithMax:
    def __init__(self):
        self.queue, self.max_queue = collections.deque([]), collections.deque([])

    def enqueue(self, val):
        self.queue.append(val)

        while self.max_queue and self.max_queue[-1] < val:
            self.max_queue.pop()

        self.max_queue.append(val)

    def dequeue(self):
        result = self.queue.popleft()
        if result == self.max_queue[0]:
            self.max_queue.popleft()

        return result

    def get_max(self):
        return self.max_queue[0]
```

## Binary Search Trees

From EPI (page 211):

> BST's are a workhorse of data structure and can be used to solve almost every data structures problem reasonably efficiently. They can efficiently search for a key as well as find the min and max elements, look for the successor or predecessor of a search key and enumerate the keys in a range in sorted order.

### Is a Binary Tree also a BST?

Given a tree T, determine if it satisfies the BST property.

Three approaches:

1. Traverse the tree, maintaining a lower and upper bound for each recursive call. If the lower and upper are violated then return False else True.
2. Complete an inorder traversal, and see if the nodes are in sorted order. A sorted in order traversal implies the BST property is satisfied.
3. Traverse the tree in depth order (using BFS), maintaining a lower and upper bound and return false if the constraint is violated else True

Approach 1:

```python
def is_bst(tree):
    def is_in_range(tree, lower=float('-inf'), upper=float('inf')):
        if not tree:
            return True
        if not lower <= tree.data <= upper:
            return False
        return is_in_range(tree.left, lower, tree.data) and \
            is_in_range(tree.right, tree.data, upper)

    return is_in_range(tree)

tree_1 = BinaryTreeNode(19, BinaryTreeNode(7,
                                       BinaryTreeNode(3,
                                                      BinaryTreeNode(2),
                                                      BinaryTreeNode(5)),
                                       BinaryTreeNode(11,
                                                      None,
                                                      BinaryTreeNode(17,
                                                                     BinaryTreeNode(13)))),
                        BinaryTreeNode(43,
                                       BinaryTreeNode(23,
                                                      None,
                                                      BinaryTreeNode(37,
                                                                     BinaryTreeNode(29),
                                                                     BinaryTreeNode(41))),
                                       BinaryTreeNode(47,
                                                      None,
                                                      BinaryTreeNode(53))))


tree_2 = BinaryTreeNode(19, BinaryTreeNode(7,
                                       BinaryTreeNode(3,
                                                      BinaryTreeNode(21),
                                                      BinaryTreeNode(5)),
                                       BinaryTreeNode(11,
                                                      None,
                                                      BinaryTreeNode(17,
                                                                     BinaryTreeNode(13)))),
                        BinaryTreeNode(43,
                                       BinaryTreeNode(23,
                                                      None,
                                                      BinaryTreeNode(37,
                                                                     BinaryTreeNode(29),
                                                                     BinaryTreeNode(41))),
                                       BinaryTreeNode(27,
                                                      None,
                                                      BinaryTreeNode(53))))

assert is_bst_2(tree_1) == True
assert is_bst_2(tree_2) == False
```

Approach 2:

```python
def is_bst_2(tree):
    def is_in_range(tree):
        if not tree:
            return True

        left_satisfied = is_in_range(tree.left)

        if tree.data < prev['val']:
            return False

        prev['val'] = tree.data

        return left_satisfied and is_in_range(tree.right)

    prev = {}
    prev['val'] = float('-inf') # use object in order to reference prev.val in helper
    return is_in_range(tree)
```

Approach 3:

```python
import collections

QueueEntry = collections.namedtuple('QueueEntry', ('node', 'lower', 'upper'))

def is_binary_tree_bst(tree):
    bfs_queue = collections.deque([QueueEntry(tree, float('-inf'), float('inf'))])

    while bfs_queue:
        entry = bfs_queue.popleft()
        if entry.node:
            if not entry.lower <= entry.node.data <= entry.upper:
                return False
            bfs_queue.extend(
                (QueueEntry(entry.node.left, entry.lower, entry.node.data),
                 QueueEntry(entry.node.right, entry.node.data, entry.upper)))

    return True

assert is_binary_tree_bst(tree_1) == True
assert is_binary_tree_bst(tree_2) == False
```

This approach has the advantage that we'll return early if the bst property is violated early in depth or if it lies in the right subtree (the other approaches explore the left subtree first).

### First value greater than key in BST

Given a tree T and integer k, return the value that would appear after k in an in order traversal of T.

We could do an in order traversal to find the first greater than k, but this does not take advantage of the BST property and hence does more work than required (O(n)).

Instead, we'll traverse through the tree, keeping a candidate value and updating it when we encounter a value greater than k. We traverse right if k >= curr_node.data else left.

```python
def find_first_greater_than(tree, k):
    def find_greater(tree, candidate=None):
        if not tree:
            return candidate

        if k < tree.data:
            return find_greater(tree.left, tree.data)
        else: # k >= tree.data
            return find_greater(tree.right, candidate)


    return find_greater(tree)

def find_first_greater_than_iterative(tree, k):
    candidate, subtree = None, tree

    while subtree:
        if k < subtree.data:
            candidate, subtree = subtree.data, subtree.left
        else:
            subtree = subtree.right

    return candidate

assert find_first_greater_than(tree_1, 23) == 29
assert find_first_greater_than(tree_1, 13) == 17
assert find_first_greater_than(tree_1, 31) == 37
```

### Find the K largest elements in a BST

Given a BST T, and an integer k return the k largest elements in T in descending order.

We can do a reverse in-order traversal, store the elements we encounter and return when we have k elements.

```python
def find_k_largest_bst(tree, k):
    def reverse_in_order(tree):
        if tree and len(candidates) < k:
            reverse_in_order(tree.right)
            if len(candidates) < k:
                candidates.append(tree.data)
                reverse_in_order(tree.left)


    candidates = []
    reverse_in_order(tree)
    return candidates

assert find_k_largest_bst(tree_1, 5) == [53, 47, 43, 41, 37]
assert find_k_largest_bst(tree_1, 2) == [53, 47]
```

### Compute LCA in a BST

Given a BST T and two nodes s and t. Find the LCA of s and t in T.

In a BST, the LCA of two nodes is the first node that is in between the range of the s and t. This is the last point at which they break into separate subtrees.

Using this fact, we can traverse through the tree and return the first element that matches this condition.

```python
def lca_bst(tree, a, b):
    min_node, max_node = (a, b) if a < b else (b, a)

    def find_lca(tree):
        if not tree:
            return None
        if min_node <= tree.data <= max_node:
            return tree.data
        if tree.data > max_node:
            return find_lca(tree.left)
        return find_lca(tree.right)

    return find_lca(tree)


def lca_bst_iterative(tree, a, b):
    while tree and not (a.data <= tree.data <= b.data):
        while tree.data < a.data:
            tree = tree.right

        while tree.data > b.data:
            tree = tree.left

    return tree

assert lca_bst(tree_1, 31, 53) == 43
assert lca_bst(tree_1, 7, 13) == 7
assert lca_bst(tree_1, 2, 53) == 19
```

### Reconstruct a BST from traversal data

Given a preorder traversal of a BST with unique keys, reconstruct the BST.

Note: A preorder traversal sequence has 1-1 mapping to a BST (inorder does not).

The first element of the sequence always contains the root node. All following elements that are less than the root are in the left subtree, greater elements are in the right subtree.

We can apply this reasoning recursively and end up with this algorithm.

```python
tree_3 = BinaryTreeNode(5,
                        BinaryTreeNode(3, BinaryTreeNode(2, BinaryTreeNode(1)), BinaryTreeNode(4)),
                        BinaryTreeNode(8, BinaryTreeNode(7)))

def bst_from_preorder_traversal(sequence):
    def create_tree(sequence):
        if len(sequence) == 0:
            return None
        return BinaryTreeNode(
            sequence[0],
            create_tree([i for i in sequence if i < sequence[0]]),
            create_tree([i for i in sequence if i > sequence[0]]))

    return create_tree(sequence)

assert is_bst_equal(bst_from_preorder_traversal(preorder(tree_3)), tree_3)
assert is_bst_equal(bst_from_preorder_traversal(preorder(tree_1)), tree_1)
```

However, this approach takes O(n^2) in the worst case (a left-skewed tree) where n is the length of the sequence.

The recurrence relation is T(n) = T(n - 1) + O(n) = O(n^2)

This approach does repeated calculation to determine smaller and larger elements from the root. Instead, we can do this comparison as we're creating the subtree.

We can do this by providing a range of valid values for each subtree. This leads to the following algorithms:

```python
def bst_from_preorder_efficient(sequence):
    def create_tree(lower, upper):
        if root_idx[0] == len(sequence):
            return None

        root = sequence[root_idx[0]]

        if not lower <= root <= upper:
            return None

        root_idx[0] += 1

        left = create_tree(lower, root)
        right = create_tree(root, upper)

        return BinaryTreeNode(root, left, right)

    root_idx = [0]
    return create_tree(float('-inf'), float('inf'))
```

### Closest entries in k sorted arrays

Given k sorted arrays, find the minimum interval containing at least 1 element from each array.

The insight that leads to an algorithm is that we can keep a candidate range starting with the smallest in all arrays and then iteratively check the next possible range by removing the smallest elenent from the range and adding the next smallest element from the same array.

This is because removing the smallest element from a candidate range will shorten the range. We continue to check ranges in this manner until at least one of the arrays runs out of elements.

We use the `bintrees` module to implement a red black tree. This is because we need efficient retrieval of min, max and insertion, deletion for our candidate range. A BST is perfect for this.

```python
import bintrees

def find_closest(sorted_arrays):
    iters = bintrees.RBTree()

    for idx, sorted_array in enumerate(sorted_arrays):
        it = iter(sorted_array)
        first_min = next(it, None)
        if first_min is not None:
            iters.insert((first_min, idx), it)

    min_range = [float('-inf'), float('inf')]
    while True:
        min_value, min_idx = iters.min_key()
        max_value = iters.max_key()[0]

        if min_value - max_value < min_range[1] - min_range[0]:
            min_range = [min_value, max_value]
        it = iters.pop_min()[1] # key, val
        next_min = next(it, None)
        if next_min is None:
            return min_range
        iters.insert((next_min, min_idx), it)
```

### Build a min height BST from a sorted array

Given a sorted array, build a minimum height BST with it's elements.

The minimum height subtree is obtained by recursively taking the median element as root in each subtree. Intuitively this is because this leads to the most balanced tree possible.

```python
def build_bst_from_sorted_array(arr):
    def build_bst(start, end):
        if end <= start:
            return None
        mid = (start + end) // 2
        return BinaryTreeNode(arr[mid], build_bst(start, mid), build_bst(mid + 1, end))

    return build_bst(0, len(arr))
```

### Determine if 3 nodes are totally ordered

Begin search from both nodes until mid is encountered. Continue searching until unfound is encountered.

```python
def is_totally_ordered(tree, middle, node1, node2):
    next_node = middle
    found_mid = False
    found_dec = False

    curr_node1 = node1
    curr_node2 = node2

    while (curr_node1 or curr_node2) and not found_dec:
        if curr_node1:
            if curr_node1.data == next_node.data:
                if not found_mid:
                    next_node = node2
                    found_mid = True
                else:
                    found_dec = True

            curr_node1 = curr_node1.left if curr_node1.data > next_node.data else curr_node1.right

        if curr_node2:
            if curr_node2.data == next_node.data:
                if not found_mid:
                    next_node = node1
                    found_mid = True
                else:
                    found_dec = True

            curr_node2 = curr_node2.left if curr_node2.data > next_node.data else curr_node2.right


    return found_dec

assert is_totally_ordered(tree_1, tree_1.get_node(23), tree_1.get_node(43), tree_1.get_node(37)) == True
assert is_totally_ordered(tree_1, tree_1.get_node(3), tree_1.get_node(7), tree_1.get_node(5)) == True
assert is_totally_ordered(tree_1, tree_1.get_node(43), tree_1.get_node(41), tree_1.get_node(53)) == False
```

## Range search

Given a BST and a range [a, b] return all the elements in the BST that fall into [a, b]

We'll prune the search using the BST property. If the current node's key is less than the minimum,
then the left subtree cannot contain anything in the range, same argument if key is greater than max for right subtree.

Otherwise, the current node is within range

```python
def range_search(tree, lower, upper):
    def range_search_helper(tree):
        if not tree:
            return
        if lower <= tree.data <= upper:
            range_search_helper(tree.left)
            items.append(tree.data)
            range_search_helper(tree.right)
        elif upper < tree.data:
            range_search_helper(tree.left)
        else:
            range_search_helper(tree.right)

    items = []
    range_search_helper(tree)
    return items

assert range_search(tree_1, 16, 31) == [17, 19, 23, 29]
assert range_search(tree_1, 4, 20) == [5, 7, 11, 13, 17, 19]
```

### Depth Order Binary Tree Traversal

Compute binary tree nodes in increasing depth order

```
      1
  2       3
4   5   6   7
```

=>

```
[[1], [2, 3], [4, 5, 6, 7]]
```

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

## Heaps

https://medium.com/@codingfreak/heap-practice-problems-and-interview-questions-b678ff3b694c

### Stream median

**Problem Statement**

Given a continuous stream of numbers, return the running median at every input.

**Intuition**

In order to avoid a full blown search every time we add a new number, we have to somehow leverage the result of previous computations.

We can do this by splitting the running numbers into two roughly equal halves. A max heap for the bottom half and min heap for the top.

Then querying for the median becomes a straight forward case of either averaging the max of the bottom and min of top if the halves are equal or returning the priority element in the larger half.

A further simplification can be made to reduce some code complexity by adding all of the numbers to the min heap initially and evicting it's min element into the max heap immediately. And then adding a check to ensure that the length of the max heap is not larger than the min heap.

This ensures that the min heap is always the larger half and hence contains the median when the stream length is odd.

**Solution**

```python
def stream_median(sequence: Iterator[int]) -> List[float]:
    min_heap, max_heap, median = [], [], []

    for num in sequence:
        heapq.heappush(min_heap, -heapq.heappushpop(max_heap, -num))

        if len(max_heap) == len(min_heap) - 1:
            heapq.heappush(max_heap, -heapq.heappop(min_heap))
            median.append(-max_heap[0])
        else:
            median.append((min_heap[0] + -max_heap[0]) / 2)

    return median
```

### K Closest Stars (Amazon Question)

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
```

### Sort an almost sorted array

**Problem statement**

Sort an almost sorted array. An almost sorted array is one in which each element is at most k away from it's position.

**Approach**

Since each element is at most k away from it's actual spot, we can use a min heap with k values containing all of the valid elements for our particular index. If we advance through the array sequentially, popping the min of these values will give the correct element.

The approach we'll take will be very similar to the K closest stars question described above.

Advance through the array, adding nums to a min heap.

After the heap has at least k elems, pop min into our result.

**Solution**

```python
def sort_approximately_sorted_array(sequence: Iterator[int],
                                    k: int) -> List[int]:
    min_heap = []
    ans = []

    for i, num in enumerate(sequence):
        heapq.heappush(min_heap, num)

        if len(min_heap) > k:
            ans.append(heapq.heappop(min_heap))

    while min_heap:
        ans.append(heapq.heappop(min_heap))

    return ans
```

## Hashmaps

### Subarray sum

https://leetcode.com/problems/subarray-sum-equals-k/

```python
def subarray_sum(self, nums: List[int], k: int) -> int:
    num_sums, curr_sum = 0, 0
    sum_map = collections.defaultdict(lambda: 0)
    sum_map[0] += 1
    for i in range(len(nums)):
        curr_sum += nums[i]
        num_sums += sum_map[curr_sum - k]
        sum_map[curr_sum] += 1

    return num_sums
```

# Interview Problems

This section includes harder problems with a detailed explanation of getting to the right answer in an interview environment.

Each problem should include:

1. Clarifying questions to idenfity hidden assumptions
2. Examples (some provided, some created)
3. Test cases (written with mocha) -- this step is skipped
4. Illustrations of thought processes or insights that lead to solutions
5. Discussion of solutions, alternatives, time complexities and trade offs
6. Actual code
7. Discussion of optimizations

## Reorganize string

Source: https://leetcode.com/problems/reorganize-string

Difficulty: Medium

**Problem Description**

Given a string S, check if the letters can be rearranged so that two characters that are adjacent to each other are not the same.

If possible, output any possible result. If not possible, return the empty string.

**Questions**

1. Will differences in case matter here?
   > Ans: Assume everything will be lowercase

**Examples**

```
"aab" => "aba"
"aaa" => ""
"aaab" => ""
"aaabb" => "ababa"
```

**Insights**

A generalization can be made using the examples above: If the frequency of any character is above ceil(len(S) / 2), it can't be rearranged.

_Approach #1_

Sort the array by character frequency, then interleave the most common characters (top half), with the least common characters (bottom half)

TODO: add time complexity

_Approach #2_

Greedy with heap

TODO: add deets

## Strong Password Checker

Source: https://leetcode.com/problems/strong-password-checker/

Difficulty: Hard

**Problem Description**

A password is considered strong if below conditions are all met:

1. It has at least 6 characters and at most 20 characters.
2. It must contain at least one lowercase letter, at least one uppercase letter, and at least one digit.
3. It must NOT contain three repeating characters in a row ("...aaa..." is weak, but "...aa...a..." is strong, assuming other conditions are met).

Write a function strongPasswordChecker(s), that takes a string s as input, and return the MINIMUM change required to make s a strong password. If s is already strong, return 0.

Insertion, deletion or replace of any one character are all considered as one change.

**Questions**

1. Are there any constraints on the size of the string?

   > Ans: min. length of 1

2. Would 3 repeating character of different casing "aAa" be considered weak?
   > Ans: No

**Examples**

For examples, assume min is 4 and max is 7

```
Below min:
    ""    => "****" => 4 adds = 4
    "a"   => "a***" => 3 adds = 3
    "/_"  => "/***" => 2 adds + 1 replace = 3
    "aba" => "ab**" => 1 add + 1 replace (missing digit or uppercase) = 2
    "a1A" => "a1A*" => 1 add = 1
    "aaa" => "aa**" => 1 add + 1 replace (fixes repeating chars + missing digit / upper) = 2

Above max:
    "12345678" => "12345**X" => 2 replace (1 lower, 1 upper) + 1 delete = 3
    "123456aB" => "12345XaB" => 1 delete = 1
    "111456aB" => "11X456aB" => 1 delete = 1
    "11145678" => "11*456*X" => 2 replace + 1 delete = 3
    "aaaaaaaa" => "aa*aa*aX" => 2
    "aaa11111" => "aaX11*1" => 1 delete (used to break up group of a) + 1 replace (break up group of 1)
```

This is an important example: It indicates we need to selectively use deletes to break up repeating groups IN SOME CASES. In most cases though, edits work better than deletes to replace groups (when a group is larger than 3)

```
In range:
    Repeating chars:
        "aaaaaaa" => "aa*aa*a" => 2 replace = 2
        "aaa1111" => "aa*11*1" => 2 replace = 2
        "aabaaab" => "aabaa**" => 2 replace (1 digit + 1 upper -- takes care of repeating) = 2
        "bbaaa" => "bba**" => 2 replace = 2

    Missing lower, upper, or digit
        "abcD" => "ab*D" = 1 replace = 1
        "abcdeF" => "abcd*F" = 1 replace = 1
```

**Insights**

First off, let's systematically specify our constraints and cases to approach this question because there's immense potential for error due to the edge cases.

Constraints:

A: Length

B: Casing & digits

C: Repeating characters

Cases:

1. Password is in the range of 6 and 20
2. Password is less than 6
3. Password is greater than 20

Case 1)

Since we don't need to do any adds or removes when the password is already in the correct length range, only edits remain. The edits from constraints B & C could be used to satisfy each other so we have to figure out when overlaps occur and how to determine the minimum amount of edits.

Looking at the examples, we can see that any replacement for constraint B could be used to satisfy constraint C

> Example (min = 4, max = 10): `"....." (5 .'s) => "..A.1a" => 3`

> Constraint B requires 3 edits

> Constraint C requires 2 edits

Since they both overlap here we can take the max of constraints B and C

The converse is also true: any replacement for constraint C could be used to satisfy constraint B.

> Example (min = 4, max = 10): `".........A" (9 .'s + 1 A) => "..1..b..*A" => 3`

> Constraint B requires 2 edits

> Constraint C requires 3 edits

Again, the max of both is the answer

Hence we see that the min edits required for this case is `max(B, C)`

The formula for deriving the min edits for constraint C can be observed in these examples

> let L be the length of a repeating sequence. Then min edits for C = `L / 3`

Case 2)

We have to add `min_pass_len - s.length` characters to satisfy constraint A. These characters can be used to satisfy constraints B and C as well.

> Example (min = 6, max = 20): `"aaaGG"` => `"aa1aGG"` (1 add)

> Example (min = 6, max = 20): `"aaaaa"` => `"aa1aaB"` (1 add, 1 replace)

Hence we have `max(A, B, C)`

We need to revise our formula for getting the min edits for C to `Ceil(L / 2) - 1` (because we're using additions instead of replacements)

Case 3)

In order to satisfy constraint A, we need `s.length - max_pass_len` deletes.

The deletes here cannot satisfy any requirements from constraint B since those require replacements or additions.

They can however, satisfy constraint C by breaking up groups of repeating characters. With the caveat that it usually requires more deletes than edits to break up a group of repeating characters.

> Example (min = 2, max = 7): `"aaaaa1BC"` => `"aaaAa1B"` (1 delete, 1 replacement)

Here we used replacements to break up the repeating characters. If we used deletes it would look like this

> Example (min = 2, max = 7): `"aaaaa1BC"` => `"aa1BC"` (3 deletes)

In general, it takes 3 deletes to break up a group of repeating characters with length 5, whereas it would take 1 replacement to do the same thing.

If the group of repeating characters is of length 3, then they take the same amount of edits.

> Example (min = 2, max = 7): `"123aBaaa"` => `"123aBaa"` (1 delete)

In this case, we would prefer to take the delete as it knocks two birds with one stone (A and C).

It takes `L - 2` deletes to break up a repeating sequence of length `L`
It still takes `L / 3` replacements to break up a repeating sequence

We now need to determine how to place our deletes to optimally break up repeating characters

If we have a repeating sequence of length 6, using 1 delete on it would reduce the number of required replacements from 2 to 1.

A sequence of length 5 would require 3 deletes to bring replacements down by 1 (1 -> 0)

A sequence of length 7 would require 2 deletes to bring replacements down by 1 (2 -> 1)

Generalizing...

Let `K = (L - 2) % 3`. It takes `K if K != 0 else 3` deletes to bring down the number of replacements by 1.

Hence we should allocate our deletes to remove as many replacements as possible.

We can do this by sorting the repeating sequences by number of deletes required to bring it down, and take the smallest ones until we run out of deletes. And then calculate the number of replacements required on the remaining sequences.

Therefore, we need to:

- Remove `X = s.length - max_pass_len` characters to satisfy A
- Place these removes strategically to break up as many repeating character sequences as possible
- Replace the remaining repeating sequences (call this `Y`)
- Replace the violations for constraint B (call this `Z`)

Note: B and C still overlap

Hence we have the formula `X + max(Y, Z)`

**Code**

```javascript
const isBetweenAscii = (lower, upper) => {
  const lowerCharCode = lower.charCodeAt(0);
  const upperCharCode = upper.charCodeAt(0);
  return (char) => {
    return (
      lowerCharCode <= char.charCodeAt(0) && char.charCodeAt(0) <= upperCharCode
    );
  };
};

const isDigit = isBetweenAscii("0", "9");

const isUpper = isBetweenAscii("A", "Z");

const isLower = isBetweenAscii("a", "z");

var strongPasswordChecker = function (s, MIN = 6, MAX = 20) {
  const n = s.length;

  const sSplit = s.split("").concat("-1"); // have some padding at the end for our iterator below

  const computeConstraintsBC = (
    repeatingSeqCalc = (val) => Math.floor(val / 3),
    repeatingSeqCallback
  ) => {
    let repeating_seq_len = 1;
    let constraintC = 0;

    let upper = false,
      lower = false,
      digit = false;

    for (let i = 0; i <= s.length; i++) {
      const char = s.charAt(i);

      if (i > 0 && char === s.charAt(i - 1)) {
        repeating_seq_len += 1;
      } else if (i > 0 && repeating_seq_len > 0) {
        // ended a repeating sequence
        if (repeating_seq_len > 2) {
          constraintC += repeatingSeqCalc(repeating_seq_len);
          if (repeatingSeqCallback) repeatingSeqCallback(repeating_seq_len);
        }
        repeating_seq_len = 1;
      }

      if (isDigit(char)) digit = true;
      else if (isUpper(char)) upper = true;
      else if (isLower(char)) lower = true;
    }

    const constraintB = [lower, upper, digit].filter((isValid) => !isValid)
      .length;

    return { constraintB, constraintC };
  };

  // Case 1
  if (n >= MIN && n <= MAX) {
    const { constraintB, constraintC } = computeConstraintsBC();

    return Math.max(constraintB, constraintC);
  }

  // Case 2
  if (n < MIN) {
    const constraintA = MIN - n;

    const { constraintB, constraintC } = computeConstraintsBC(
      (val) => Math.ceil(val / 2) - 1
    );

    return Math.max(constraintA, constraintB, constraintC);
  }

  // Case 3: s.length > MAX
  const constraintA = n - MAX;

  const repeatingSequences = [];

  // calculate how to use deletes to optimally break up repeating chars
  const { constraintB } = computeConstraintsBC(
    () => 0,
    (val) => repeatingSequences.push(val)
  );

  // function to get number of deletes required to bring down total replacements on sequence
  const getNumDelForSeq = (seq) => {
    const num = (seq - 2) % 3;
    return num === 0 ? 3 : num;
  };

  // sort by num deletes required to bring replacements down
  const comparator = (seqA, seqB) =>
    getNumDelForSeq(seqB) - getNumDelForSeq(seqA);

  repeatingSequences.sort(comparator);

  let numDeletes = constraintA;
  let j = repeatingSequences.length - 1;
  while (numDeletes > 0 && j >= 0) {
    if (repeatingSequences[j] <= 2) {
      j -= 1;
      continue;
    }

    let numDeletesRequired = repeatingSequences[j];
    if (numDeletesRequired < numDeletes) {
      numDeletes -= numDeletesRequired;
      repeatingSequences[j] -= numDeletesRequired;
    } else {
      repeatingSequences[j] -= numDeletes;
      break;
    }
  }

  const constraintC = repeatingSequences.reduce((total, seq) => {
    if (seq > 2) total += Math.floor(seq / 3);
    return total;
  }, 0);

  return constraintA + Math.max(constraintB, constraintC);
};
```

And tests

```javascript
describe("Test strongPasswordChecker -- CASE 1", function () {
  [
    [".....", 3],
    [".........A", 3],
    ["aaab", 2],
    ["aaa111", 2],
  ].forEach(([input, output]) =>
    it(`input: ${input}`, () => {
      const actual = strongPasswordChecker(input, 4, 10);
      assert(actual === output, `Expected: ${output}, Actual: ${actual}`);
    })
  );
});

describe("Test strongPasswordChecker -- CASE 2", function () {
  [
    ["aaaGG", 1],
    ["aaaaa", 2],
  ].forEach(([input, output]) =>
    it(`input: ${input}`, () => {
      const actual = strongPasswordChecker(input, 6, 20);
      assert(actual === output, `Expected: ${output}, Actual: ${actual}`);
    })
  );
});

describe("Test strongPasswordChecker -- CASE 3", function () {
  [
    ["aaaaa1BC", 2],
    ["123aBaaa", 1],
    ["aaaaaaaaaa", 5],
    ["11111aaaaaa", 3, 1, 10],
  ].forEach(([input, output, min = 1, max = 7]) =>
    it(`input: ${input}`, () => {
      const actual = strongPasswordChecker(input, min, max);
      assert(actual === output, `Expected: ${output}, Actual: ${actual}`);
    })
  );
});

mocha.run();
```

## License Key Formatting

Source: https://leetcode.com/problems/license-key-formatting/

Difficulty: Easy

**Problem Description**

You are given a license key represented as a string S which consists only alphanumeric character and dashes. The string is separated into N+1 groups by N dashes.

Given a number K, we would want to reformat the strings such that each group contains exactly K characters, except for the first group which could be shorter than K, but still must contain at least one character. Furthermore, there must be a dash inserted between two groups and all lowercase letters should be converted to uppercase.

Given a non-empty string S and a number K, format the string according to the rules described above.

**Questions**

This is pretty straightforward, no further clarification is needed really.

**Examples**

```
S = "5F3Z-2e-9-w", K = 4 => "5F3Z-2E9W"

S = "2-5g-3-J", K = 2 => "2-5G-3J"

S = "1-2-3". K = 3 => "123"

S = "1-2-3". K = 4 => "123"

S = "1-2-3", K = 1 => "1-2-3"
```

**Insights**

The problem is essentially asking us to remove all dashes and group the elements together such that each group has exactly K elements, except the first which can have 1 <= K.

Therefore, the number of groups can be computed as follows:

Let `L = length of S without dashes`,

If `L < K` then `numGroups = 1` and just return S without dashes (in uppercase)

else `numGroups = Math.ceil(L / K)` and the number of elements in the first group is `L % K`. Hence just return these groups joined by a dash.

**Code**

```javascript
const licenseKeyFormatting = (S, K) => {
  const s = S.toUpperCase().split("-").join("");

  if (s.length < K) return s;

  const numGroups = Math.ceil(s.length / K);
  const firstGroupSize = s.length % K;

  const ans = [];

  if (firstGroupSize !== 0) {
    ans.push(s.slice(0, firstGroupSize));
  }

  for (let i = firstGroupSize; i < s.length; i = i + K) {
    ans.push(s.slice(i, i + K));
  }

  return ans.join("-");
};
```

## Longest Absolute File Path

Source: https://leetcode.com/problems/longest-absolute-file-path/

Difficulty: Medium

**Problem Description**

Suppose we have the file system represented in the following picture:

![File directory](https://assets.leetcode.com/uploads/2020/08/28/mdir.jpg)

We will represent the file system as a string where "\n\t" mean a subdirectory of the main directory, "\n\t\t" means a subdirectory of the subdirectory of the main directory and so on. Each folder will be represented as a string of letters and/or digits. Each file will be in the form "s1.s2" where s1 and s2 are strings of letters and/or digits.

For example, the file system above is represented as

> `"dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext"`.

Given a string input representing the file system in the explained format, return the length of the longest absolute path to a file in the abstracted file system. If there is no file in the system, return 0.

**Examples**

![Longest abs file path example 1](https://assets.leetcode.com/uploads/2020/08/28/dir1.jpg)

```
"dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext" => 20
```

Explanation: We have only one file and its path is `"dir/subdir2/file.ext"` of length 20.

The path `"dir/subdir1"` doesn't contain any files.

![Longest abs file path example 2](https://assets.leetcode.com/uploads/2020/08/28/dir1.jpg)

```
"dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext" => 32
```

Explanation: We have two files: `"dir/subdir1/file1.ext"` of length 21 `"dir/subdir2/subsubdir2/file2.ext"` of length 32.

We return 32 since it is the longest path to a file.

```
"a" => 0
```

Explanation: We do not have any files

**Insights**

The keen reader will notice that the input string is a tree given in depth first format. The problem should be reducible to some graph traversal algorithm but the main hurdle is parsing the input into a usable format.

A few questions initially come to mind...

1. Should we parse it before traversing?
2. Which traversal algorithm is ideal here?

My intuition is that we can traverse and parse at the same time. That is, if we choose to do a depth first search (since it follows the format already given). I don't think BFS would be suitable for this question because we're not looking for the shortest path and it doesn't match the given format.

How would a DFS + parse as we go strategy look like?

At each step in the DFS, we need the current node, the length of the path so far and the node's children.

Assuming we have the current node and path so far, how can we obtain the children?

If the current node is a leaf, it has no children. Hence calculate the length of the current path and update the answer accordingly.

Otherwise:

If we're at an arbitrary depth `i`, the children of our current node would be specified by `\n` followed by `i + 1` `\t`'s

For example, the children of the root directory are specified by `\n\t{directory or file name}`

We can iterate through our current location in the string and look for substrings matching that specification. As soon as we encounter one, we recurse into that node.

After the recursion for that node is complete, we continue looking for matching directories.

One thing to note here is that we'll have to update our current position in the string after the node we recursed on returns. This is to ensure we don't look through the grandchildren.

Hence each node should return the position of the character it ended on.

We stop if we encounter `\n` followed by <`i` `\t`'s OR if we reach the end of the string.

Due to the necessity of keeping track of the current depth and the longest file path, we should also keep track of of the depth and the length of the current path (without `\t` or `\n`) joined back `/`.

Using example 2, we should traverse the following path:

```
dir start=0, end=72
            -> subdir1 start=4, end=37
                        ->  file1.ext start=15 end=24
                        ->  subsubdir1 start=27 end=37
            -> subdir2 start=39, end=72
                        ->  subsubdir2 start=49, end=72
                                           -> file2.ext start=63, end=72
```

Time complexity: `O(n)` Since we're just traversing through the string without repeating computation
Space complexity: `O(h)` where `h` is the depth of the deepest node (stored in the call stack)

**Code**

```javascript
const lengthLongestPath = function (input) {
  const dfs = (start, depth = 0, pathLength = 0) => {
    if (start === input.length - 1) {
      return start;
    }

    const prefix = `\n${"\t".repeat(depth + 1)}`;

    if (!input.startsWith(prefix, start)) {
      return start;
    }

    const nextNode = input.indexOf("\n", start + prefix.length);
    const nextNodeIdx = nextNode !== -1 ? nextNode : input.length;
    const name = input.substring(start + prefix.length, nextNodeIdx);

    const currPathLength = pathLength + name.length + 1;

    if (name.includes(".")) {
      // file!
      ans = Math.max(currPathLength - 1, ans);
      return nextNodeIdx;
    }

    let i = nextNodeIdx;

    while (i < input.length) {
      const end = dfs(i, depth + 1, currPathLength);
      if (i === end) break;
      i = end;
    }

    return i;
  };

  let ans = 0;
  input = "\n" + input;
  dfs(0, -1);
  return ans;
};
```

Test cases

```javascript
describe("Test longest absolute file path", function () {
  [
    [
      "dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext",
      32,
    ],
    ["dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext", 20],
    ["a", 0],
    ["a.txt", 5],
  ].forEach(([input, output]) =>
    it(`input: ${input}`, () => {
      const actual = lengthLongestPath(input);
      assert(actual === output, `Expected: ${output}, Actual: ${actual}`);
    })
  );
});

mocha.run();
```

## X of a Kind in a Deck of Cards

Source: https://leetcode.com/problems/x-of-a-kind-in-a-deck-of-cards/

Difficulty: Easy

**Problem Description**

In a deck of cards, each card has an integer written on it.

Return true if and only if you can choose X >= 2 such that it is possible to split the entire deck into 1 or more groups of cards, where:

1. Each group has exactly X cards.
2. All the cards in each group have the same integer.

Constraints:

1 <= deck.length <= 10^4
0 <= deck[i] < 10^4

**Examples**

Example 1:

```
[1,2,3,4,4,3,2,1] => true
```

Explanation: Possible partition `[1,1],[2,2],[3,3],[4,4]`.

Example 2:

```
[1,1,1,2,2,2,3,3] => false
```

Explanation: No possible partition.

Example 3:

```
[1] => false
```

Explanation: No possible partition.

Example 4:

```
[1,1] => true
```

Explanation: Possible partition `[1,1]`.

Example 4:

```
[1,1,1] => true
```

Explanation: Possible partition `[1,1,1]`.

Example 5:

```
[1,1,2,2,2,2] => true
```

Explanation: Possible partition `[1,1],[2,2],[2,2]`.

**Insights**

Let's clarify the constraints a bit more.

Find an X such that

1. Total of L / X groups
2. All groups have X cards
3. There has to be X groups of matching integers
4. There has to be at least 2 cards in a group

What are the possible value for X?

`2 <= X <= length of the smallest repeating group`

Hence, a brute force strategy could be to find the frequencies of all unique characters, determine the LCD, return true LCD iff >= 2.

> Example: [3,3,3,3,2,2,2,2,2,2,4,4,4,4,4,4,4,4] => [[3,3], [2,2], ...]

Time complexity would be `O(n^2)`

Space complexity is `O(n)` since we're storing an map of frequencies.

**Code**

```javascript
var hasGroupsSizeX = function (deck) {
  const frequencies = {};

  for (let i = 0; i < deck.length; i++) {
    if (!frequencies[deck[i]]) frequencies[deck[i]] = 0;

    frequencies[deck[i]] += 1;
  }

  const frequencyValues = Object.values(frequencies);
  const lowest = frequencyValues.reduce(
    (l, x) => Math.min(l, x),
    Number.POSITIVE_INFINITY
  );

  if (lowest < 2) return false;

  let i = 2;

  while (i <= lowest) {
    const temp = i;
    for (let freq of frequencyValues) {
      if (freq % i !== 0) {
        i += 1;
        break;
      }
    }
    if (i === temp) return true;
  }

  return false;
};
```

## Count Square Submatrices with All Ones

Source: https://leetcode.com/problems/count-square-submatrices-with-all-ones/

Difficulty: Medium

**Problem Description**

Given a m \* n matrix of ones and zeros, return how many square submatrices have all ones.

Constraints:

1 <= arr.length <= 300
1 <= arr[0].length <= 300
0 <= arr[i][j] <= 1

**Examples**

Example 1:

```
[
  [0,1,1,1],
  [1,1,1,1],
  [0,1,1,1]
] => 15
```

Explanation:
There are 10 squares of side 1.
There are 4 squares of side 2.
There is 1 square of side 3.
Total number of squares = 10 + 4 + 1 = 15.

Example 2:

```
[
  [1,0,1],
  [1,1,0],
  [1,1,0]
] => 7
```

Explanation:
There are 6 squares of side 1.  
There is 1 square of side 2.
Total number of squares = 6 + 1 = 7.

**Insights**

At first glance, this seems like a fairly routine connected components question using BFS. However, the code could get quite complex finding inner and outer squares using this method (especially double counting).

A simpler way might be to find all squares starting (top left) at a particular index. This ensures we never double count as a square can only begin in one position.

The brute force way to implement this approach could be to go left to right, top to bottom, and for each '1' in the grid, check all of the elements below and to the right of it to see how many squares begin at this point.

This however, fails to take advantage of previous computations.

For example: If we initially search all the squares beginning at 0,1, then later on in the traversal we find all the squares beginning at 1,2, we have to recompute the values at 1,3, 2,2, 2,4 to determine if it has a larger square.

```
[
    [0, 1, 1, 1],
    [1, 1, 1, 1],
    [0, 1, 1, 1]
]
```

But we know that 1,2 must have a larger square ending at 2,4 because we previously determined this while computing the larger square for 0,1.

A DP approach follow from this intuition.

Let S(i, j) be the total number of squares ending at position i, j

Base cases:

```
M[i,j] == 0 => S(i,j) == 0

M[i,j] == 1 && i == 0 || j == 0  => S(i,j) == 1
```

Recursive relation:

```
if M[i,j] == 1
    S(i,j) = Min(S(i - 1, j), (S(i, j - 1), S(i - 1, j - 1)) + 1
else
    S(i,j) = 0
```

For the previous example, S would look like:

```
[
    [0, 1, 1, 1],
    [1, 1, 2, 2],
    [0, 1, 2, 3]
]
```

total = 15

S for Example 2:

```
[
  [1,0,1],
  [1,1,0],
  [1,2,0]
]
```

total = 7

Another example

```
[
    [1, 1, 0, 1],
    [1, 1, 1, 1],
    [1, 1, 0, 1]
    [1, 1, 1, 1]
]
```

```
[
    [1, 1, 1, 1],
    [1, 2, 2, 0],
    [1, 2, 0, 1],
    [1, 2, 1, 1]
]
```

Bottom up implementation of the DP table works well here

**Code**

```javascript
const countSquares = function (matrix) {
  const n = matrix.length,
    m = matrix[0].length;
  const S = [...Array(n)].map((_) => [...Array(m)].fill(0));

  let ans = 0;

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      if (matrix[i][j] === 0) continue;
      if (i == 0 || j == 0) S[i][j] = 1;
      else {
        S[i][j] = Math.min(S[i - 1][j], S[i][j - 1], S[i - 1][j - 1]) + 1;
      }

      ans += S[i][j];
    }
  }

  return ans;
};
```

## Build Array Where You Can Find The Maximum Exactly K Comparisons

Source: https://leetcode.com/problems/build-array-where-you-can-find-the-maximum-exactly-k-comparisons/

Difficulty: Hard

**Problem Description**

Given three integers n, m and k. Consider the following algorithm to find the maximum element of an array of positive integers:

![build array](https://assets.leetcode.com/uploads/2020/04/02/e.png)

You should build the array arr which has the following properties:

1. arr has exactly n integers.
2. 1 <= arr[i] <= m where (0 <= i < n).
3. After applying the mentioned algorithm to arr, the value search_cost is equal to k.

Return the number of ways to build the array arr under the mentioned conditions. As the answer may grow large, the answer must be computed modulo 10^9 + 7.

**Examples**

Example 1:

```
n = 2, m = 3, k = 1
Output: 6
```

Explanation: The possible arrays are [1, 1], [2, 1], [2, 2], [3, 1], [3, 2] [3, 3]

Example 2:

```
n = 5, m = 2, k = 3
Output: 0
```

Explanation: There are no possible arrays that satisify the mentioned conditions.

Example 3:

```
n = 9, m = 1, k = 1
Output: 1
```

Explanation: The only possible array is [1, 1, 1, 1, 1, 1, 1, 1, 1]

Example 4:

```
n = 50, m = 100, k = 25
Output: 34549172
```

Explanation: Don't forget to compute the answer modulo 1000000007

Example 5:

```
n = 37, m = 17, k = 7
Output: 418930126
```

TODO: come back to this

## Minimum Time Visiting All Points

Source: https://leetcode.com/problems/minimum-time-visiting-all-points/

Difficulty: Easy

**Problem Description**

On a plane there are n points with integer coordinates points[i] = [xi, yi]. Your task is to find the minimum time in seconds to visit all points.

You can move according to the next rules:

- In one second always you can either move vertically, horizontally by one unit or diagonally (it means to move one unit vertically and one unit horizontally in one second).
- You have to visit the points in the same order as they appear in the array.

Constraints:

- `points.length == n`
- `1 <= n <= 100`
- `points[i].length == 2`
- `-1000 <= points[i][0], points[i][1] <= 1000`

**Examples**

![visiting points example 1](https://assets.leetcode.com/uploads/2019/11/14/1626_example_1.PNG)

```
points = [[1,1],[3,4],[-1,0]] => 7
```

Explanation: One optimal path is [1,1] -> [2,2] -> [3,3] -> [3,4] -> [2,3] -> [1,2] -> [0,1] -> [-1,0]  
Time from [1,1] to [3,4] = 3 seconds
Time from [3,4] to [-1,0] = 4 seconds
Total time = 7 seconds

```
[[3,2],[-2,2]] => 5
```

**Insights**

It's very important to look at the specifications of the question here because the second rule makes it a much easier problem to solve. I.e. we have to visit the points in the same order they appear in the array (otherwise this would be a closest pairs problem, which is much more involved).

The example lays out a good strategy to connect all points. Iterate through all consecutive pairs of points, find the minimum distance to between then and add to a total count.

The minimum distance subroutine is the main subproblem to solve.

Looking at the example again, we can see that to travel from point (1,1) to (3,4) it takes 3 moves (2 diagonal, 1 upward. This is the same distance as (3,1) to (3,4). Because the diagonal moves allow us to traverse both x and y and the same time, the larger of the 2 distances will always dominate our minimum required distance. We can cover all the required x moves while also making y moves, but still be left with 1 y move since that distance is larger by 1.

Therefore, min distance from p1 to p2 = max(dx, dy), where dx = |x1 - x2| and dy = |y2 - y1|

This leads to the following linear time algorithm

**Code**

```javascript
const minTimeToVisitAllPoints = function (points) {
  let ans = 0;

  for (let i = 1; i < points.length; i++) {
    const [x1, y1] = points[i - 1];
    const [x2, y2] = points[i];

    const dx = Math.abs(x1 - x2);
    const dy = Math.abs(y1 - y2);

    ans += Math.max(dx, dy);
  }

  return ans;
};
```

## Subtree of another tree

Source: https://leetcode.com/problems/subtree-of-another-tree/

Difficulty: Easy

**Problem Description**

Given two non-empty binary trees s and t, check whether tree t has exactly the same structure and node values with a subtree of s. A subtree of s is a tree consists of a node in s and all of this node's descendants. The tree s could also be considered as a subtree of itself.

**Examples**

Given tree s:

```
     3
    / \
   4   5
  / \
 1   2
```

In order: [1,4,2,3,5]
Preorder: [3,4,1,2,5]

Given tree t:

```
   4
  / \
 1   2
```

In order: [1,4,2]
Preorder: [4,1,2]

Return true, because t has the same structure and node values with a subtree of s.

Using the same t for all subsequent examples

Given tree s:

```
     3
    / \
   4   5
  / \
 1   2
      \
       0
```

In order: [1,4,2,0,3,5]
Preorder: [3,4,1,2,0,5]

Given tree s:

```
     3
    / \
   5   4
      / \
     1   2
    /
   0
```

In order: [5,3,0,1,4,2]
Preorder: [3,5,4,1,0,2]

False

Given tree s:

```
     4
    / \
   1   2
```

In order: [1,4,2]
Preorder: [4,1,2]

True

Given tree s:

```
     2
    /
   4
  /
 1
```

In order: [1,4,2]
Preorder: [2,4,1]

False

**Insights**

Observing the preorder and inorder sequences, we can see that the inorder is not always indicative of a matching subtree but the preorder is in most cases.

Lets focus on the false positives for preorder:

Example 2:
Given tree s:

```
     3
    / \
   4   5
  / \
 1   2
      \
       0
```

Preorder: [3,4,1,2,0,5]

We can solve this by also including null values in the sequence

Preorder: [3,4,1,null,null,2,null,0,5]

Performing a search for a subarray matching the preorder traversal of t within s's preorder traversal (including nulls) gives us the correct answer!

Time Complexity: `O(n*m)`

This leads to the following algorithm

**Code**

```javascript
const isSubtree = function (s, t) {
  const getPreorder = (root) => {
    const stack = [];
    const preorder = [];

    while (stack.length || root) {
      if (!root) {
        preorder.push("null");
      } else {
        while (root) {
          preorder.push(`#${root.val}`);
          stack.push(root);
          root = root.left;
        }
        preorder.push("null");
      }

      if (stack.length) {
        root = stack.pop();
        root = root.right;
      }
    }

    preorder.push("null");

    return preorder.join(" ");
  };

  const seqS = getPreorder(s);
  const seqT = getPreorder(t);

  return seqS.includes(seqT);
};
```

Note: prepending '#' to node values allows us to use string search instead of array matching (avoids the scenario where "2 null null" matches "12 null null")

## The Maze II

Source: https://leetcode.com/problems/the-maze-ii/

Difficulty: Medium

**Problem Description**

There is a ball in a maze with empty spaces and walls. The ball can go through empty spaces by rolling up, down, left or right, but it won't stop rolling until hitting a wall. When the ball stops, it could choose the next direction.

Given the ball's start position, the destination and the maze, find the shortest distance for the ball to stop at the destination. The distance is defined by the number of empty spaces traveled by the ball from the start position (excluded) to the destination (included). If the ball cannot stop at the destination, return -1.

The maze is represented by a binary 2D array. 1 means the wall and 0 means the empty space. You may assume that the borders of the maze are all walls. The start and destination coordinates are represented by row and column indexes.

**Examples**

![Maze 1](https://assets.leetcode.com/uploads/2018/10/12/maze_1_example_1.png)

```
Input 1: a maze represented by a 2D array

0 0 1 0 0
0 0 0 0 0
0 0 0 1 0
1 1 0 1 1
0 0 0 0 0

Input 2: start coordinate (rowStart, colStart) = (0, 4)
Input 3: destination coordinate (rowDest, colDest) = (4, 4)

Output: 12
```

Explanation: One shortest way is : left -> down -> left -> down -> right -> down -> right.
The total distance is 1 + 1 + 3 + 1 + 2 + 2 + 2 = 12.

![Maze 2](https://assets.leetcode.com/uploads/2018/10/13/maze_1_example_2.png)

```
Input 1: a maze represented by a 2D array

0 0 1 0 0
0 0 0 0 0
0 0 0 1 0
1 1 0 1 1
0 0 0 0 0

Input 2: start coordinate (rowStart, colStart) = (0, 4)
Input 3: destination coordinate (rowDest, colDest) = (3, 2)

Output: -1
```

Explanation: There is no way for the ball to stop at the destination.
