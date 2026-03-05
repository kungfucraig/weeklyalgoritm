# Fenwick Tree (Binary Indexed Tree)

*Algorithm of the Day · 2026-03-04*

## Introduction

The Fenwick Tree, also known as a Binary Indexed Tree (BIT), is one of the most elegant data structures in competitive programming and systems engineering alike. Invented by Peter Fenwick in 1994, it provides a way to efficiently compute prefix sums and update individual elements in an array, both in O(log n) time — a dramatic improvement over the naive O(n) prefix sum recalculation. What makes it special is that it achieves this with remarkably little code and almost no overhead compared to a plain array.

At first glance, the problem seems simple: given an array of numbers, answer queries like "what is the sum of elements from index 1 to index k?" and handle updates like "add a value to element at index i." A plain array gives you O(1) updates but O(n) prefix queries, while a precomputed prefix sum array gives O(1) queries but O(n) updates. A segment tree solves both in O(log n) but uses 2-4x the memory and involves complex code. The Fenwick Tree hits the sweet spot: O(log n) for both operations, using only n+1 array slots, and the entire implementation fits in about 10 lines of code.

The secret behind the Fenwick Tree is a clever exploitation of binary representations of indices. Each position in the tree is responsible for storing a partial sum that covers a range determined by the lowest set bit of its index. This bit-manipulation trick — using `i & (-i)` to isolate the lowest set bit — is the heartbeat of the data structure and gives it both its power and its elegance. Understanding this connection between binary arithmetic and range coverage is one of those "aha" moments that deepens your appreciation for how mathematics and computer science intertwine.

## How It Works

A Fenwick Tree is stored as a simple 1-indexed array `tree[]` of the same size as the input. Each index `i` in the tree stores the sum of a specific range of elements from the original array. The range that index `i` is responsible for is determined by the lowest set bit (LSB) of `i` in binary. Specifically, `tree[i]` stores the sum of elements from index `i - LSB(i) + 1` to index `i`, where `LSB(i) = i & (-i)`. For example, index 12 (binary 1100) has LSB = 4, so `tree[12]` stores the sum of original elements at indices 9 through 12.

*Fenwick Tree responsibility ranges for an array of size 8*

```
Index (binary)  | LSB  | tree[i] covers indices
-------------------------------------------------
  1  (0001)     |  1   | [1, 1]    (1 element)
  2  (0010)     |  2   | [1, 2]    (2 elements)
  3  (0011)     |  1   | [3, 3]    (1 element)
  4  (0100)     |  4   | [1, 4]    (4 elements)
  5  (0101)     |  1   | [5, 5]    (1 element)
  6  (0110)     |  2   | [5, 6]    (2 elements)
  7  (0111)     |  1   | [7, 7]    (1 element)
  8  (1000)     |  8   | [1, 8]    (8 elements)

Tree structure (parent covers children's ranges):

              tree[8]
           /          \
      tree[4]         tree[6]
      /    \          /    \
  tree[2] tree[3] tree[5] tree[7]
    /
 tree[1]
```

**Prefix sum query (sum from index 1 to i):** Start at index `i` and accumulate `tree[i]` into the result. Then strip the lowest set bit by computing `i = i - (i & -i)`. Repeat until `i` becomes 0. Each step moves to the index that covers the next non-overlapping range to the left. Because each step removes at least one set bit, the loop runs at most O(log n) times. For example, to compute prefix_sum(7): start at 7 (0111), add tree[7] (covers [7,7]); move to 6 (0110), add tree[6] (covers [5,6]); move to 4 (0100), add tree[4] (covers [1,4]); move to 0, stop. Result = tree[7] + tree[6] + tree[4].

*Query path: prefix_sum(7) traversal*

```
Querying prefix_sum(7):

  i = 7 (0111) --> add tree[7]  (sum of [7,7])
       |  strip LSB: 0111 - 0001 = 0110
  i = 6 (0110) --> add tree[6]  (sum of [5,6])
       |  strip LSB: 0110 - 0010 = 0100
  i = 4 (0100) --> add tree[4]  (sum of [1,4])
       |  strip LSB: 0100 - 0100 = 0000
  i = 0 --> STOP

  Result = tree[7] + tree[6] + tree[4]
         = a[7]   + (a[5]+a[6]) + (a[1]+a[2]+a[3]+a[4])
         = sum of a[1..7]  ✓
```

**Point update (add a value delta to index i):** Start at index `i` and add `delta` to `tree[i]`. Then add the lowest set bit by computing `i = i + (i & -i)`. Repeat until `i` exceeds n. Each step moves to the next ancestor index whose range includes the updated position. This ensures all partial sums that include position `i` are updated. Again, at most O(log n) steps are needed. For example, to update index 3: update tree[3] (0011), then tree[4] (0100), then tree[8] (1000), stop.

*Update path: updating index 3 propagates upward*

```
Updating index 3 with delta:

  i = 3 (0011) --> tree[3] += delta  (covers [3,3])
       |  add LSB: 0011 + 0001 = 0100
  i = 4 (0100) --> tree[4] += delta  (covers [1,4])
       |  add LSB: 0100 + 0100 = 1000
  i = 8 (1000) --> tree[8] += delta  (covers [1,8])
       |  add LSB: 1000 + 1000 = 10000 > 8
  STOP

  All tree nodes whose range includes index 3 are updated.
```

**Range sum queries** are handled by computing `prefix_sum(r) - prefix_sum(l - 1)`, giving the sum of elements from index `l` to `r` in O(log n). The Fenwick Tree can also be extended to support range updates with point queries (by storing differences), or even range updates with range queries (using two BITs). There are also 2D variants for handling prefix sums over matrices. Despite these extensions, the core idea — exploiting the lowest set bit to define non-overlapping ranges — remains the same.

## Example

Let's build a Fenwick Tree from the array [0, 1, 3, 2, 5, 1, 4, 2] (1-indexed, so index 0 is unused). We'll construct the tree, perform prefix sum queries, demonstrate a point update, and compute a range sum. This implementation is clean and practical — you could drop it directly into a competitive programming solution or use it as a building block in a production system that needs efficient cumulative frequency tracking.

```python
class FenwickTree:
    """Binary Indexed Tree supporting point updates and prefix sum queries."""

    def __init__(self, n):
        """Initialize a Fenwick Tree of size n (1-indexed)."""
        self.n = n
        self.tree = [0] * (n + 1)

    @classmethod
    def from_array(cls, arr):
        """Build a Fenwick Tree from a 1-indexed array in O(n) time."""
        n = len(arr) - 1  # arr[0] is unused; elements are arr[1..n]
        ft = cls(n)
        # Copy values into tree, then propagate partial sums upward
        for i in range(1, n + 1):
            ft.tree[i] += arr[i]
            parent = i + (i & -i)
            if parent <= n:
                ft.tree[parent] += ft.tree[i]
        return ft

    def update(self, i, delta):
        """Add delta to element at index i. O(log n)"""
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)  # Move to next responsible ancestor

    def prefix_sum(self, i):
        """Return sum of elements from index 1 to i. O(log n)"""
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)  # Strip lowest set bit
        return s

    def range_sum(self, l, r):
        """Return sum of elements from index l to r (inclusive). O(log n)"""
        return self.prefix_sum(r) - self.prefix_sum(l - 1)


# --- Worked Example ---
# Original array (1-indexed): [_, 1, 3, 2, 5, 1, 4, 2]
arr = [0, 1, 3, 2, 5, 1, 4, 2]  # index 0 unused

# Build the Fenwick Tree in O(n)
ft = FenwickTree.from_array(arr)

# Query: prefix sum up to index 7
# Expected: 1 + 3 + 2 + 5 + 1 + 4 + 2 = 18
print(f"prefix_sum(7) = {ft.prefix_sum(7)}")  # Output: 18

# Query: prefix sum up to index 4
# Expected: 1 + 3 + 2 + 5 = 11
print(f"prefix_sum(4) = {ft.prefix_sum(4)}")  # Output: 11

# Query: range sum from index 3 to 6
# Expected: 2 + 5 + 1 + 4 = 12
print(f"range_sum(3, 6) = {ft.range_sum(3, 6)}")  # Output: 12

# Update: add 3 to index 3 (value changes from 2 to 5)
ft.update(3, 3)

# Verify: prefix_sum(7) should now be 18 + 3 = 21
print(f"After update, prefix_sum(7) = {ft.prefix_sum(7)}")  # Output: 21

# Verify: range_sum(3, 6) should now be 12 + 3 = 15
print(f"After update, range_sum(3, 6) = {ft.range_sum(3, 6)}")  # Output: 15

# Query: prefix_sum of full array
print(f"prefix_sum(7) = {ft.prefix_sum(7)}")  # Output: 21
print(f"Total sum = {ft.prefix_sum(ft.n)}")  # Output: 21
```

## Performance Analysis

Both the prefix sum query and point update operations run in O(log n) time because the binary representation of any index has at most floor(log2(n)) + 1 bits, and each operation either strips or adds the lowest set bit per iteration. Construction from an existing array can be done in O(n) using the cascading approach shown in the code above (propagating each node's value to its immediate parent). The space usage is minimal — just a single array of n+1 integers, making it one of the most space-efficient data structures for this problem. Compared to a segment tree, which requires 2n to 4n space and more complex code, the Fenwick Tree's constant factors are significantly smaller in both time and space.

| Metric | Complexity |
|--------|------------|
| Best case | `O(1) — query/update when the index has few set bits` |
| Average case | `O(log n) — for both prefix_sum and update` |
| Worst case | `O(log n) — for both prefix_sum and update` |
| Space | `O(n) — a single array of n+1 elements` |

## Use Cases

- Competitive programming: Fenwick Trees are a staple for problems involving prefix sums, inversions counting, and dynamic order statistics due to their simplicity and speed.
- Cumulative frequency tables: Used in arithmetic coding and data compression to maintain and query symbol frequencies that change over time, enabling adaptive compression schemes.
- Database systems and analytics: Efficient computation of running totals, cumulative aggregates, and sliding window statistics in OLAP systems where data is frequently updated.
- Counting inversions in arrays: By processing elements and querying how many previously seen elements are greater, Fenwick Trees solve the inversion counting problem in O(n log n), useful in ranking and recommendation systems.
- 2D range sum queries: Extended to two dimensions for image processing and spatial data analysis, where you need to quickly compute sums over rectangular sub-regions of a matrix that receives frequent updates.

## References & Further Reading

- [A New Data Structure for Cumulative Frequency Tables](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=2b8db5de0be31e5027f3d0e082cfb7a4e6b8b82e) — Peter M. Fenwick  
  The original 1994 paper by Fenwick introducing the Binary Indexed Tree and explaining its connection to binary arithmetic.
- [Fenwick tree — Wikipedia](https://en.wikipedia.org/wiki/Fenwick_tree)  
  Comprehensive overview of the data structure with diagrams, pseudocode, and discussion of variants.
- [Binary Indexed Trees — TopCoder Tutorial](https://www.topcoder.com/thrive/articles/Binary%20Indexed%20Trees) — boba5551  
  A well-known competitive programming tutorial that explains BIT with worked examples and extensions to 2D.
- **Competitive Programming 3** — Steven Halim, Felix Halim  
  Popular competitive programming textbook with thorough coverage of Fenwick Trees alongside segment trees and other range query structures.
- [Algorithms for Competitive Programming — Fenwick Tree](https://cp-algorithms.com/data_structures/fenwick.html)  
  Detailed tutorial covering basic BIT, range update variants, 2D BIT, and applications with clean C++ implementations.
- **Introduction to Algorithms (CLRS), 4th Edition** — Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein  
  While BIT is not directly covered, the textbook's treatment of augmented data structures and prefix computations provides essential foundational context.
