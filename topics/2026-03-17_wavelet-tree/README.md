# Wavelet Tree

*Algorithm of the Day · 2026-03-17*

## Introduction

The Wavelet Tree is a remarkably elegant data structure that answers a wide variety of queries on sequences in O(log σ) time, where σ is the alphabet size. Originally introduced by Grossi, Gupta, and Vitter in 2003, it decomposes a sequence over an alphabet into a balanced binary tree of bitvectors, enabling operations like rank (how many times does symbol c appear up to position i?), select (where is the k-th occurrence of symbol c?), range quantile queries, and even range frequency queries — all from a single, unified structure.

What makes the Wavelet Tree so fascinating is its versatility. It sits at the intersection of succinct data structures and practical sequence analysis. While a naive approach to answering rank queries for arbitrary alphabets might require O(σ) separate structures, the Wavelet Tree consolidates everything into O(n log σ) bits of space — essentially the same as the original sequence — while supporting a rich menu of operations. It has become a foundational building block in compressed full-text indices (like the FM-index), computational geometry, and document retrieval.

For working software engineers, the Wavelet Tree is a powerful tool to have in your mental toolkit. Anytime you face a problem involving positional queries over sequences with moderate alphabet sizes — think DNA sequences, word-level text operations, permutation inversions, or even 2D range counting — a Wavelet Tree can often provide an elegant and efficient solution where brute-force approaches would be too slow.

## How It Works

A Wavelet Tree is built over a sequence S[0..n-1] drawn from an alphabet Σ = {0, 1, ..., σ-1}. The key idea is to recursively partition the alphabet in half at each level of a binary tree. At the root, we split the alphabet into a 'left half' and a 'right half'. We create a bitvector B where B[i] = 0 if S[i] belongs to the left half of the alphabet, and B[i] = 1 if it belongs to the right half. The characters marked 0 are sent (in order) to the left child, and those marked 1 are sent to the right child. This process continues recursively until each leaf represents a single symbol.

*Wavelet Tree construction for sequence S = [3, 0, 1, 3, 2, 1, 0, 2] with alphabet {0,1,2,3}*

```
Alphabet split: {0,1} = left, {2,3} = right

                    Root: S = [3, 0, 1, 3, 2, 1, 0, 2]
                    Bitvector B = [1, 0, 0, 1, 1, 0, 0, 1]
                   /                                    \
         Left child                              Right child
         S_L = [0, 1, 1, 0]                     S_R = [3, 3, 2, 2]
         Split: {0}=L, {1}=R                     Split: {2}=L, {3}=R
         B_L = [0, 1, 1, 0]                     B_R = [1, 1, 0, 0]
        /            \                          /            \
    Leaf '0'      Leaf '1'                  Leaf '2'      Leaf '3'
    [0, 0]        [1, 1]                    [2, 2]        [3, 3]
```

The magic of the Wavelet Tree lies in how it navigates queries using rank operations on the bitvectors at each node. Each bitvector supports rank0(i) — the count of 0s up to position i — and rank1(i) — the count of 1s up to position i — both in O(1) time using standard succinct bitvector techniques (popcount with superblock/block indexing). This O(1) rank on bitvectors translates every sequence query into a walk down the tree of depth O(log σ).

To answer rank(c, i) — 'how many times does character c appear in S[0..i]?' — we start at the root. If c belongs to the left half of the alphabet, we compute i' = rank0(B, i) - 1 (the new position in the left child's sequence) and recurse left. If c belongs to the right half, we compute i' = rank1(B, i) - 1 and recurse right. At each level, the alphabet range halves, and the position index is mapped via the bitvector's rank. When we reach the leaf for c, the mapped position + 1 gives us the answer.

*Example query: rank(1, 5) — count occurrences of '1' in S[0..5] = [3,0,1,3,2,1]*

```
Root: B = [1, 0, 0, 1, 1, 0, 0, 1]
  '1' is in left half {0,1}, so go LEFT
  i' = rank0(B, 5) - 1 = 3 - 1 = 2
  (Three 0-bits in B[0..5]: positions 1,2,5)

Left child: B_L = [0, 1, 1, 0]
  '1' is in right half {1}, so go RIGHT
  i' = rank1(B_L, 2) - 1 = 2 - 1 = 1
  (Two 1-bits in B_L[0..2]: positions 1,2)

Leaf '1': position 1 => answer = 1 + 1 = 2

Verification: S[0..5] = [3,0,1,3,2,1] contains '1' twice. Correct!
```

Beyond basic rank and select, the Wavelet Tree supports a quantile query: given a range S[l..r], find the k-th smallest element. This is done by navigating top-down: at each node, count how many elements in the range go left (using rank0 differences) versus right. If k is within the left count, recurse left with an updated range; otherwise, recurse right. This gives O(log σ) range-quantile queries without any additional data structures — a result that would otherwise require complex structures like merge-sort trees or persistent segment trees.

The space used by a Wavelet Tree is n bits per level, and there are ⌈log₂ σ⌉ levels, yielding n⌈log₂ σ⌉ bits total — the same as the information-theoretic minimum for storing the sequence. With additional succinct indexing overhead of o(n log σ) bits for rank/select support on each bitvector, the structure is nearly as compact as the raw data while being far more powerful.

## Example

Let's implement a Wavelet Tree that supports rank, access (retrieve the symbol at a position), and quantile (k-th smallest in a range) queries. We'll build it over the sequence [3, 0, 1, 3, 2, 1, 0, 2] with alphabet {0, 1, 2, 3}. The bitvector rank operations use a simple prefix-sum approach for clarity, though production implementations would use popcount-based succinct bitvectors for O(1) rank.

```python
class WaveletTree:
    """Wavelet Tree supporting rank, access, and quantile queries."""

    def __init__(self, data, lo=None, hi=None):
        """Build the wavelet tree over `data` with alphabet range [lo, hi]."""
        if lo is None:
            lo = min(data)
        if hi is None:
            hi = max(data)
        self.lo = lo
        self.hi = hi
        self.bitvector = []
        self.left = None
        self.right = None

        # Base case: single symbol leaf
        if lo >= hi:
            self.count = len(data)  # number of occurrences at this leaf
            return

        mid = (lo + hi) // 2  # left half: [lo, mid], right half: [mid+1, hi]
        left_data = []
        right_data = []

        for val in data:
            if val <= mid:
                self.bitvector.append(0)
                left_data.append(val)
            else:
                self.bitvector.append(1)
                right_data.append(val)

        # Precompute prefix sums for O(1) rank queries on the bitvector
        # prefix_ones[i] = number of 1-bits in bitvector[0..i-1]
        self.prefix_ones = [0] * (len(self.bitvector) + 1)
        for i, b in enumerate(self.bitvector):
            self.prefix_ones[i + 1] = self.prefix_ones[i] + b

        # Recurse on children
        if left_data:
            self.left = WaveletTree(left_data, lo, mid)
        if right_data:
            self.right = WaveletTree(right_data, mid + 1, hi)

    def _rank1(self, i):
        """Count of 1-bits in bitvector[0..i] (inclusive)."""
        return self.prefix_ones[i + 1]

    def _rank0(self, i):
        """Count of 0-bits in bitvector[0..i] (inclusive)."""
        return (i + 1) - self._rank1(i)

    def rank(self, c, i):
        """Count occurrences of symbol c in the original sequence S[0..i]."""
        if self.lo >= self.hi:
            return i + 1 if self.lo == c else 0

        mid = (self.lo + self.hi) // 2
        if c <= mid:
            # c is in the left half; map index using rank0
            new_i = self._rank0(i) - 1
            if new_i < 0 or self.left is None:
                return 0
            return self.left.rank(c, new_i)
        else:
            # c is in the right half; map index using rank1
            new_i = self._rank1(i) - 1
            if new_i < 0 or self.right is None:
                return 0
            return self.right.rank(c, new_i)

    def access(self, i):
        """Retrieve the symbol at position i in the original sequence."""
        if self.lo >= self.hi:
            return self.lo

        if self.bitvector[i] == 0:
            new_i = self._rank0(i) - 1
            return self.left.access(new_i)
        else:
            new_i = self._rank1(i) - 1
            return self.right.access(new_i)

    def quantile(self, l, r, k):
        """Find the k-th smallest element (1-indexed) in S[l..r]."""
        if self.lo >= self.hi:
            return self.lo

        # Count how many elements in S[l..r] go to the left child
        ones_before_l = self.prefix_ones[l] if l > 0 else 0
        ones_up_to_r = self.prefix_ones[r + 1]
        right_count = ones_up_to_r - ones_before_l
        left_count = (r - l + 1) - right_count

        if k <= left_count:
            # k-th smallest is in the left subtree
            new_l = self._rank0(l - 1) if l > 0 else 0
            new_r = self._rank0(r) - 1
            return self.left.quantile(new_l, new_r, k)
        else:
            # k-th smallest is in the right subtree
            new_l = self._rank1(l - 1) if l > 0 else 0
            new_r = self._rank1(r) - 1
            return self.right.quantile(new_l, new_r, k - left_count)


# --- Demo ---
data = [3, 0, 1, 3, 2, 1, 0, 2]
print(f"Sequence: {data}")
wt = WaveletTree(data)

# Access: retrieve each element
print("\nAccess queries:")
for i in range(len(data)):
    print(f"  access({i}) = {wt.access(i)}  (expected {data[i]})")

# Rank: count occurrences of each symbol up to position 5
print("\nRank queries (up to index 5):")
for c in range(4):
    result = wt.rank(c, 5)
    expected = data[:6].count(c)
    print(f"  rank({c}, 5) = {result}  (expected {expected})")

# Quantile: k-th smallest in range [1, 6]
print("\nQuantile queries on S[1..6]:")
subseq = sorted(data[1:7])
for k in range(1, 7):
    result = wt.quantile(1, 6, k)
    print(f"  quantile(1, 6, {k}) = {result}  (expected {subseq[k-1]})")
```

## Performance Analysis

All primary operations on a Wavelet Tree — rank, select, access, and quantile — run in O(log σ) time, where σ is the size of the alphabet. This is because each operation performs a single root-to-leaf traversal of the tree, which has ⌈log₂ σ⌉ levels, doing O(1) work (bitvector rank/select) at each level. Construction takes O(n log σ) time as each element is processed once per level. The space consumption is n⌈log₂ σ⌉ bits for the bitvectors plus o(n log σ) bits for rank/select auxiliary structures, which is near-optimal.

| Metric | Complexity |
|--------|------------|
| Best case | `O(log σ) per query` |
| Average case | `O(log σ) per query` |
| Worst case | `O(log σ) per query; O(n log σ) construction` |
| Space | `O(n log σ) bits` |

## Use Cases

- Compressed full-text indexing (FM-index): Wavelet Trees provide the rank operation over the BWT (Burrows-Wheeler Transform) needed for backward search in compressed text indices, enabling pattern matching in space close to the compressed text size.
- Range quantile and range frequency queries: Given a static array, answer 'what is the k-th smallest element in positions l to r?' or 'how many times does value c appear in positions l to r?' in O(log σ) time — useful in databases, competitive programming, and analytics.
- Document retrieval and information retrieval: Wavelet Trees on document arrays enable efficient listing of distinct documents containing a query pattern, a key primitive in search engines.
- Computational geometry: By mapping 2D points to a sequence, Wavelet Trees can answer orthogonal range counting queries (how many points lie in a rectangle?) in O(log σ) time, serving as a practical alternative to range trees.
- Bioinformatics sequence analysis: DNA and protein sequences over small alphabets benefit from Wavelet Tree-based indices for compressed storage and fast pattern counting in genomic databases.

## References & Further Reading

- [Wavelet Trees for All](https://users.dcc.uchile.cl/~gnavarro/ps/jda14.pdf) — Gonzalo Navarro  
  A comprehensive survey of Wavelet Tree variants, operations, and applications by one of the leading researchers in succinct data structures.
- [Compressed Suffix Arrays and Suffix Trees with Applications to Text Indexing and String Matching](https://doi.org/10.1145/1005813.1041680) — Roberto Grossi, Ankur Gupta, Jeffrey Scott Vitter  
  The original paper that introduced the Wavelet Tree as part of compressed suffix array constructions.
- [Wavelet Tree (Wikipedia)](https://en.wikipedia.org/wiki/Wavelet_tree)  
  A concise overview of the Wavelet Tree data structure with references to key papers and complexity results.
- **Compact Data Structures: A Practical Approach** — Gonzalo Navarro  
  A textbook covering succinct and compact data structures including Wavelet Trees, bitvectors, and their applications to text indexing.
- [Succinct Data Structures Library (SDSL)](https://github.com/simongog/sdsl-lite) — Simon Gog, Timo Beller, Alistair Moffat, Matthias Petri  
  A highly optimized C++ library implementing Wavelet Trees and other succinct data structures, widely used in research and practice.
