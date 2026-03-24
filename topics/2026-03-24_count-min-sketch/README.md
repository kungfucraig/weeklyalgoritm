# Count-Min Sketch

*Algorithm of the Day · 2026-03-24*

## Introduction

Imagine you're building a system that processes billions of events per day — web requests, network packets, search queries — and you need to answer questions like "how many times has this item appeared?" Keeping exact counts for every distinct item is often infeasible: the cardinality might be in the hundreds of millions, memory is constrained, and you need answers fast. The Count-Min Sketch is a probabilistic data structure that provides approximate frequency estimates using dramatically less memory than exact counting, with mathematically bounded error guarantees.

Introduced by Graham Cormode and S. Muthukrishnan in 2005, the Count-Min Sketch belongs to the family of streaming or sketch data structures designed for the data stream model, where elements arrive one at a time and you cannot afford to store or revisit the entire input. Unlike a hash map that gives exact answers, a Count-Min Sketch trades a small, controllable amount of accuracy for enormous savings in space. It only over-estimates frequencies (never under-estimates), which makes it particularly useful in settings where false positives are tolerable but false negatives are not.

What makes the Count-Min Sketch especially elegant is its simplicity: it's essentially a two-dimensional array of counters with multiple hash functions — conceptually similar to a Bloom filter, but for counting rather than membership testing. Despite this simplicity, it has found its way into production systems at massive scale, from network traffic monitoring at Cisco to query optimization in Apache Spark, and from trending topic detection at Twitter to frequency estimation in database engines like PostgreSQL.

## How It Works

A Count-Min Sketch consists of a 2D array of counters with dimensions d × w, where d is the number of hash functions (rows) and w is the width of each row (number of columns). The parameters are chosen based on desired accuracy: w = ⌈e/ε⌉ and d = ⌈ln(1/δ)⌉, where ε controls the error margin and δ controls the probability of exceeding that error. Each of the d rows is associated with an independent pairwise hash function that maps items to columns in the range [0, w-1]. All counters are initialized to zero.

*Count-Min Sketch structure: d rows × w columns with d independent hash functions*

```
                     w columns
            0     1     2     3     4     5     6
          +-----+-----+-----+-----+-----+-----+-----+
  h1(x) ->|  0  |  0  |  0  |  0  |  0  |  0  |  0  |  Row 0
          +-----+-----+-----+-----+-----+-----+-----+
  h2(x) ->|  0  |  0  |  0  |  0  |  0  |  0  |  0  |  Row 1
          +-----+-----+-----+-----+-----+-----+-----+
  h3(x) ->|  0  |  0  |  0  |  0  |  0  |  0  |  0  |  Row 2
          +-----+-----+-----+-----+-----+-----+-----+
  h4(x) ->|  0  |  0  |  0  |  0  |  0  |  0  |  0  |  Row 3
          +-----+-----+-----+-----+-----+-----+-----+

  Each hash function h_i maps an item to one column per row.
```

To INSERT (or update) an item x with a count of c (typically c=1 for each occurrence), you compute h_i(x) for each row i from 0 to d-1, and increment the counter at position [i, h_i(x)] by c. This is an O(d) operation, and since d is a small constant determined at initialization time, insertion is effectively O(1). Multiple different items may hash to the same cell (a collision), which causes over-counting — this is the fundamental source of approximation error.

*Inserting item 'apple' (count += 1): each hash maps to one cell per row*

```
  Item: 'apple'
  h1('apple') = 2,  h2('apple') = 5,  h3('apple') = 1,  h4('apple') = 4

            0     1     2     3     4     5     6
          +-----+-----+-----+-----+-----+-----+-----+
  Row 0   |  0  |  0  | [1] |  0  |  0  |  0  |  0  |  <-- increment col 2
          +-----+-----+-----+-----+-----+-----+-----+
  Row 1   |  0  |  0  |  0  |  0  |  0  | [1] |  0  |  <-- increment col 5
          +-----+-----+-----+-----+-----+-----+-----+
  Row 2   |  0  | [1] |  0  |  0  |  0  |  0  |  0  |  <-- increment col 1
          +-----+-----+-----+-----+-----+-----+-----+
  Row 3   |  0  |  0  |  0  |  0  | [1] |  0  |  0  |  <-- increment col 4
          +-----+-----+-----+-----+-----+-----+-----+
```

To QUERY the estimated frequency of an item x, you compute h_i(x) for each row, retrieve the counter at [i, h_i(x)], and return the MINIMUM of all d values. The key insight is that each counter may be inflated by collisions from other items, but it can never be less than the true count. By taking the minimum across d independent hash functions, you significantly reduce the chance that all d counters are simultaneously inflated by a large amount. Formally, the estimate f̂(x) satisfies: f(x) ≤ f̂(x) ≤ f(x) + ε·N with probability at least 1-δ, where f(x) is the true frequency and N is the total count of all items.

*Querying item 'apple' after more insertions — take the minimum across rows*

```
  After inserting: apple x3, banana x2, cherry x1, date x1

            0     1     2     3     4     5     6
          +-----+-----+-----+-----+-----+-----+-----+
  Row 0   |  1  |  0  | [4] |  0  |  2  |  0  |  0  |
          +-----+-----+-----+-----+-----+-----+-----+
  Row 1   |  0  |  1  |  2  |  0  |  0  | [3] |  1  |
          +-----+-----+-----+-----+-----+-----+-----+
  Row 2   |  0  | [3] |  0  |  1  |  1  |  2  |  0  |
          +-----+-----+-----+-----+-----+-----+-----+
  Row 3   |  2  |  0  |  0  |  1  | [3] |  0  |  1  |
          +-----+-----+-----+-----+-----+-----+-----+

  Query 'apple': counters = [4, 3, 3, 3]
  Estimate = min(4, 3, 3, 3) = 3   (true count = 3)  Exact!
```

An important property of the Count-Min Sketch is that it is mergeable: two sketches built with the same hash functions and dimensions can be combined by element-wise addition of their counter arrays. This makes it ideal for distributed systems where each node maintains a local sketch and periodically merges with a central aggregator. Additionally, the sketch supports deletion by decrementing counters (called Count-Min-Log or conservative update variants), though plain deletion can lead to negative bias. The conservative update optimization only increments a counter if it is currently the minimum, which reduces over-estimation in practice.

## Example

Let's build a Count-Min Sketch from scratch and use it to estimate word frequencies in a stream of text. We'll simulate a scenario where we're tracking the frequency of words in a stream of 100,000 tokens, then compare our estimates against the true counts to see how close the approximations are. We'll use ε=0.001 and δ=0.01, meaning we want the error to be at most 0.1% of total stream size, with 99% confidence.

```python
import hashlib
import random
import math
from collections import Counter

class CountMinSketch:
    def __init__(self, epsilon, delta):
        """Initialize with error rate epsilon and failure probability delta."""
        self.w = math.ceil(math.e / epsilon)     # width of each row
        self.d = math.ceil(math.log(1.0 / delta)) # number of rows (hash functions)
        self.table = [[0] * self.w for _ in range(self.d)]
        self.total = 0
        # Generate random seeds for each hash function
        self.seeds = [random.randint(0, 2**32 - 1) for _ in range(self.d)]
        print(f"Sketch dimensions: {self.d} rows x {self.w} cols")
        print(f"Total counters: {self.d * self.w}")
        print(f"Memory: ~{self.d * self.w * 4 / 1024:.1f} KB (4 bytes/counter)")

    def _hash(self, item, seed):
        """Hash an item with a given seed to a column index."""
        # Use MD5 for simplicity; production systems use faster hashes
        h = hashlib.md5(f"{seed}:{item}".encode()).hexdigest()
        return int(h, 16) % self.w

    def add(self, item, count=1):
        """Add an item to the sketch with an optional count."""
        self.total += count
        for i in range(self.d):
            col = self._hash(item, self.seeds[i])
            self.table[i][col] += count

    def query(self, item):
        """Estimate the frequency of an item (returns min across all rows)."""
        estimates = []
        for i in range(self.d):
            col = self._hash(item, self.seeds[i])
            estimates.append(self.table[i][col])
        return min(estimates)

    def merge(self, other):
        """Merge another sketch into this one (must have same dimensions/seeds)."""
        assert self.w == other.w and self.d == other.d
        for i in range(self.d):
            for j in range(self.w):
                self.table[i][j] += other.table[i][j]
        self.total += other.total


# --- Demonstration ---
random.seed(42)

# Create a Zipfian-like word distribution (realistic for natural language)
words = [f"word_{i}" for i in range(5000)]
weights = [1.0 / (i + 1) for i in range(5000)]  # Zipf's law
total_weight = sum(weights)
probs = [w / total_weight for w in weights]

# Generate a stream of 100,000 tokens
stream = random.choices(words, weights=probs, k=100000)

# Build exact counts for comparison
exact_counts = Counter(stream)

# Build Count-Min Sketch with epsilon=0.001, delta=0.01
cms = CountMinSketch(epsilon=0.001, delta=0.01)
for token in stream:
    cms.add(token)

# Compare estimates vs true counts for top-20 most frequent words
print("\n{:<15} {:>10} {:>10} {:>10}".format(
    "Word", "True", "Estimate", "Error"))
print("-" * 48)

errors = []
for word, true_count in exact_counts.most_common(20):
    estimate = cms.query(word)
    error = estimate - true_count  # Always >= 0
    errors.append(error)
    print("{:<15} {:>10} {:>10} {:>10}".format(
        word, true_count, estimate, f"+{error}"))

# Overall statistics across ALL words
all_errors = []
for word in exact_counts:
    est = cms.query(word)
    all_errors.append(est - exact_counts[word])

print(f"\n--- Overall Statistics ---")
print(f"Total stream items: {cms.total}")
print(f"Distinct items: {len(exact_counts)}")
print(f"Mean absolute error: {sum(all_errors)/len(all_errors):.2f}")
print(f"Max error: {max(all_errors)}")
print(f"Error bound (epsilon * N): {0.001 * 100000:.0f}")
print(f"Items within error bound: {sum(1 for e in all_errors if e <= 100)}/{len(all_errors)}")
```

## Performance Analysis

The Count-Min Sketch achieves constant-time operations with space that depends only on the desired accuracy, not on the number of distinct items. Specifically, insertion and query each require d hash computations, where d = O(ln(1/δ)), and the total space is O((1/ε) · ln(1/δ)) counters. For fixed accuracy parameters, all operations are O(1). The mergeability of two sketches takes O(w·d) time, which is proportional to the sketch size. This makes the Count-Min Sketch particularly attractive for streaming and distributed settings where both time and space must be carefully controlled.

| Metric | Complexity |
|--------|------------|
| Best case | `O(d) per operation — effectively O(1) for fixed accuracy parameters` |
| Average case | `O(d) per operation — same for insert, query, and point delete` |
| Worst case | `O(d) per operation — no worst-case degradation beyond the fixed d hash computations` |
| Space | `O(w × d) = O((1/ε) · ln(1/δ)) counters, independent of stream length or number of distinct items` |

## Use Cases

- Network traffic monitoring and heavy-hitter detection: ISPs and CDNs use Count-Min Sketches to identify the most frequent source IPs, detect DDoS attacks, and monitor bandwidth usage per flow without storing per-flow state for millions of connections.
- Database query optimization: Systems like Apache Spark, PostgreSQL, and Google's Dremel use frequency sketches to estimate value distributions and selectivity, helping the query planner choose efficient join strategies and index usage.
- Real-time trending and analytics: Social media platforms like Twitter use Count-Min Sketches to detect trending hashtags and topics from firehose streams, where maintaining exact counts for millions of terms would be prohibitively expensive.
- Natural language processing and text mining: Count-Min Sketches serve as memory-efficient replacements for frequency dictionaries when building n-gram language models, computing pointwise mutual information, or performing feature hashing in large-scale ML pipelines.
- Click fraud and anomaly detection: Ad-tech platforms use streaming frequency sketches to flag anomalous click patterns in real time, identifying IPs or user agents with suspiciously high activity without maintaining full click logs in memory.

## References & Further Reading

- [An Improved Data Stream Summary: The Count-Min Sketch and its Applications](https://www.cs.rutgers.edu/~muthu/cm-latin.pdf) — Graham Cormode, S. Muthukrishnan  
  The original 2005 paper introducing the Count-Min Sketch with full theoretical analysis and applications to point queries, range queries, and inner products.
- [Count-Min Sketch — Wikipedia](https://en.wikipedia.org/wiki/Count%E2%80%93min_sketch)  
  A well-maintained overview of the data structure, its variants, and theoretical properties.
- [Approximating Data with the Count-Min Data Structure](http://dimacs.rutgers.edu/~graham/pubs/papers/cmencyc.pdf) — Graham Cormode, S. Muthukrishnan  
  An accessible survey article by the original authors covering practical aspects and extensions of the Count-Min Sketch.
- **Sketch of the Day: Count-Min Sketch** — Aggregate Knowledge (blog)  
  A practitioner-focused blog post explaining the intuition, implementation considerations, and real-world deployment of Count-Min Sketches.
- [Mining of Massive Datasets (Chapter 4: Mining Data Streams)](http://www.mmds.org/) — Jure Leskovec, Anand Rajaraman, Jeffrey D. Ullman  
  A comprehensive textbook chapter covering streaming algorithms including Count-Min Sketch, with practical examples and exercises.
- **Probabilistic Data Structures and Algorithms for Big Data Applications** — Andrii Gakhov  
  A book dedicated to probabilistic data structures including Count-Min Sketch, Bloom filters, and HyperLogLog, with implementation details and comparisons.
