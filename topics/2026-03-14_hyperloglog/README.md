# HyperLogLog

*Algorithm of the Day · 2026-03-14*

## Introduction

Imagine you're running a large-scale web analytics platform and you need to answer a deceptively simple question: how many unique visitors hit your site today? If you have billions of events, storing every unique identifier in a hash set would consume enormous amounts of memory. What if you could estimate that count using only about 12 kilobytes of memory, regardless of whether you're counting thousands or billions of distinct elements? That's exactly what HyperLogLog (HLL) delivers.

HyperLogLog is a probabilistic cardinality estimation algorithm introduced by Philippe Flajolet and colleagues in 2007. It belongs to the family of streaming algorithms — data structures that process elements in a single pass and use sublinear space. HLL builds on a beautiful probabilistic observation: if you hash elements uniformly at random, the maximum number of leading zeros you observe in the binary hash values is a surprisingly good estimator of how many distinct elements you've seen. The intuition is that seeing a run of k leading zeros is roughly a 1-in-2^k event, so if your longest observed run is k, you've probably seen around 2^k distinct items.

What makes HyperLogLog special among cardinality estimators is its remarkable combination of accuracy and memory efficiency. With just m = 2^p registers (where p is typically 14, giving 16,384 registers of 6 bits each — about 12 KB), HLL achieves a standard error of roughly 1.04/√m ≈ 0.81%. This means you can count billions of distinct elements with sub-percent error using less memory than a typical JPEG thumbnail. HLL has become a workhorse in production systems: Redis has it built in, Google's BigQuery uses a variant called HLL++, and virtually every major analytics platform relies on it.

## How It Works

The core idea behind HyperLogLog rests on a probabilistic thought experiment. Suppose you flip a fair coin repeatedly and record the length of the longest run of heads you see. If you've only flipped a few times, long runs are unlikely. But if you've flipped millions of times, you'll almost certainly see runs of 20 or more. In the same way, if you hash each element to a binary string, the maximum number of leading zeros across all hashed values gives an estimate of how many distinct elements you've processed. A single such estimator is very noisy, so HLL uses a technique called stochastic averaging to reduce variance.

Here's the algorithm step by step. First, choose a parameter p (typically 4 to 16). This determines m = 2^p registers, each initialized to 0. When an element arrives, hash it to a uniform binary string. Use the first p bits of the hash as an index to select one of the m registers. Then, in the remaining bits, count the position of the first 1-bit (equivalently, the number of leading zeros plus one). Update the selected register to be the maximum of its current value and this count. After processing all elements, combine the registers using a harmonic mean to produce the final estimate.

*HyperLogLog: hashing and register update*

```
  Element x
      |
      v
  hash(x) = 0 1 1 0 | 0 0 0 1 0 1 1 0 ...
             |_____| |_________________________|
             p bits         remaining bits
             = 6            leading zeros = 3
             (register      rank = 3 + 1 = 4
              index)

  Registers (m = 2^p):
  +----+----+----+----+----+----+----+----+---
  | R0 | R1 | R2 | R3 | R4 | R5 | R6 | R7 |...
  |  2 |  1 |  0 |  5 |  3 |  0 | *4*|  1 |...
  +----+----+----+----+----+----+----+----+---
                                  ^
                                  |
                       R[6] = max(old=2, 4) = 4
```

The final estimation step computes the harmonic mean of the register values. The raw estimate is E = alpha_m * m^2 * (sum of 2^(-R[j]) for j=0..m-1)^(-1), where alpha_m is a bias-correction constant that depends on m (for example, alpha_16384 ≈ 0.7213/(1 + 1.079/16384)). The harmonic mean is crucial because it is robust against outlier registers that have unusually high or low values, giving much better accuracy than a simple arithmetic mean would.

There are two important correction ranges. For small cardinalities (when many registers are still zero), the raw HLL estimate has significant bias, so the algorithm falls back to Linear Counting — a different estimator that uses the fraction of empty registers: E_LC = m * ln(m / V), where V is the number of zero-valued registers. For extremely large cardinalities (near 2^32 for 32-bit hashes), a large-range correction prevents hash collisions from causing underestimation. Google's HLL++ variant improves on the original by using empirically-derived bias correction for the small-to-medium range instead of the hard switchover to Linear Counting.

*Merging two HyperLogLog sketches (union operation)*

```
  HLL Sketch A:        HLL Sketch B:        Merged (A union B):
  +---+---+---+---+    +---+---+---+---+    +---+---+---+---+
  | 3 | 1 | 5 | 2 |    | 2 | 4 | 3 | 2 |    | 3 | 4 | 5 | 2 |
  +---+---+---+---+    +---+---+---+---+    +---+---+---+---+

  Rule: merged[i] = max(A[i], B[i]) for each register i

  This allows distributed counting: each node maintains
  its own sketch, and merging is a simple element-wise max.
```

One of HyperLogLog's most powerful properties is mergeability. To compute the cardinality of the union of two sets, you simply take the element-wise maximum of their register arrays. This makes HLL ideal for distributed systems: each node can independently maintain a small sketch, and sketches can be merged at any time with no loss of accuracy compared to having processed all elements in a single stream. This property is why HLL is the backbone of cardinality estimation in distributed databases, real-time analytics pipelines, and network monitoring systems.

## Example

Let's implement a complete HyperLogLog from scratch and test it by estimating the number of distinct elements in a stream. We'll generate 100,000 random strings, insert them, and compare the HLL estimate against the true count. We use p=14 (16,384 registers) for a standard error of about 0.81%. The implementation includes the small-range correction using Linear Counting.

```python
import hashlib
import math
import random
import string

class HyperLogLog:
    def __init__(self, p=14):
        """Initialize HLL with 2^p registers."""
        self.p = p
        self.m = 1 << p  # number of registers
        self.registers = [0] * self.m
        # Bias correction constant alpha_m
        if self.m == 16:
            self.alpha = 0.673
        elif self.m == 32:
            self.alpha = 0.697
        elif self.m == 64:
            self.alpha = 0.709
        else:
            self.alpha = 0.7213 / (1.0 + 1.079 / self.m)

    def _hash(self, item):
        """Hash an item to a 64-bit integer."""
        h = hashlib.sha256(str(item).encode('utf-8')).hexdigest()
        return int(h[:16], 16)  # use first 64 bits

    def _leading_zeros(self, value, max_bits=64):
        """Count leading zeros in the binary representation,
        considering only (max_bits - p) bits after the register index."""
        remaining_bits = max_bits - self.p
        if value == 0:
            return remaining_bits  # all zeros
        count = 0
        # Check bits from the top of the remaining portion
        for i in range(remaining_bits - 1, -1, -1):
            if value & (1 << i):
                break
            count += 1
        return count

    def add(self, item):
        """Add an element to the HLL sketch."""
        h = self._hash(item)
        # First p bits determine the register index
        register_index = h >> (64 - self.p)
        # Remaining bits are used to count leading zeros
        remaining = h & ((1 << (64 - self.p)) - 1)
        # rank = number of leading zeros + 1
        rank = self._leading_zeros(remaining) + 1
        # Update register with the maximum rank seen
        self.registers[register_index] = max(
            self.registers[register_index], rank
        )

    def estimate(self):
        """Return the estimated cardinality."""
        # Compute raw HLL estimate using harmonic mean
        indicator = sum(2.0 ** (-r) for r in self.registers)
        raw_estimate = self.alpha * self.m * self.m / indicator

        # Small range correction (Linear Counting)
        if raw_estimate <= 2.5 * self.m:
            # Count registers that are still zero
            zeros = self.registers.count(0)
            if zeros > 0:
                # Use Linear Counting estimate
                return self.m * math.log(self.m / zeros)
            else:
                return raw_estimate
        # Large range correction (for 32-bit hashes; not needed with 64-bit)
        elif raw_estimate > (1.0 / 30.0) * (1 << 64):
            return -(1 << 64) * math.log(1.0 - raw_estimate / (1 << 64))
        else:
            return raw_estimate

    def merge(self, other):
        """Merge another HLL sketch into this one (union)."""
        assert self.p == other.p, "HLL sketches must have same precision"
        for i in range(self.m):
            self.registers[i] = max(self.registers[i], other.registers[i])


# --- Demonstration ---
if __name__ == "__main__":
    hll = HyperLogLog(p=14)
    true_set = set()
    n = 100_000

    # Generate random strings and add to both HLL and a true set
    random.seed(42)
    for _ in range(n):
        item = ''.join(random.choices(string.ascii_lowercase, k=10))
        hll.add(item)
        true_set.add(item)

    true_count = len(true_set)
    estimated = hll.estimate()
    error_pct = abs(estimated - true_count) / true_count * 100

    print(f"True distinct count: {true_count}")
    print(f"HLL estimate:        {estimated:.0f}")
    print(f"Error:               {error_pct:.2f}%")
    print(f"Memory used:         {hll.m * 6 / 8:.0f} bytes "
          f"({hll.m} registers x 6 bits)")

    # Demonstrate merge: split stream across two sketches
    hll_a = HyperLogLog(p=14)
    hll_b = HyperLogLog(p=14)
    items = list(true_set)
    for item in items[:len(items)//2]:
        hll_a.add(item)
    for item in items[len(items)//2:]:
        hll_b.add(item)
    hll_a.merge(hll_b)

    print(f"\nMerged HLL estimate: {hll_a.estimate():.0f}")
    print(f"Merge error:         "
          f"{abs(hll_a.estimate() - true_count) / true_count * 100:.2f}%")
```

## Performance Analysis

HyperLogLog offers constant time per insertion and query, and uses logarithmically small space relative to the cardinality. Each add operation requires one hash computation and one register update, both O(1). The estimation step requires iterating over all m registers, which is O(m) — but m is a fixed constant chosen at initialization (typically 16,384), so this is effectively O(1) in terms of the number of elements processed. The space usage is O(m) = O(2^p) registers, each storing at most ~6 bits (enough to count up to 64 leading zeros), giving about 12 KB for p=14. Merging two sketches is O(m). The key insight is that space is completely independent of the number of distinct elements — whether you count 1,000 or 10 billion distinct items, the memory footprint remains the same.

| Metric | Complexity |
|--------|------------|
| Best case | `O(1) per add/query` |
| Average case | `O(1) per add/query` |
| Worst case | `O(1) per add/query` |
| Space | `O(m) where m = 2^p, typically ~12 KB for p=14` |

## Use Cases

- Real-time unique visitor counting in web analytics (e.g., counting distinct IPs or user IDs per day across billions of events with minimal memory)
- Redis PFADD/PFCOUNT commands: Redis natively implements HyperLogLog, enabling distributed applications to perform cardinality estimation with a simple API and automatic merging across keys
- Database query optimization in systems like BigQuery, Presto, and Apache Druid, which use HLL sketches for fast approximate COUNT(DISTINCT ...) queries over massive datasets
- Network monitoring and security: counting distinct source/destination IP pairs in high-speed packet streams to detect DDoS attacks or port scans without storing per-flow state
- Social media metrics: platforms like Facebook and Twitter use HLL-based approaches to compute unique reach, unique viewers, and similar metrics across distributed data centers

## References & Further Reading

- [HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm](http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf) — Philippe Flajolet, Éric Fusy, Olivier Gandouet, Frédéric Meunier  
  The original 2007 paper introducing HyperLogLog with full mathematical analysis of its accuracy guarantees.
- [HyperLogLog in Practice: Algorithmic Engineering of a State of The Art Cardinality Estimation Algorithm](https://research.google/pubs/pub40671/) — Stefan Heule, Marc Nunkesser, Alexander Hall  
  Google's paper on HLL++, which improves HyperLogLog's accuracy for small and intermediate cardinalities using empirical bias correction.
- [HyperLogLog — Wikipedia](https://en.wikipedia.org/wiki/HyperLogLog)  
  A comprehensive overview of the algorithm, its history, variants, and implementations with accessible mathematical exposition.
- [Redis HyperLogLog Documentation](https://redis.io/docs/data-types/hyperloglogs/)  
  Official Redis documentation on its native HyperLogLog implementation, including PFADD, PFCOUNT, and PFMERGE commands.
- **Probabilistic Data Structures and Algorithms for Big Data Applications** — Andrii Gakhov  
  A book covering Bloom filters, Count-Min Sketch, HyperLogLog, and other streaming/sketching algorithms with practical implementation guidance.
- [New cardinality estimation algorithms for HyperLogLog sketches](https://arxiv.org/abs/1702.01284) — Otmar Ertl  
  Proposes improved estimators for HyperLogLog that eliminate the need for empirical bias correction tables used in HLL++.
