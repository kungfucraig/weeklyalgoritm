# Locality-Sensitive Hashing (LSH)

*Algorithm of the Day · 2026-03-07*

## Introduction

Imagine you have a database of millions of images, documents, or high-dimensional feature vectors, and you need to find items that are "similar" to a given query — not exact matches, but approximate nearest neighbors. A naive brute-force scan is O(n·d) per query (where d is dimensionality), which becomes impractical at scale. This is the infamous "curse of dimensionality": traditional indexing structures like KD-trees degrade to linear scans as dimensions grow beyond about 20.

Locality-Sensitive Hashing (LSH) is a family of techniques that elegantly sidestep this curse by using randomized hash functions with a remarkable property: similar items are more likely to collide (hash to the same bucket) than dissimilar items. This is the exact opposite of what conventional hash functions try to achieve! By hashing a query and only comparing it against items in the same bucket, LSH provides sub-linear query times with provable probabilistic guarantees on recall.

Introduced by Indyk and Motwani in 1998, LSH has become one of the most important tools in large-scale machine learning, information retrieval, and data mining. It powers similarity search at companies like Spotify (song recommendations), Google (near-duplicate detection), and Uber (matching riders with drivers). If you work with high-dimensional data at any meaningful scale, understanding LSH is essential.

## How It Works

The core idea of LSH is to define a family of hash functions H such that for any two points p and q, the probability of a hash collision is a monotonically increasing function of their similarity (or decreasing function of their distance). Formally, a hash family H is (d1, d2, p1, p2)-sensitive if: for any points p, q where dist(p,q) ≤ d1, Pr[h(p) = h(q)] ≥ p1; and for any points p, q where dist(p,q) ≥ d2, Pr[h(p) = h(q)] ≤ p2, where d1 < d2 and p1 > p2. The gap between p1 and p2 is what gives LSH its power.

For cosine similarity, one of the most popular LSH families is based on random hyperplane projections (SimHash, by Charikar 2002). Each hash function picks a random vector r from a standard Gaussian distribution and computes h(v) = sign(r · v). The probability that two vectors get the same sign equals 1 - θ/π, where θ is the angle between them. Vectors that point in nearly the same direction almost always get the same hash bit, while orthogonal vectors get the same bit only 50% of the time.

*Random hyperplane partitioning in 2D: a single hyperplane (dashed line) splits similar points into the same half-space*

```
                  |  r (random normal)
                  |  ^
                  | /
        - - - - -/- - - - - - -   <-- hyperplane (perpendicular to r)
                /|
    A *        / |        * B
      * C     /  |
             /   |        * D
            /    |

    h(v) = sign(r . v)

    Points A, C are on the same side  --> h(A) = h(C) = +1
    Points B, D are on the other side --> h(B) = h(D) = -1
    Similar points (close angle) likely get same hash value
```

A single hash bit provides very weak discrimination. To amplify the gap between p1 and p2, LSH uses two strategies combined: AND-amplification and OR-amplification. In AND-amplification, we concatenate k hash functions into a single band — two items must agree on ALL k bits to collide, which makes collisions rarer overall but dramatically reduces false positives. In OR-amplification, we create L independent bands (hash tables) — items need to collide in at least ONE band to be considered candidates. By tuning k and L, we can achieve any desired trade-off between recall (finding true neighbors) and query speed.

*AND + OR amplification: L bands of k hash functions each*

```
    Input vector v
        |
        v
    +-------------------------------------------+
    | Band 1: [h1(v), h2(v), ..., hk(v)]  --> bucket in Table 1 |
    | Band 2: [h1'(v), h2'(v), ..., hk'(v)] --> bucket in Table 2 |
    | Band 3: [h1''(v), h2''(v),..., hk''(v)]--> bucket in Table 3 |
    |  ...                                                        |
    | Band L: [h1*(v), h2*(v), ..., hk*(v)] --> bucket in Table L |
    +-------------------------------------------+

    Candidate = any item sharing a bucket in ANY of the L tables

    k large --> fewer false positives (AND tightens)
    L large --> fewer false negatives (OR loosens)

    Overall collision probability:
    P(candidate) = 1 - (1 - p^k)^L

    where p = Pr[single hash collision]
```

The query process works in two phases. In the offline (indexing) phase, every data point is hashed into all L hash tables. In the online (query) phase, the query point is hashed into the same L tables, candidate items are collected from all matching buckets (union), duplicates are removed, and then exact distances are computed only for this small candidate set. The final result is the nearest neighbor(s) among the candidates. Because the candidate set is typically much smaller than the full dataset, queries run in sub-linear time.

Different distance metrics call for different LSH families. For Jaccard similarity, MinHash (Broder 1997) is the canonical choice: it randomly permutes the universe of elements and takes the minimum index. For Euclidean distance, p-stable distributions (Datar et al. 2004) project onto random lines and quantize into bins. For cosine similarity, SimHash uses random hyperplanes as described above. The framework is general — any hash family satisfying the locality-sensitive property works.

## Example

Let's implement a simple LSH index for cosine similarity using random hyperplane projections (SimHash). We'll create a small dataset of 10,000 random vectors in 128 dimensions, index them with LSH using L=10 bands of k=8 hash bits each, and then query for approximate nearest neighbors. We'll compare the LSH result against a brute-force scan to see how well it works.

This implementation is intentionally kept simple and self-contained. Production systems like Faiss or Annoy use more sophisticated techniques, but the core idea is the same.

```python
import numpy as np
from collections import defaultdict
import time

class SimHashLSH:
    """Locality-Sensitive Hashing for cosine similarity using random hyperplanes."""
    
    def __init__(self, dim, num_bands=10, band_width=8, seed=42):
        """
        dim: dimensionality of input vectors
        num_bands (L): number of hash tables
        band_width (k): number of hash bits per band
        """
        self.dim = dim
        self.L = num_bands
        self.k = band_width
        self.rng = np.random.RandomState(seed)
        
        # Generate random hyperplanes: L bands, each with k hyperplanes
        # Shape: (L, k, dim)
        self.hyperplanes = self.rng.randn(self.L, self.k, dim)
        
        # One hash table (dict) per band
        self.tables = [defaultdict(list) for _ in range(self.L)]
        self.data = {}  # id -> vector
    
    def _hash_vector(self, vec, band_idx):
        """Compute the hash signature for a vector in a given band.
        Projects onto k random hyperplanes and takes the sign."""
        projections = self.hyperplanes[band_idx] @ vec  # shape: (k,)
        # Convert signs to a binary string (hashable key)
        bits = tuple((projections > 0).astype(int))
        return bits
    
    def index(self, vectors, ids=None):
        """Insert a batch of vectors into all L hash tables."""
        n = len(vectors)
        if ids is None:
            ids = list(range(n))
        
        for i, (vec, vid) in enumerate(zip(vectors, ids)):
            self.data[vid] = vec
            for band_idx in range(self.L):
                bucket_key = self._hash_vector(vec, band_idx)
                self.tables[band_idx][bucket_key].append(vid)
    
    def query(self, vec, top_k=5):
        """Find approximate nearest neighbors by cosine similarity."""
        # Phase 1: Collect candidates from all L tables
        candidates = set()
        for band_idx in range(self.L):
            bucket_key = self._hash_vector(vec, band_idx)
            candidates.update(self.tables[band_idx][bucket_key])
        
        if not candidates:
            return [], 0
        
        # Phase 2: Compute exact cosine similarity for candidates only
        scores = []
        for cid in candidates:
            cv = self.data[cid]
            # Cosine similarity = dot(a,b) / (|a| * |b|)
            cos_sim = np.dot(vec, cv) / (np.linalg.norm(vec) * np.linalg.norm(cv) + 1e-10)
            scores.append((cid, cos_sim))
        
        # Sort by similarity (descending) and return top_k
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k], len(candidates)


def brute_force_cosine(query, data, top_k=5):
    """Exact nearest neighbors by cosine similarity (baseline)."""
    norms = np.linalg.norm(data, axis=1)
    query_norm = np.linalg.norm(query)
    similarities = data @ query / (norms * query_norm + 1e-10)
    top_indices = np.argsort(-similarities)[:top_k]
    return [(idx, similarities[idx]) for idx in top_indices]


# --- Demo ---
np.random.seed(42)
N = 10000       # number of vectors
DIM = 128       # dimensionality
TOP_K = 5

# Generate random data
data = np.random.randn(N, DIM).astype(np.float32)

# Build LSH index: L=10 bands, k=8 bits per band
lsh = SimHashLSH(dim=DIM, num_bands=10, band_width=8)

start = time.time()
lsh.index(data)
index_time = time.time() - start
print(f"Indexed {N} vectors in {index_time:.3f}s")

# Query with a random vector
query = np.random.randn(DIM).astype(np.float32)

start = time.time()
lsh_results, num_candidates = lsh.query(query, top_k=TOP_K)
lsh_time = time.time() - start

start = time.time()
exact_results = brute_force_cosine(query, data, top_k=TOP_K)
exact_time = time.time() - start

# Compare results
print(f"\nLSH query: {lsh_time*1000:.2f}ms, candidates examined: {num_candidates}/{N}")
print(f"Brute force: {exact_time*1000:.2f}ms")
print(f"\nLSH Top-{TOP_K}:")
for vid, sim in lsh_results:
    print(f"  id={vid:5d}, cosine_sim={sim:.4f}")

print(f"\nExact Top-{TOP_K}:")
for vid, sim in exact_results:
    print(f"  id={vid:5d}, cosine_sim={sim:.4f}")

# Check recall: how many of the true top-k did LSH find?
exact_ids = set(vid for vid, _ in exact_results)
lsh_ids = set(vid for vid, _ in lsh_results)
recall = len(exact_ids & lsh_ids) / TOP_K
print(f"\nRecall@{TOP_K}: {recall:.0%}")
print(f"Speedup: {exact_time/lsh_time:.1f}x (with {num_candidates/N:.1%} of data examined)")
```

## Performance Analysis

LSH provides sub-linear query time for approximate nearest neighbor search, with tunable trade-offs between accuracy (recall) and speed. The key parameters are L (number of hash tables/bands) and k (hash bits per band). Increasing L improves recall but increases both space and query time linearly. Increasing k reduces the number of candidates (faster queries) but may miss true neighbors. The theoretical guarantee for (1+ε)-approximate nearest neighbors in d dimensions requires O(n^(1/(1+ε))) query time with O(n^(1+1/(1+ε))) space, using the standard LSH framework. In practice, careful parameter tuning yields 90%+ recall while examining only 1-5% of the dataset.

| Metric | Complexity |
|--------|------------|
| Best case | `O(L·k·d) — query hashing cost when buckets are small` |
| Average case | `O(L·k·d + L·n^(1/(1+ε))·d) — hashing plus candidate verification` |
| Worst case | `O(n·d) — degenerates to brute force if all points land in same buckets` |
| Space | `O(n·L + L·k·d) — storing n items across L hash tables plus the hash functions` |

## Use Cases

- Near-duplicate detection: Google uses SimHash/LSH to detect near-duplicate web pages during crawling, avoiding redundant indexing of billions of pages that differ only in boilerplate or ads.
- Recommendation systems: Spotify and similar services use LSH to find songs/users with similar feature vectors in embedding spaces, enabling real-time recommendation with millions of items.
- Large-scale image retrieval: Given a query image, LSH on deep learning feature vectors enables sub-second similar image search across databases of billions of images (used in Google Images, Pinterest Lens).
- Genome sequence alignment: MinHash-based LSH (as in the Mash tool) rapidly identifies similar genomic sequences from massive metagenomic datasets, reducing comparison time from days to minutes.
- Entity resolution and record linkage: LSH on text shingles helps identify matching records across large databases (e.g., matching customer records across systems) without comparing every pair.
- Anomaly detection in network security: LSH can quickly find traffic patterns similar to known attack signatures across high-dimensional feature spaces in real-time network monitoring.

## References & Further Reading

- [Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality](https://dl.acm.org/doi/10.1145/276698.276876) — Piotr Indyk, Rajeev Motwani  
  The foundational 1998 STOC paper that introduced the LSH framework for approximate nearest neighbor search.
- [Similarity Estimation Techniques from Rounding Algorithms](https://dl.acm.org/doi/10.1145/509907.509965) — Moses Charikar  
  Introduces SimHash for cosine similarity using random hyperplane rounding, connecting LSH to semidefinite programming.
- [Mining of Massive Datasets — Chapter 3: Finding Similar Items](http://www.mmds.org/) — Jure Leskovec, Anand Rajaraman, Jeffrey D. Ullman  
  An excellent, freely available textbook chapter covering MinHash, LSH, and banding techniques with practical examples.
- [Locality-Sensitive Hashing Scheme Based on p-Stable Distributions](https://dl.acm.org/doi/10.1145/997817.997857) — Mayur Datar, Nicole Immorlica, Piotr Indyk, Vahab Mirrokni  
  Extends LSH to Euclidean distance using projections onto p-stable distributions, a widely used variant.
- [Locality-Sensitive Hashing — Wikipedia](https://en.wikipedia.org/wiki/Locality-sensitive_hashing)  
  A comprehensive overview of LSH families, amplification techniques, and applications with links to key papers.
- [Mash: fast genome and metagenome distance estimation using MinHash](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-0997-x) — Brian D. Ondov, Todd J. Treangen, Páll Melsted, Adam B. Mallonee, Nicholas H. Bergman, Sergey Koren, Adam M. Phillippy  
  Demonstrates practical application of MinHash/LSH in bioinformatics for rapid genome distance estimation.
