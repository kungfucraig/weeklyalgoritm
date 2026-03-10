# Disjoint Set Union (Union-Find)

*Algorithm of the Day · 2026-03-10*

## Introduction

Disjoint Set Union, universally known as Union-Find, is one of the most elegant data structures in computer science. It solves a deceptively simple problem: given a collection of elements partitioned into non-overlapping sets, how do you efficiently determine whether two elements belong to the same set, and how do you merge two sets together? Despite the simplicity of the problem statement, the solution — when augmented with two key optimizations — achieves an amortized time complexity so close to O(1) that it took decades to fully analyze.

The structure was first studied in the late 1960s, but it was Robert Tarjan's landmark 1975 analysis that revealed Union-Find operations run in amortized O(α(n)) time, where α is the inverse Ackermann function — a function that grows so extraordinarily slowly that for any input size conceivable in the physical universe, its value never exceeds 4. This makes Union-Find effectively constant-time per operation in practice, which is remarkable for a structure that manages dynamic equivalence classes.

Union-Find is a workhorse in graph algorithms (most notably Kruskal's minimum spanning tree algorithm), network connectivity analysis, image processing, and even in compilers for type inference and equivalence checking. If you've ever needed to track connected components as edges are added to a graph, Union-Find is almost certainly the right tool. Its combination of conceptual simplicity, ease of implementation, and blazing performance makes it one of the most useful data structures a working engineer can have in their toolkit.

## How It Works

Union-Find maintains a forest of trees, where each tree represents one disjoint set. Every element points to a parent, and the root of each tree serves as the canonical representative (or 'leader') of that set. Initially, every element is its own root — a forest of singleton trees. The two fundamental operations are Find(x), which follows parent pointers from x up to the root to identify which set x belongs to, and Union(x, y), which merges the sets containing x and y by linking one root to the other.

*Initial state: 6 elements, each in its own set*

```
parent[]:  [0, 1, 2, 3, 4, 5]

  (0)   (1)   (2)   (3)   (4)   (5)

Each element is its own root (parent[i] = i)
```

The naive version of Union-Find can degenerate into long chains (essentially linked lists), making Find take O(n) time. Two optimizations fix this. The first is Union by Rank (or Union by Size): when merging two trees, always attach the smaller (or shallower) tree under the root of the larger one. This ensures that tree heights grow logarithmically. The rank of a node is an upper bound on its height, and it only increases when two trees of equal rank are merged.

*Union by Rank: merging set {0,1,2} and set {3,4}*

```
Before Union(2, 4):

    (0)  rank=1          (3)  rank=1
   / \                   |
 (1) (2)                (4)

After Union(2, 4):  attach smaller root (3) under larger root (0)

        (0)  rank=1
       / | \
     (1)(2)(3)
             |
            (4)

(Rank stays 1 because ranks were equal? No — ranks were equal,
 so rank of new root increases to 2)

        (0)  rank=2
       / | \
     (1)(2)(3)
             |
            (4)
```

The second optimization is Path Compression: during a Find operation, after walking up to the root, we make every node along the path point directly to the root. This flattens the tree dramatically, so future Find operations on any of those nodes will be nearly instant. Path compression alone gives amortized O(log n) per operation, but combined with union by rank, the amortized cost drops to O(α(n)) — the inverse Ackermann function. Since α(n) ≤ 4 for n up to approximately 10^{80} (more atoms than exist in the observable universe), this is effectively constant.

*Path Compression during Find(5)*

```
Before Find(5):               After Find(5):

    (0)                            (0)
     |                          / | | \
    (1)                       (1)(2)(3)(5)
     |                             |
    (2)                           (4)
     |                  
    (3)                 All nodes on the path to root
   / \                  now point directly to (0)
 (4) (5)                
```

The implementation is remarkably concise. The entire data structure can be represented with just two arrays: parent[] and rank[] (or size[]). Find is a small recursive or iterative function, and Union calls Find on both arguments and then links the roots. In practice, many competitive programmers and production systems implement Union-Find in under 20 lines of code. Despite this simplicity, the theoretical analysis by Tarjan and later by Seidel and Sharir is among the most intricate in the analysis of algorithms, involving the Ackermann function and its inverse.

## Example

Let's implement Union-Find and use it to solve a classic problem: given a list of edges in an undirected graph, determine how many connected components exist and detect whether adding an edge would create a cycle (which is the core logic in Kruskal's MST algorithm). We'll process edges between 7 nodes (0 through 6) and watch the components merge.

The code below includes both path compression and union by rank, and demonstrates cycle detection: if two nodes already share the same root when we try to union them, that edge would form a cycle.

```python
class UnionFind:
    def __init__(self, n):
        # Each element starts as its own parent (self-loop = root)
        self.parent = list(range(n))
        # Rank is an upper bound on tree height
        self.rank = [0] * n
        # Track number of distinct components
        self.components = n

    def find(self, x):
        """Find root of x with path compression."""
        if self.parent[x] != x:
            # Recursively find root and compress path
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """Merge sets containing x and y. Returns False if already same set (cycle)."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False  # Already in same set — edge would create a cycle

        # Union by rank: attach smaller tree under larger tree's root
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        self.components -= 1
        return True

    def connected(self, x, y):
        """Check if x and y are in the same set."""
        return self.find(x) == self.find(y)


# --- Demo: process edges and detect cycles ---
if __name__ == "__main__":
    n_nodes = 7
    edges = [(0, 1), (1, 2), (3, 4), (4, 5), (5, 6), (2, 3), (0, 5)]

    uf = UnionFind(n_nodes)
    mst_edges = []

    print(f"Starting with {uf.components} components\n")

    for u, v in edges:
        if uf.union(u, v):
            mst_edges.append((u, v))
            print(f"Union({u}, {v}): merged -> {uf.components} components")
        else:
            print(f"Union({u}, {v}): CYCLE detected (same component), skipped")

    print(f"\nFinal components: {uf.components}")
    print(f"MST edges (if graph were weighted): {mst_edges}")
    print(f"Are 0 and 6 connected? {uf.connected(0, 6)}")
    print(f"\nInternal parent array: {uf.parent}")
    print(f"Internal rank array:   {uf.rank}")
```

## Performance Analysis

With both path compression and union by rank, a sequence of m operations (any mix of Union and Find) on n elements takes O(m · α(n)) total time, where α is the inverse Ackermann function. Since α(n) ≤ 4 for all practical input sizes (n < 10^80), each operation is effectively O(1) amortized. Without optimizations, Find can degrade to O(n) if the tree becomes a long chain. Fredman and Sacks proved in 1989 that the O(m · α(n)) bound is tight — no pointer-based Union-Find structure can do better. Space is always O(n) for the parent and rank arrays.

| Metric | Complexity |
|--------|------------|
| Best case | `O(1) — Find on a root or recently compressed node` |
| Average case | `O(α(n)) amortized per operation` |
| Worst case | `O(log n) per single operation (amortized over many operations: O(α(n)))` |
| Space | `O(n)` |

## Use Cases

- Kruskal's Minimum Spanning Tree Algorithm — Union-Find efficiently determines whether adding the next cheapest edge would create a cycle, enabling the greedy MST construction in O(E log E) time.
- Dynamic connectivity in networks — tracking whether two servers, routers, or users are in the same connected component as links are added (e.g., in social network 'friend of a friend' queries or network topology monitoring).
- Image processing and percolation — labeling connected regions of pixels with the same color or intensity (connected component labeling), widely used in computer vision, medical imaging, and physics simulations of percolation.
- Equivalence class merging in compilers — unifying type variables during type inference (e.g., in Hindley-Milner type systems used by ML, Haskell, and Rust's trait solver) where type constraints create equivalence relations.
- Least Common Ancestor (offline) — Tarjan's offline LCA algorithm uses Union-Find to batch-answer ancestor queries on trees in nearly linear time.
- Cycle detection in undirected graphs — determining if adding an edge creates a cycle, useful in circuit design, dependency resolution, and constraint satisfaction problems.

## References & Further Reading

- **Efficiency of a Good But Not Linear Set Union Algorithm** — Robert Endre Tarjan  
  The 1975 landmark paper establishing the O(m · α(n)) amortized bound for Union-Find with path compression and union by rank.
- [Disjoint-set data structure — Wikipedia](https://en.wikipedia.org/wiki/Disjoint-set_data_structure)  
  Comprehensive overview of Union-Find including pseudocode, complexity proofs, and historical context.
- **Introduction to Algorithms (CLRS), Chapter 21: Data Structures for Disjoint Sets** — Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein  
  The standard textbook treatment of Union-Find with rigorous amortized analysis using the Ackermann function.
- [Algorithms, 4th Edition — Union-Find](https://algs4.cs.princeton.edu/15uf/) — Robert Sedgewick, Kevin Wayne  
  Accessible introduction with Java implementations, visualizations, and case studies including percolation.
- **On the Inverse Ackermann Function** — Raimund Seidel, Micha Sharir  
  A simplified and refined analysis of the Union-Find complexity bound, making Tarjan's original proof more accessible.
- [CP-Algorithms: Disjoint Set Union](https://cp-algorithms.com/data_structures/disjoint_set_union.html)  
  Practical guide with C++ implementations covering weighted Union-Find, rollback, and competitive programming applications.
