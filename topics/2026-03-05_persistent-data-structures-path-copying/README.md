# Persistent Data Structures (Path Copying)

*Algorithm of the Day · 2026-03-05*

## Introduction

Imagine you're building a collaborative text editor, a version control system, or an undo/redo mechanism. You need to keep track of every previous version of your data structure so users can jump back to any point in history. The naive approach — deep-copying the entire structure on every modification — is prohibitively expensive. Persistent data structures solve this elegantly by sharing unchanged portions between versions, giving you full version history at a fraction of the cost.

A data structure is called **persistent** if it preserves all previous versions of itself when modified. Instead of mutating in place (ephemeral), each update operation produces a new version while keeping the old one intact. The most intuitive and widely-used technique for achieving persistence is **path copying**, which works beautifully on tree-based structures. When you modify a node, you create a copy of that node and every ancestor on the path back to the root, while leaving all other nodes shared between the old and new versions.

Path copying is the backbone of functional programming languages' standard libraries (Clojure's vectors, Haskell's `Data.Map`, Scala's immutable collections) and underpins powerful techniques in competitive programming and systems design. Understanding it unlocks a mental model where immutability is not a limitation but a superpower — enabling lock-free concurrency, trivial snapshots, and elegant temporal queries.

## How It Works

The core idea is deceptively simple. Consider a balanced binary search tree (BST). In an ephemeral BST, an insertion or deletion mutates nodes in place. In a persistent BST using path copying, we never mutate. Instead, when we insert a value, we create new copies of only the nodes along the root-to-leaf path that the insertion touches. Every other node in the tree is shared between the old version and the new version. Since a balanced BST has O(log n) depth, each update creates only O(log n) new nodes.

*Path copying after inserting key 6 into Version 0 of a BST*

```
  Version 0 (root_v0)          Version 1 (root_v1)
  ================          ===================

        [5]                       [5']  <-- new copy
       /   \                     /   \
     [3]   [8]                [3]   [8'] <-- new copy
     / \   / \                / \   / \
   [1] [4][7] [9]           [1][4][7'][9]
                                  /         
                                [6] <-- new node

  Nodes [3], [1], [4], [9] are SHARED between versions.
  Only nodes on the path from root to insertion point are copied.
  root_v0 still points to the original [5], preserving Version 0.
```

Each version of the data structure is identified by its root pointer. We maintain an array (or list) of root pointers — `roots[0]`, `roots[1]`, `roots[2]`, etc. To query version `k`, we simply start traversal from `roots[k]`. To create version `k+1`, we perform path copying starting from `roots[k]` and store the new root. This means any version can be the basis for creating a new version, naturally forming a version tree (not just a linear history).

The technique generalizes beyond BSTs. It works on any pointer-based tree structure: tries, segment trees, heaps, and even more complex balanced trees like red-black trees or AVL trees. The key requirement is that the structure is tree-shaped (each node has exactly one parent), so that the 'path to root' is well-defined and bounded. For a balanced tree of n elements, each update copies O(log n) nodes, and each query takes the same O(log n) time as the ephemeral version.

*Version history as a DAG of shared nodes*

```
  roots[]:
  +---+---+---+---+
  | 0 | 1 | 2 | 3 |   <-- array of root pointers
  +---+---+---+---+
    |   |   |   |
    v   v   v   v
   r0  r1  r2  r3      <-- each root is a separate tree
    \  /\  |  /         but they share most internal nodes
     \/  \ | /
      \   \|/
    [shared subtrees]

  Total memory: O(n + m * log n)
    n = initial elements
    m = number of updates
```

A particularly powerful application is the **persistent segment tree**, widely used in competitive programming. A segment tree on an array of n elements has O(n) nodes. Each point update touches O(log n) nodes along a root-to-leaf path. By path copying, we get a fully persistent segment tree where we can query any historical version's range sum, range min, or other aggregate in O(log n) time. This enables solving problems like 'find the k-th smallest element in an arbitrary subarray' by building a persistent segment tree over sorted value insertions and differencing two versions.

## Example

Let's implement a persistent balanced BST (using a simple treap-like approach with random priorities for balance) that supports insert and lookup across versions. We'll build several versions by inserting values, then demonstrate querying different historical versions to show that old versions remain intact.

The implementation stores an explicit list of roots. Each insert creates a new version by path-copying along the insertion path. We use a simple recursive approach where each modification returns a new node rather than mutating the existing one.

```python
import random
from typing import Optional, List, Tuple

class Node:
    """Immutable node for a persistent treap (BST + heap priorities for balance)."""
    __slots__ = ('key', 'priority', 'left', 'right', 'size')
    
    def __init__(self, key: int, priority: float,
                 left: Optional['Node'] = None,
                 right: Optional['Node'] = None):
        self.key = key
        self.priority = priority
        self.left = left
        self.right = right
        self.size = 1 + _size(left) + _size(right)

def _size(node: Optional[Node]) -> int:
    return node.size if node else 0

def split(node: Optional[Node], key: int) -> Tuple[Optional[Node], Optional[Node]]:
    """Split tree into (<=key, >key). Returns NEW nodes on the path (path copying)."""
    if node is None:
        return (None, None)
    if key >= node.key:
        # node goes to the left result; split right subtree
        left_part, right_part = split(node.right, key)
        # Create a NEW node instead of mutating (path copying!)
        new_node = Node(node.key, node.priority, node.left, left_part)
        return (new_node, right_part)
    else:
        left_part, right_part = split(node.left, key)
        new_node = Node(node.key, node.priority, right_part, node.right)
        return (left_part, new_node)

def merge(left: Optional[Node], right: Optional[Node]) -> Optional[Node]:
    """Merge two treaps. All keys in left < all keys in right. Returns NEW root."""
    if left is None:
        return right
    if right is None:
        return left
    if left.priority > right.priority:
        # left root wins; merge left's right child with right
        merged = merge(left.right, right)
        return Node(left.key, left.priority, left.left, merged)
    else:
        merged = merge(left, right.left)
        return Node(right.key, right.priority, merged, right.right)

class PersistentBST:
    """A fully persistent BST using path copying via treap split/merge."""
    
    def __init__(self):
        self.roots: List[Optional[Node]] = [None]  # Version 0 is empty
    
    def insert(self, key: int, version: int = -1) -> int:
        """Insert key based on a given version. Returns the new version number."""
        if version == -1:
            version = len(self.roots) - 1  # default: latest version
        root = self.roots[version]
        # Split around the key, create new node, merge back
        left, right = split(root, key - 1)  # left has keys < key
        new_node = Node(key, random.random(), None, None)
        merged = merge(merge(left, new_node), right)
        self.roots.append(merged)
        return len(self.roots) - 1
    
    def search(self, key: int, version: int = -1) -> bool:
        """Search for key in a specific version."""
        if version == -1:
            version = len(self.roots) - 1
        node = self.roots[version]
        while node:
            if key == node.key:
                return True
            elif key < node.key:
                node = node.left
            else:
                node = node.right
        return False
    
    def inorder(self, version: int = -1) -> List[int]:
        """Return sorted keys of a specific version."""
        if version == -1:
            version = len(self.roots) - 1
        result = []
        def _traverse(node):
            if node:
                _traverse(node.left)
                result.append(node.key)
                _traverse(node.right)
        _traverse(self.roots[version])
        return result

# --- Demo ---
random.seed(42)
bst = PersistentBST()

# Build versions by sequential inserts
v1 = bst.insert(5)   # Version 1: {5}
v2 = bst.insert(3)   # Version 2: {3, 5}
v3 = bst.insert(8)   # Version 3: {3, 5, 8}
v4 = bst.insert(1)   # Version 4: {1, 3, 5, 8}
v5 = bst.insert(6)   # Version 5: {1, 3, 5, 6, 8}

# Branch from version 2 to create an alternate history
v6 = bst.insert(10, version=2)  # Version 6: {3, 5, 10}

print("All versions are preserved:")
for v in range(len(bst.roots)):
    print(f"  Version {v}: {bst.inorder(v)}")

print(f"\nSearch for 8 in version 3: {bst.search(8, 3)}")   # True
print(f"Search for 8 in version 2: {bst.search(8, 2)}")     # False (not yet inserted)
print(f"Search for 10 in version 6: {bst.search(10, 6)}")   # True (alternate branch)
print(f"Search for 10 in version 5: {bst.search(10, 5)}")   # False (main branch)

# Output:
# All versions are preserved:
#   Version 0: []
#   Version 1: [5]
#   Version 2: [3, 5]
#   Version 3: [3, 5, 8]
#   Version 4: [1, 3, 5, 8]
#   Version 5: [1, 3, 5, 6, 8]
#   Version 6: [3, 5, 10]
```

## Performance Analysis

For a balanced tree of n elements, path copying adds O(log n) overhead per update (creating that many new nodes). Queries are unchanged at O(log n). The space cost is O(n) for the initial structure plus O(log n) per update, since each update only allocates new nodes along one root-to-leaf path. Over m updates, total space is O(n + m log n). This is a dramatic improvement over the O(n * m) cost of full copying. The technique preserves all the time complexity guarantees of the underlying ephemeral data structure.

| Metric | Complexity |
|--------|------------|
| Best case | `O(log n) per insert/search` |
| Average case | `O(log n) per insert/search` |
| Worst case | `O(log n) per insert/search (with balanced tree guarantee)` |
| Space | `O(n + m log n) for n initial elements and m updates` |

## Use Cases

- Version control systems and undo/redo: Each edit creates a new version of the document's index structure, with old versions trivially accessible for diff, blame, or rollback operations.
- Functional programming language runtimes: Languages like Clojure, Haskell, and Scala use persistent data structures as their default collections, enabling safe concurrent access without locks since data is never mutated.
- Persistent segment trees in competitive programming: Solving problems like 'k-th smallest in a subarray' or 'count of values in a range across historical states' by maintaining O(n log n) versions of a segment tree.
- Database MVCC (Multi-Version Concurrency Control): Databases like PostgreSQL use persistence concepts to allow readers to see consistent snapshots while writers create new versions, avoiding read-write locks.
- Retroactive/temporal data structures: Enabling queries like 'what was the state of this data structure at time t?' which arise in event sourcing architectures, audit logging, and time-travel debugging.

## References & Further Reading

- [Making Data Structures Persistent](https://www.cs.cmu.edu/~sleator/papers/making-data-structures-persistent.pdf) — James R. Driscoll, Neil Sarnak, Daniel D. Sleator, Robert E. Tarjan  
  The foundational 1986 paper that formalizes partial and full persistence via path copying and the node-copying technique.
- **Purely Functional Data Structures** — Chris Okasaki  
  The definitive book on persistent and purely functional data structures, covering amortization, lazy evaluation, and numerous persistent designs.
- [Persistent Data Structures — Wikipedia](https://en.wikipedia.org/wiki/Persistent_data_structure)  
  A comprehensive overview of partial, full, and confluent persistence with links to key papers and implementations.
- [MIT 6.851: Advanced Data Structures — Lecture 1: Persistent Data Structures](https://courses.csail.mit.edu/6.851/spring21/lectures/L01.html) — Erik Demaine  
  Erik Demaine's graduate lecture covering the fat node method, path copying, and the Driscoll et al. node-copying technique for O(1) amortized persistence.
- [Clojure's Persistent Vectors (PersistentVector.java)](https://github.com/clojure/clojure/blob/master/src/jvm/clojure/lang/PersistentVector.java) — Rich Hickey  
  Real-world production implementation of a persistent vector using a 32-way branching trie with path copying, achieving near-O(1) operations.
- [Persistent Segment Tree — CP-Algorithms](https://cp-algorithms.com/data_structures/segment_tree.html)  
  Practical guide to implementing persistent segment trees for competitive programming, including the k-th smallest element problem.
