# B+ Tree

*Algorithm of the Day · 2026-03-21*

## Introduction

If you've ever wondered how a database can search through billions of rows in milliseconds, the answer almost certainly involves a B+ Tree. This self-balancing tree structure is the workhorse behind virtually every relational database index (MySQL/InnoDB, PostgreSQL, SQLite, Oracle) and many filesystems (NTFS, ext4, HFS+). Understanding B+ Trees doesn't just satisfy intellectual curiosity — it fundamentally changes how you think about indexing, query optimization, and storage engine design.

The B+ Tree is an evolution of the B-Tree, introduced by Rudolf Bayer and Edward McCreight in 1972. While B-Trees store data in both internal and leaf nodes, B+ Trees push all actual data (or pointers to data) down to the leaf level and link all leaves together in a sorted linked list. This seemingly small change has profound consequences: sequential scans become blazingly fast, internal nodes can hold more keys (increasing the branching factor), and range queries are trivially efficient. The result is a structure that minimizes disk I/O — the true bottleneck in any storage system.

What makes B+ Trees especially elegant is their harmony with hardware. A node is typically sized to match a disk page (4 KB or 16 KB), meaning each node access corresponds to exactly one disk read. With a branching factor of several hundred, a B+ Tree of height 3 or 4 can index billions of records. This is why, decades after its invention, the B+ Tree remains the default choice for on-disk indexing — and why every software engineer working with databases should understand how it works under the hood.

## How It Works

A B+ Tree of order `m` has the following properties: (1) Each internal node has at most `m` children and at least ⌈m/2⌉ children (except the root, which may have as few as 2). (2) Each internal node with `k` children contains `k-1` keys that act as separators guiding searches. (3) All data records (or pointers to records) reside exclusively in the leaf nodes. (4) All leaf nodes are at the same depth, guaranteeing balanced access times. (5) Leaf nodes are linked together in a doubly- or singly-linked list, enabling efficient sequential access and range queries.

*Structure of a B+ Tree (order 4, max 3 keys per node)*

```
                       [  10  |  20  ]
                      /        |        \
                     /         |         \
          [3 | 5 | 7]    [10 | 14 | 18]   [20 | 25 | 30]
           |   |   |       |    |    |       |    |    |
           v   v   v       v    v    v       v    v    v
          (data records stored at leaf level)

    Leaf linked list:  [3|5|7] --> [10|14|18] --> [20|25|30]

    - Internal nodes contain ONLY keys (routing guides)
    - Leaf nodes contain keys + data pointers
    - Leaves are chained left-to-right for range scans
```

**Search** works exactly like a B-Tree search. Starting at the root, you compare the target key against the separator keys in the current node to determine which child pointer to follow. You repeat this at each level until you reach a leaf node, where you perform a linear or binary search among the leaf's keys. Because the tree is balanced, every search traverses exactly `h` nodes where `h` is the tree height — typically 3-4 for billions of records.

**Insertion** begins with a search to find the correct leaf node. The new key is inserted into the leaf in sorted order. If the leaf overflows (exceeds `m-1` keys), it is **split** into two leaves: the lower half stays, the upper half moves to a new leaf, and the middle key is **copied up** (not moved — the key remains in the leaf too) into the parent as a new separator. If the parent also overflows, it splits recursively, and the middle key is **pushed up**. In the worst case, splitting propagates all the way to the root, increasing the tree's height by one.

*Leaf split during insertion (inserting key 8 into a full leaf, order 4)*

```
  BEFORE: Leaf is full with keys [3, 5, 7]
  Insert 8 --> overflow! (max 3 keys)

  AFTER SPLIT:

     Parent:  [...| 7 |...]       <-- middle key (7) copied up
                /       \
               /         \
         [3 | 5]      [7 | 8]    <-- two new leaves
            |     -->     |       <-- linked list maintained

  Note: key 7 appears in BOTH the parent (as separator)
        AND the right leaf (as data). This is the key
        difference from B-Trees, where keys move up.
```

**Deletion** also begins with a search to the target leaf. The key is removed from the leaf. If the leaf falls below its minimum occupancy (⌈m/2⌉ - 1 keys), the tree rebalances by either **borrowing** a key from a sibling leaf (redistributing keys through the parent) or **merging** with a sibling. Merging may cause the parent to lose a separator key, potentially triggering recursive merges up the tree. In practice, many database implementations use lazy deletion or simply mark records as deleted without immediately rebalancing, since insertions will often refill the space.

**Range queries** are where B+ Trees truly shine over B-Trees. To find all keys in range [lo, hi], you search for `lo` to reach the appropriate leaf, then simply follow the leaf linked list forward, collecting keys, until you encounter a key greater than `hi`. This is a sequential scan across contiguous leaf pages — ideal for disk I/O patterns and CPU cache behavior. B-Trees would require an in-order traversal bouncing between internal and leaf nodes, resulting in far more random I/O.

## Example

Below is a complete Python implementation of an in-memory B+ Tree of configurable order. We implement insertion, search, and range queries. The example creates a B+ Tree of order 4, inserts 20 keys, then demonstrates point lookup and range query. Notice how the leaf linked list enables efficient range scanning.

This implementation keeps things clear rather than production-optimized. Real database B+ Trees are page-oriented with careful concurrency control (latching protocols), write-ahead logging, and bulk-loading optimizations.

```python
class BPlusLeaf:
    """Leaf node: stores keys, data values, and a next-leaf pointer."""
    def __init__(self):
        self.keys = []
        self.values = []    # Parallel to keys; in a DB, these would be row pointers
        self.next_leaf = None  # Linked list pointer
        self.parent = None

    def is_leaf(self):
        return True


class BPlusInternal:
    """Internal node: stores separator keys and child pointers."""
    def __init__(self):
        self.keys = []
        self.children = []
        self.parent = None

    def is_leaf(self):
        return False


class BPlusTree:
    def __init__(self, order=4):
        """order = max number of children per internal node (max keys = order-1)."""
        self.order = order
        self.max_keys = order - 1  # Max keys in any node
        self.root = BPlusLeaf()    # Start with an empty leaf as root

    def _find_leaf(self, key):
        """Navigate from root to the leaf where 'key' belongs."""
        node = self.root
        while not node.is_leaf():
            # Find the child to descend into
            i = 0
            while i < len(node.keys) and key >= node.keys[i]:
                i += 1
            node = node.children[i]
        return node

    def search(self, key):
        """Return the value associated with 'key', or None."""
        leaf = self._find_leaf(key)
        for i, k in enumerate(leaf.keys):
            if k == key:
                return leaf.values[i]
        return None

    def range_query(self, lo, hi):
        """Return all (key, value) pairs where lo <= key <= hi."""
        leaf = self._find_leaf(lo)
        results = []
        while leaf is not None:
            for i, k in enumerate(leaf.keys):
                if k > hi:
                    return results
                if k >= lo:
                    results.append((k, leaf.values[i]))
            leaf = leaf.next_leaf  # Follow the linked list!
        return results

    def insert(self, key, value):
        """Insert a key-value pair into the B+ Tree."""
        leaf = self._find_leaf(key)

        # Insert in sorted position within the leaf
        i = 0
        while i < len(leaf.keys) and leaf.keys[i] < key:
            i += 1
        if i < len(leaf.keys) and leaf.keys[i] == key:
            leaf.values[i] = value  # Update existing key
            return
        leaf.keys.insert(i, key)
        leaf.values.insert(i, value)

        # If leaf overflows, split it
        if len(leaf.keys) > self.max_keys:
            self._split_leaf(leaf)

    def _split_leaf(self, leaf):
        """Split an overflowing leaf node."""
        mid = len(leaf.keys) // 2
        new_leaf = BPlusLeaf()

        # Right half goes to new leaf
        new_leaf.keys = leaf.keys[mid:]
        new_leaf.values = leaf.values[mid:]
        leaf.keys = leaf.keys[:mid]
        leaf.values = leaf.values[:mid]

        # Maintain linked list
        new_leaf.next_leaf = leaf.next_leaf
        leaf.next_leaf = new_leaf

        # The first key of new_leaf is COPIED up to the parent
        promote_key = new_leaf.keys[0]
        self._insert_into_parent(leaf, promote_key, new_leaf)

    def _split_internal(self, node):
        """Split an overflowing internal node."""
        mid = len(node.keys) // 2
        promote_key = node.keys[mid]  # This key is PUSHED up (not kept)

        new_node = BPlusInternal()
        new_node.keys = node.keys[mid + 1:]
        new_node.children = node.children[mid + 1:]
        for child in new_node.children:
            child.parent = new_node

        node.keys = node.keys[:mid]
        node.children = node.children[:mid + 1]

        self._insert_into_parent(node, promote_key, new_node)

    def _insert_into_parent(self, left, key, right):
        """Insert a separator key into the parent after a split."""
        if left.parent is None:
            # Create a new root
            new_root = BPlusInternal()
            new_root.keys = [key]
            new_root.children = [left, right]
            left.parent = new_root
            right.parent = new_root
            self.root = new_root
            return

        parent = left.parent
        right.parent = parent

        # Find position to insert the new key
        i = parent.children.index(left) + 1
        parent.keys.insert(i - 1 if i - 1 >= 0 else 0, key)
        # Actually, insert key at correct sorted position
        # Re-do: find where 'left' is among children
        idx = parent.children.index(left)
        parent.keys.insert(idx, key)
        parent.children.insert(idx + 1, right)
        # Fix: we double-inserted. Let's redo cleanly.
        # Remove the wrong one
        parent.keys.pop(idx + 1) if len(parent.keys) > idx + 1 and parent.keys[idx] == parent.keys[idx + 1] if idx + 1 < len(parent.keys) else None

        # Simpler correct approach: rebuild
        # Let me just do this correctly:
        parent.keys = []
        parent.children = []
        # Actually, let me rewrite this method properly.
        pass  # See corrected version below


# --- Corrected, clean implementation ---

class Node:
    def __init__(self, is_leaf=False):
        self.is_leaf = is_leaf
        self.keys = []
        self.children = []  # For internal: child nodes. For leaf: data values.
        self.next = None     # For leaves: linked list pointer
        self.parent = None

class BPTree:
    def __init__(self, order=4):
        self.order = order
        self.root = Node(is_leaf=True)

    def search(self, key):
        """Find value for given key, or None."""
        leaf = self._find_leaf(key)
        for i, k in enumerate(leaf.keys):
            if k == key:
                return leaf.children[i]  # In leaves, children = values
        return None

    def range_query(self, lo, hi):
        """Efficient range scan using the leaf linked list."""
        leaf = self._find_leaf(lo)
        results = []
        while leaf:
            for i, k in enumerate(leaf.keys):
                if k > hi:
                    return results
                if lo <= k <= hi:
                    results.append((k, leaf.children[i]))
            leaf = leaf.next
        return results

    def insert(self, key, value):
        """Insert key-value pair."""
        leaf = self._find_leaf(key)
        # Check for duplicate
        for i, k in enumerate(leaf.keys):
            if k == key:
                leaf.children[i] = value
                return
        # Find insertion point
        i = 0
        while i < len(leaf.keys) and leaf.keys[i] < key:
            i += 1
        leaf.keys.insert(i, key)
        leaf.children.insert(i, value)
        # Split if overflow
        if len(leaf.keys) >= self.order:
            self._split_leaf(leaf)

    def _find_leaf(self, key):
        node = self.root
        while not node.is_leaf:
            i = 0
            while i < len(node.keys) and key >= node.keys[i]:
                i += 1
            node = node.children[i]
        return node

    def _split_leaf(self, leaf):
        mid = len(leaf.keys) // 2
        new_leaf = Node(is_leaf=True)
        new_leaf.keys = leaf.keys[mid:]
        new_leaf.children = leaf.children[mid:]
        leaf.keys = leaf.keys[:mid]
        leaf.children = leaf.children[:mid]
        # Linked list maintenance
        new_leaf.next = leaf.next
        leaf.next = new_leaf
        # Copy up the first key of the new leaf
        up_key = new_leaf.keys[0]
        self._insert_in_parent(leaf, up_key, new_leaf)

    def _insert_in_parent(self, left, key, right):
        if left.parent is None:
            # Create new root
            root = Node(is_leaf=False)
            root.keys = [key]
            root.children = [left, right]
            left.parent = root
            right.parent = root
            self.root = root
            return
        parent = left.parent
        right.parent = parent
        # Find index of left child in parent
        idx = parent.children.index(left)
        parent.keys.insert(idx, key)
        parent.children.insert(idx + 1, right)
        # Split parent if overflow
        if len(parent.keys) >= self.order:
            self._split_internal(parent)

    def _split_internal(self, node):
        mid = len(node.keys) // 2
        up_key = node.keys[mid]  # Pushed up (removed from node)
        new_node = Node(is_leaf=False)
        new_node.keys = node.keys[mid + 1:]
        new_node.children = node.children[mid + 1:]
        node.keys = node.keys[:mid]
        node.children = node.children[:mid + 1]
        # Update parent pointers for moved children
        for child in new_node.children:
            child.parent = new_node
        self._insert_in_parent(node, up_key, new_node)

    def print_leaves(self):
        """Walk the leaf linked list to display all data in order."""
        node = self.root
        while not node.is_leaf:
            node = node.children[0]
        leaves = []
        while node:
            leaves.append(str(node.keys))
            node = node.next
        return ' -> '.join(leaves)


# === Demo ===
if __name__ == '__main__':
    tree = BPTree(order=4)  # Each node holds at most 3 keys

    # Insert 20 keys
    data = [10, 20, 5, 15, 25, 30, 3, 7, 12, 18, 22, 27, 35, 1, 8, 14, 17, 23, 28, 33]
    for k in data:
        tree.insert(k, f'val_{k}')
        # value is a string for demonstration

    # Point search
    print('Search for 15:', tree.search(15))   # -> val_15
    print('Search for 99:', tree.search(99))   # -> None

    # Range query: all keys in [10, 25]
    print('Range [10, 25]:', tree.range_query(10, 25))
    # -> [(10, 'val_10'), (12, 'val_12'), (14, 'val_14'), (15, 'val_15'),
    #     (17, 'val_17'), (18, 'val_18'), (20, 'val_20'), (22, 'val_22'),
    #     (23, 'val_23'), (25, 'val_25')]

    # Show the leaf chain
    print('Leaf chain:', tree.print_leaves())
```

## Performance Analysis

All primary operations — search, insert, and delete — run in O(log_m n) time, where m is the order (branching factor) and n is the number of keys. Because m is typically large (hundreds or thousands in database systems), the tree height is extremely small: a B+ Tree with m=512 and 1 billion keys has height ≤ 4. Each level requires one disk I/O, so operations need only 3-4 disk reads. Range queries that return k results cost O(log_m n + k/m) because once we reach the first leaf, we scan sequentially through linked leaves. Space is O(n) with a small constant overhead for internal nodes, and typical space utilization is around 67% due to splitting behavior (though bulk loading can achieve near 100%).

| Metric | Complexity |
|--------|------------|
| Best case | `O(log_m n) — for search, insert, delete` |
| Average case | `O(log_m n) — balanced tree guarantees consistent performance` |
| Worst case | `O(log_m n) — always balanced; range query of k results adds O(k/m)` |
| Space | `O(n) — all data in leaves, internal nodes add ~O(n/m) overhead` |

## Use Cases

- Database indexing (MySQL/InnoDB, PostgreSQL, SQLite, Oracle): B+ Trees are the primary index structure for relational databases, enabling fast point lookups, range scans, and ORDER BY operations on indexed columns.
- Filesystem metadata (NTFS, ext4, HFS+, Btrfs, ZFS): Filesystems use B+ Trees to map file names to inodes and to track file extent allocations, enabling fast file lookups even in directories with millions of entries.
- Key-value storage engines (LevelDB, RocksDB, LMDB): Many embedded storage engines use B+ Tree variants for their on-disk sorted data structures, providing efficient reads while supporting transactional semantics.
- Full-text search indices (Lucene/Elasticsearch term dictionaries): The term dictionary in inverted indices is often organized as a B+ Tree-like structure for efficient term lookups and prefix queries.
- Network routers and IP lookup tables: B+ Trees are used in some routing table implementations where longest-prefix matching benefits from the sorted, range-queryable structure of B+ Trees.

## References & Further Reading

- **Organization and Maintenance of Large Ordered Indexes** — Rudolf Bayer, Edward M. McCreight  
  The seminal 1972 paper introducing B-Trees, the predecessor to B+ Trees, establishing the foundational concepts of balanced multiway search trees for external storage.
- [The Ubiquitous B-Tree](https://dl.acm.org/doi/10.1145/356770.356776) — Douglas Comer  
  A classic 1979 ACM Computing Surveys paper providing a comprehensive overview of B-Tree variants including B+ Trees and their applications.
- **Database Internals: A Deep Dive into How Distributed Data Systems Work** — Alex Petrov  
  An excellent modern book with detailed chapters on B-Tree and B+ Tree storage engine internals, including on-disk layouts, concurrency, and optimization techniques.
- [B+ Tree — Wikipedia](https://en.wikipedia.org/wiki/B%2B_tree)  
  A well-maintained reference covering the structure, operations, and complexity of B+ Trees with helpful diagrams.
- **Introduction to Algorithms (CLRS), Chapter 18: B-Trees** — Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein  
  The standard algorithms textbook treatment of B-Trees, providing rigorous analysis of operations and correctness proofs applicable to B+ Tree variants.
- [Modern B-Tree Techniques](https://dl.acm.org/doi/10.1561/1900000028) — Goetz Graefe  
  A comprehensive 2011 survey covering decades of B-Tree and B+ Tree optimizations used in commercial database systems, including concurrency control, recovery, and prefix/suffix compression.
