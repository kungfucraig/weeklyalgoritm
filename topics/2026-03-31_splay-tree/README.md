# Splay Tree

*Algorithm of the Day · 2026-03-31*

## Introduction

The Splay Tree, invented by Daniel Sleator and Robert Tarjan in 1985, is one of the most elegant self-adjusting binary search trees ever devised. Unlike AVL trees or red-black trees that maintain strict balance invariants, a splay tree takes a radically different approach: every time you access a node, you rotate it all the way to the root through a specific sequence of operations called 'splaying.' There are no balance factors, no color bits, no extra bookkeeping — just a simple restructuring rule applied on every access.

What makes splay trees fascinating is their amortized efficiency guarantee: any sequence of m operations on an n-node splay tree takes O(m log n) total time, giving O(log n) amortized cost per operation — even though individual operations can take O(n) in the worst case. But the real magic goes deeper. Splay trees have a remarkable property called the 'working set' property: if you repeatedly access a small subset of elements, those elements migrate to the top of the tree, making subsequent accesses extremely fast. This means splay trees automatically adapt to your access pattern without any explicit tuning.

This self-optimizing behavior makes splay trees surprisingly practical. They've been used in garbage collectors, network routers, memory allocators, and even inside the Windows NT kernel. The persistent conjecture (still unproven!) that splay trees are 'dynamically optimal' — meaning they perform within a constant factor of any other BST algorithm on every access sequence — keeps them at the frontier of theoretical computer science as well.

## How It Works

A splay tree is a binary search tree where the core operation is 'splaying' — moving a target node to the root via a series of tree rotations. The key insight is that splaying doesn't just use simple single rotations (which can lead to poor amortized performance). Instead, it uses three carefully chosen cases based on the relationship between the target node (x), its parent (p), and its grandparent (g). These cases are: Zig, Zig-Zig, and Zig-Zag.

**Zig (terminal case):** When x's parent p is the root, perform a single rotation to bring x to the root. This only happens as the last step of a splay operation when the tree has an even depth path.

**Zig-Zig:** When x and p are both left children (or both right children) of their respective parents, first rotate p around g, then rotate x around p. This double rotation in the same direction is what distinguishes splaying from naive 'rotate to root' — it has the critical effect of halving the depth of nodes along the access path.

**Zig-Zag:** When x is a left child and p is a right child (or vice versa), rotate x around p, then rotate x around g. This is essentially a double rotation similar to AVL tree rebalancing.

*Zig-Zig case: x and p are both left children*

```
  Before:                 After:

       g                   x
      / \                 / \
     p   D               A   p
    / \                     / \
   x   C                   B   g
  / \                         / \
 A   B                       C   D

 Step 1: Rotate p around g
 Step 2: Rotate x around p
```

*Zig-Zag case: x is right child of p, p is left child of g*

```
  Before:                 After:

       g                   x
      / \                /   \
     p   D              p     g
    / \                / \   / \
   A   x              A   B C   D
      / \
     B   C

 Step 1: Rotate x around p
 Step 2: Rotate x around g
```

All standard BST operations — search, insert, delete — are built on top of the splay operation. To **search** for a key, perform a normal BST search and then splay the found node (or the last node visited if not found) to the root. To **insert**, insert the new node as in a normal BST and then splay it to the root. To **delete**, splay the target node to the root, remove it, then splay the largest element in the left subtree to become the new root and attach the right subtree to it.

The reason the zig-zig case is so important can be understood intuitively: when a node is accessed deep in the tree, the zig-zig rotations roughly halve the depth of every node on the path from the root to the accessed node. This means that even if one access is expensive (reaching deep into the tree), it restructures the tree so that future accesses are cheaper. This 'pay it forward' behavior is what gives splay trees their O(log n) amortized bound, proven rigorously using a potential function argument.

*Example: splaying node 1 in a degenerate right-leaning tree*

```
  Initial:       After splay(1):

  5                   1
   \                   \
    4                   3
     \                 / \
      3               2   5
       \                 /
        2               4
         \
          1

  The deep path is compressed!
  Tree height reduced from 5 to 3.
```

## Example

Let's implement a splay tree that supports insert, search, and in-order traversal. We'll insert the values [5, 4, 3, 2, 1] (which would create a degenerate tree in a naive BST) and then search for various elements to demonstrate how the tree self-adjusts. After each operation, the accessed node becomes the new root, keeping frequently accessed elements near the top.

```python
class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

class SplayTree:
    def __init__(self):
        self.root = None

    def _right_rotate(self, x):
        """Rotate x's left child up (right rotation around x)."""
        y = x.left
        x.left = y.right
        y.right = x
        return y

    def _left_rotate(self, x):
        """Rotate x's right child up (left rotation around x)."""
        y = x.right
        x.right = y.left
        y.left = x
        return y

    def _splay(self, root, key):
        """Splay the tree rooted at 'root' so that the node with 'key'
        (or the last accessed node) becomes the new root.
        Uses top-down splaying for simplicity."""
        if root is None or root.key == key:
            return root

        # Key lies in left subtree
        if key < root.key:
            if root.left is None:
                return root

            # Zig-Zig (left-left): key < root.left.key
            if key < root.left.key:
                # Recursively bring key to root.left.left
                root.left.left = self._splay(root.left.left, key)
                # First rotation: rotate root right
                root = self._right_rotate(root)

            # Zig-Zag (left-right): key > root.left.key
            elif key > root.left.key:
                # Recursively bring key to root.left.right
                root.left.right = self._splay(root.left.right, key)
                # First rotation: left rotate root.left
                if root.left.right is not None:
                    root.left = self._left_rotate(root.left)

            # Second rotation (Zig step): rotate root right
            if root.left is None:
                return root
            else:
                return self._right_rotate(root)

        else:  # Key lies in right subtree
            if root.right is None:
                return root

            # Zig-Zig (right-right): key > root.right.key
            if key > root.right.key:
                root.right.right = self._splay(root.right.right, key)
                root = self._left_rotate(root)

            # Zig-Zag (right-left): key < root.right.key
            elif key < root.right.key:
                root.right.left = self._splay(root.right.left, key)
                if root.right.left is not None:
                    root.right = self._right_rotate(root.right)

            if root.right is None:
                return root
            else:
                return self._left_rotate(root)

    def insert(self, key):
        """Insert a key and splay it to the root."""
        if self.root is None:
            self.root = Node(key)
            return

        # Splay the closest node to root
        self.root = self._splay(self.root, key)

        if self.root.key == key:
            return  # Duplicate key, do nothing

        new_node = Node(key)
        if key < self.root.key:
            new_node.right = self.root
            new_node.left = self.root.left
            self.root.left = None
        else:
            new_node.left = self.root
            new_node.right = self.root.right
            self.root.right = None

        self.root = new_node

    def search(self, key):
        """Search for a key; splay it (or last visited) to root."""
        self.root = self._splay(self.root, key)
        if self.root and self.root.key == key:
            return True
        return False

    def inorder(self, node=None, result=None):
        """In-order traversal to verify BST property."""
        if result is None:
            result = []
            node = self.root
        if node:
            self.inorder(node.left, result)
            result.append(node.key)
            self.inorder(node.right, result)
        return result

    def print_tree(self, node=None, level=0, prefix="Root: "):
        """Pretty print the tree structure."""
        if node is None and level == 0:
            node = self.root
        if node is not None:
            print(" " * (level * 4) + prefix + str(node.key))
            if node.left or node.right:
                if node.left:
                    self.print_tree(node.left, level + 1, "L--- ")
                else:
                    print(" " * ((level + 1) * 4) + "L--- (nil)")
                if node.right:
                    self.print_tree(node.right, level + 1, "R--- ")
                else:
                    print(" " * ((level + 1) * 4) + "R--- (nil)")


# --- Demo ---
tree = SplayTree()

# Insert values that would create a degenerate tree in naive BST
for val in [5, 4, 3, 2, 1]:
    tree.insert(val)
    print(f"After inserting {val}, root = {tree.root.key}")

print("\nTree structure after all inserts:")
tree.print_tree()
print(f"In-order: {tree.inorder()}")

# Search for 5 — it will be splayed to root
print(f"\nSearch for 5: {tree.search(5)}")
print(f"Root is now: {tree.root.key}")
tree.print_tree()

# Search for 3 — it will be splayed to root
print(f"\nSearch for 3: {tree.search(3)}")
print(f"Root is now: {tree.root.key}")
tree.print_tree()

# Demonstrate working-set property: repeated access is fast
print("\n--- Accessing 3 repeatedly keeps it at root ---")
for _ in range(3):
    tree.search(3)
    print(f"Root after searching 3: {tree.root.key}")
```

## Performance Analysis

Splay trees provide O(log n) amortized time for all operations (search, insert, delete) over any sequence of m operations on n keys, giving a total cost of O(m log n). Individual operations can be O(n) in the worst case (e.g., accessing the deepest node in a degenerate tree), but the splaying operation restructures the tree to prevent repeated worst-case behavior. Space usage is O(n) with no extra fields needed beyond the standard BST pointers — no balance factors, colors, or priorities. The static optimality theorem guarantees that splay trees perform within a constant factor of the optimal static BST for any given access sequence, and the unproven dynamic optimality conjecture posits they match any online BST algorithm.

| Metric | Complexity |
|--------|------------|
| Best case | `O(1) — accessing the root` |
| Average case | `O(log n) amortized` |
| Worst case | `O(n) per operation, O(m log n) amortized for m operations` |
| Space | `O(n)` |

## Use Cases

- Cache-like data structures: Splay trees naturally keep recently and frequently accessed items near the root, making them ideal when access patterns exhibit temporal locality (e.g., in memory allocators or LRU-like caches).
- Network routing and IP lookup: Cisco's implementation of certain routing table structures has used splay trees to optimize for frequently accessed routes, exploiting the working-set property.
- Garbage collectors and runtime systems: Splay trees have been used in garbage collector implementations (e.g., in some versions of the Boehm GC) for managing memory blocks, where recently allocated or freed blocks are accessed frequently.
- Text editors and sequence operations: Because splay trees support efficient split and join operations in O(log n) amortized time, they're useful for implementing sequences that require frequent concatenation, splitting, and random access — similar to ropes but with self-adjusting behavior.
- Compiler optimizations: Some compilers use splay trees in symbol table implementations where certain variables are referenced far more frequently than others, benefiting from the automatic adaptation to access frequency.

## References & Further Reading

- [Self-Adjusting Binary Search Trees](https://www.cs.cmu.edu/~sleator/papers/self-adjusting.pdf) — Daniel D. Sleator, Robert E. Tarjan  
  The original 1985 JACM paper introducing splay trees, proving their amortized O(log n) bounds and several optimality properties.
- [Splay tree — Wikipedia](https://en.wikipedia.org/wiki/Splay_tree)  
  Comprehensive overview of splay tree operations, analysis, and variants with clear diagrams and references.
- **Data Structures and Algorithm Analysis in C++ (4th Edition)** — Mark Allen Weiss  
  A widely-used textbook that includes an excellent chapter on splay trees with clear explanations of top-down splaying.
- [Open Data Structures — Chapter 8: Scapegoat Trees and Splay Trees](https://opendatastructures.org/ods-java/8_Scapegoat_Trees.html) — Pat Morin  
  Free online textbook with a rigorous yet accessible treatment of splay trees including potential function analysis.
- [Dynamic Optimality — Almost (SIAM Journal on Computing)](https://erikdemaine.org/papers/Tango_SICOMP/paper.pdf) — Erik D. Demaine, Dion Harmon, John Iacono, Mihai Pătrașcu  
  Introduces Tango trees and discusses the dynamic optimality conjecture for splay trees, achieving O(log log n) competitive ratio.
- **An Introduction to Algorithms (CLRS), 3rd Edition — Problem 13-2** — Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein  
  The classic algorithms textbook includes splay trees as an advanced exercise with guidance on the amortized analysis using potential functions.
