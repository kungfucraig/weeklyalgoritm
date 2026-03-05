# Aho-Corasick Algorithm

*Algorithm of the Day · 2026-03-05*

## Introduction

Imagine you're building a content moderation system that needs to scan every user message for thousands of banned words simultaneously, or you're developing an intrusion detection system that must match incoming network packets against tens of thousands of known malware signatures in real time. Searching for each pattern one at a time would be painfully slow. The Aho-Corasick algorithm solves this problem elegantly: given a set of pattern strings and an input text, it finds all occurrences of all patterns in a single pass through the text, regardless of how many patterns you have.

Published in 1975 by Alfred Aho and Margaret Corasick, this algorithm is one of the most beautiful constructions in string matching. It builds a finite-state automaton from the set of patterns by combining a trie (prefix tree) with two auxiliary functions — a failure function (inspired by the Knuth-Morris-Pratt algorithm) and an output function. The result is a machine that processes each character of the input text exactly once, transitioning between states and emitting matches as they occur. Its time complexity is O(n + m + z) where n is the text length, m is the total length of all patterns, and z is the number of matches — meaning it's linear in input size plus output size.

The algorithm remains a workhorse in production systems today. It powers the pattern matching in tools like `grep` (specifically `fgrep`), antivirus scanners, network intrusion detection systems like Snort, and bioinformatics tools that search for DNA motifs. If you've ever needed to search for multiple strings at once, Aho-Corasick is very likely the right tool for the job.

## How It Works

The Aho-Corasick algorithm operates in two phases: a preprocessing phase that builds an automaton from the set of patterns, and a search phase that feeds the input text through this automaton. The automaton is constructed in three steps: (1) build a trie from all patterns, (2) compute failure links for every node using BFS, and (3) compute dictionary suffix links (also called output links) that chain together nodes whose suffixes are also complete patterns. Once built, the automaton processes the text character by character, following goto transitions when possible and failure links otherwise.

Step 1 — Build the Trie (Goto Function): Insert every pattern string into a trie. Each edge is labeled with a character, and each node represents a prefix of one or more patterns. Nodes where a pattern ends are marked as output nodes. The root represents the empty string. For example, inserting the patterns 'he', 'she', 'his', and 'hers' creates a trie with branches for 'h' and 's' from the root.

*Trie built from patterns: {he, she, his, hers}*

```
                  (root)
                 /      \
                h        s
               / \        \
              e*  i        h
              |    \        \
              r     s*       e*
              |              |
              s*             r
                             |
                             s*

  * = output node (complete pattern ends here)

  Paths from root:
    h -> e          = 'he'
    h -> e -> r -> s = 'hers'
    h -> i -> s      = 'his'
    s -> h -> e      = 'she'
```

Step 2 — Compute Failure Links: The failure link of a node points to the longest proper suffix of that node's string that is also a prefix of some pattern in the trie. This is computed via BFS starting from the root. All depth-1 nodes have their failure link pointing to the root. For deeper nodes, we follow the parent's failure link and try to extend it with the current character; if that fails, we follow that node's failure link, and so on until we either find a match or reach the root. This is directly analogous to the failure function in KMP, but generalized to multiple patterns.

*Failure links (dashed arrows) for key nodes*

```
  Node labeling: each node shows the string it represents.

    ''(root) <---fail--- 'h'
    ''(root) <---fail--- 's'
    ''(root) <---fail--- 'he'  (no suffix of 'he' is a trie prefix
                                 other than '' since 'e' is not a root child)
    'h'      <---fail--- 'sh'  (suffix 'h' is a prefix in the trie)
    'he'     <---fail--- 'she' (suffix 'he' matches trie node 'he')

  When scanning and we can't follow a goto edge, we follow
  the failure link and retry — just like KMP but on a trie.
```

Step 3 — Dictionary Suffix Links (Output Links): Some nodes may not themselves be output nodes, but their failure chain may pass through an output node. The dictionary suffix link of a node points to the nearest ancestor (via failure links) that is an output node. During search, when we arrive at any node, we report not only that node's pattern (if any) but also walk the dictionary suffix links to report all patterns that are suffixes of the current match. For example, when we reach the node for 'she', the failure link goes to 'he', which is itself an output node — so both 'she' and 'he' are reported.

Search Phase: We start at the root and process the text one character at a time. At each step, we attempt to follow the goto edge for the current character. If no such edge exists, we follow the failure link and retry (the root always has a self-loop for unmatched characters). When we arrive at a node, we collect all outputs by following the dictionary suffix links. Because each character causes at most a constant amortized number of state transitions (the failure links can only decrease depth, which is bounded), the entire search runs in O(n + z) time where n is the text length and z is the number of pattern occurrences found.

## Example

Let's build an Aho-Corasick automaton for the patterns ['he', 'she', 'his', 'hers'] and search the text 'ahishers'. We expect to find: 'his' at position 1, 'she' at position 3, 'he' at position 4, 'hers' at position 4. The implementation below uses a dictionary-based trie with BFS-computed failure links and dictionary suffix links.

```python
from collections import deque, defaultdict

class AhoCorasick:
    def __init__(self):
        # Each node is an integer; node 0 is root.
        self.goto = [{}]          # goto[state][char] -> next state
        self.fail = [0]           # failure link for each state
        self.output = [[]]        # list of pattern indices that end here
        self.dict_suffix = [0]    # dictionary suffix link

    def _add_pattern(self, pattern, index):
        """Insert a pattern into the trie."""
        state = 0
        for ch in pattern:
            if ch not in self.goto[state]:
                new_state = len(self.goto)
                self.goto.append({})
                self.fail.append(0)
                self.output.append([])
                self.dict_suffix.append(0)
                self.goto[state][ch] = new_state
            state = self.goto[state][ch]
        self.output[state].append(index)

    def build(self, patterns):
        """Build the automaton from a list of patterns."""
        for i, p in enumerate(patterns):
            self._add_pattern(p, i)

        # BFS to compute failure links and dictionary suffix links
        queue = deque()

        # Initialize: depth-1 nodes have fail -> root
        for ch, s in self.goto[0].items():
            self.fail[s] = 0
            self.dict_suffix[s] = 0
            queue.append(s)

        while queue:
            curr = queue.popleft()
            for ch, next_state in self.goto[curr].items():
                queue.append(next_state)

                # Walk up failure links to find the fail state
                fallback = self.fail[curr]
                while fallback != 0 and ch not in self.goto[fallback]:
                    fallback = self.fail[fallback]

                self.fail[next_state] = self.goto[fallback].get(ch, 0)
                # Avoid self-loop
                if self.fail[next_state] == next_state:
                    self.fail[next_state] = 0

                # Dictionary suffix link: nearest output node via fail chain
                f = self.fail[next_state]
                if self.output[f]:
                    self.dict_suffix[next_state] = f
                else:
                    self.dict_suffix[next_state] = self.dict_suffix[f]

    def search(self, text, patterns):
        """Search text and return list of (position, pattern) matches."""
        results = []
        state = 0

        for i, ch in enumerate(text):
            # Follow failure links until we can take a goto edge
            while state != 0 and ch not in self.goto[state]:
                state = self.fail[state]
            state = self.goto[state].get(ch, 0)

            # Collect all outputs at this state (and via dict suffix links)
            temp = state
            while temp != 0:
                for pat_idx in self.output[temp]:
                    pat = patterns[pat_idx]
                    start_pos = i - len(pat) + 1
                    results.append((start_pos, pat))
                temp = self.dict_suffix[temp]

        return results


# --- Worked example ---
patterns = ['he', 'she', 'his', 'hers']
text = 'ahishers'

ac = AhoCorasick()
ac.build(patterns)
matches = ac.search(text, patterns)

print(f"Text: '{text}'")
print(f"Patterns: {patterns}")
print(f"Matches found:")
for pos, pat in sorted(matches):
    print(f"  '{pat}' at position {pos}")

# Output:
# Text: 'ahishers'
# Patterns: ['he', 'she', 'his', 'hers']
# Matches found:
#   'his' at position 1
#   'she' at position 3
#   'he' at position 4
#   'hers' at position 4
```

## Performance Analysis

The Aho-Corasick algorithm has excellent complexity characteristics. Building the automaton takes O(m * k) time where m is the total length of all patterns and k is the alphabet size (for the trie construction and BFS failure link computation). The search phase takes O(n + z) time where n is the length of the text and z is the total number of pattern occurrences reported. This means the algorithm's running time is independent of the number of patterns — whether you search for 10 or 10,000 patterns, the search time depends only on the text length and the number of matches. Space usage is O(m * k) for the automaton in the worst case, though using hash maps for transitions (as in our implementation) reduces this to O(m) on average for sparse alphabets.

| Metric | Complexity |
|--------|------------|
| Best case | `O(n + m) — building O(m), searching O(n) with no matches` |
| Average case | `O(n + m + z) — z is the number of matches reported` |
| Worst case | `O(n + m + z) — linear in input size plus output size` |
| Space | `O(m * k) worst case for automaton (m = total pattern length, k = alphabet size); O(m) with hash-map transitions` |

## Use Cases

- Network Intrusion Detection Systems (e.g., Snort): Scanning network packets against thousands of known attack signatures simultaneously in real time, where per-packet latency is critical.
- Antivirus and Malware Scanning: Matching file contents against a large database of known malware byte signatures — the same approach used by ClamAV and similar engines.
- Bioinformatics DNA/Protein Motif Search: Searching genomic sequences for thousands of known motifs, binding sites, or primers in a single linear scan, which is far faster than running individual searches.
- Content Moderation and Sensitive Word Filtering: Scanning user-generated content on social platforms against dictionaries of prohibited words, profanity, or PII patterns at high throughput.
- Text Editors and IDE Search (e.g., fgrep): The `fgrep` utility (fixed-string grep) uses Aho-Corasick to search for multiple literal strings in files efficiently.
- Spam and Phishing Detection: Email security gateways use multi-pattern matching to scan message bodies for known phishing URLs, spam keywords, and social engineering phrases.

## References & Further Reading

- [Efficient String Matching: An Aid to Bibliographic Search](https://dl.acm.org/doi/10.1145/360825.360855) — Alfred V. Aho, Margaret J. Corasick  
  The original 1975 paper introducing the Aho-Corasick algorithm for multi-pattern string matching.
- **Introduction to Algorithms (CLRS), Chapter 32: String Matching** — Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein  
  The classic algorithms textbook covering string matching foundations including KMP, which directly motivates the failure function used in Aho-Corasick.
- [Aho-Corasick Algorithm — Wikipedia](https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm)  
  A comprehensive overview of the algorithm, its construction steps, and applications with pseudocode.
- **Flexible Pattern Matching in Strings** — Gonzalo Navarro, Mathieu Raffinot  
  An excellent book dedicated to string matching algorithms, providing in-depth coverage of Aho-Corasick alongside other multi-pattern and approximate matching techniques.
- [Aho-Corasick Automaton — CP-Algorithms](https://cp-algorithms.com/string/aho_corasick.html)  
  A practical, implementation-focused tutorial on Aho-Corasick aimed at competitive programmers, with clear code examples and optimization tips.
- **Scalable Pattern Matching for High Speed Networks (Snort)** — Marc Norton  
  A technical discussion of how Aho-Corasick is used and optimized in the Snort intrusion detection system for real-time packet inspection.
