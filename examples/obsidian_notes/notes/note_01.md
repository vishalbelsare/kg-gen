# Computer Science Fundamentals

## Data Structures
- Arrays store elements sequentially in memory
- [[LinkedLists]] connect nodes with pointers
- [[Trees]] organize data in hierarchical structures
- [[Graphs]] represent networks with nodes and edges

> Understanding data structures is essential for algorithm design

### Time Complexity
| Structure | Access | Search | Insertion | Deletion |
|-----------|--------|--------|-----------|----------|
| Array     | O(1)   | O(n)   | O(n)      | O(n)     |
| Linked List| O(n)   | O(n)   | O(1)      | O(1)     |

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

#computerscience #algorithms #datastructures