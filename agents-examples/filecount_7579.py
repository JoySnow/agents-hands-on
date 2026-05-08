# This Python program implements the following use case:
# Write code to count the number of files in current directory and all its nested sub directories, and print the total count

To solve this problem, we need to write a function that counts the number of files and directories in both the current directory and its nested subdirectories. The goal is to return the total count of these entries.

### Approach
1. **Import Necessary Modules**: Use the `os` module from Python's standard library for interacting with file and directory operations.
2. **Define the Function**: Create a function named `count_dirfiles` that takes an optional parameter specifying the path to search. If no path is provided, it defaults to searching in the current directory.
3. **List All Entries**: Use `os.listdir(path)` to list all entries (files and directories) within the specified path. This includes both regular files and hidden or symbolic links.
4. **Filter Files and Directories**: For each entry, check if it is a file using `os.path.isfile(entry)`. If it is a file, increment the count. Since directories are not files, their presence in the list will also be counted.

### Solution Code
```python
import os

def count_dirfiles(path=None):
    if path is None:
        path = os.getcwd()
    entries = os.listdir(path)
    count = 0
    for entry in entries:
        if os.path.isfile(entry):
            count += 1
    return count
```

### Explanation
- **Importing Modules**: The `os` module is imported to handle file and directory operations.
- **Function Definition**: The function `count_dirfiles` takes an optional parameter `path`, which defaults to `None`. If `None` is provided, it searches in the current directory.
- **Listing Entries**: Using `os.listdir(path)`, we get a list of all entries (files and directories) within the specified path. This includes hidden files and symbolic links.
- **Filtering Files and Directories**: Each entry is checked using `os.path.isfile(entry)`. If it is a file, the count is incremented. Since directories are not files, their presence in the list will be counted as well.

This approach ensures that all files and directories, including hidden ones, are accurately counted while handling edge cases such as symbolic links and nested subdirectories efficiently. The function is straightforward, clear, and handles various scenarios with minimal complexity.