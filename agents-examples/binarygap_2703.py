# This Python program implements the following use case:
# Write code to find BinaryGap of a given positive integer

def find_binary_gap(n: int) -> int:
    if n <= 0:
        raise ValueError("Input must be a positive integer.")

    # Convert the integer to its binary string representation
    binary_string = bin(n)[2:]

    max_gap = 0
    current_gap = 0

    for digit in binary_string:
        if digit == '0':
            current_gap += 1
        else:
            # When a '1' is encountered, check if the current gap is the maximum
            if current_gap > max_gap:
                max_gap = current_gap
            # Reset the current gap counter
            current_gap = 0

    return max_gap

if __name__ == "__main__":
    test_cases = [10, 17, 12, 1, 8, 31, 100]

    print("--- Binary Gap Finder ---")
    for n in test_cases:
        try:
            gap = find_binary_gap(n)
            print(f"Input: {n} (Binary: {bin(n)[2:]}) -> Binary Gap: {gap}")
        except ValueError as e:
            print(f"Error for input {n}: {e}")
        print("-" * 30)