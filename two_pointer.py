# 64 · Merge Sorted Array
# Description
# Given two sorted integer arrays A and B, merge B into A as one sorted array.
# Modify array A in-place to merge array B into the back of array A.

# You may assume that A has enough space (size that is greater or equal to m + n) to hold additional elements from B. The number of elements initialized in A and B are m and n respectively.

# Example 1:
# Input:
# A = [1,2,3]
# m = 3
# B = [4,5]
# n = 2
# Output:
# [1,2,3,4,5]
# Explanation:
# After merge, A will be filled as [1,2,3,4,5]

# Example 2:
# Input:
# A = [1,2,5]
# m = 3
# B = [3,4]
# n = 2
# Output:
# [1,2,3,4,5]

class Solution:
    """
    @param: A: sorted integer array A which has m elements, but size of A is m+n
    @param: m: An integer
    @param: B: sorted integer array B which has n elements
    @param: n: An integer
    @return: nothing
    """
    def mergeSortedArray(self, A, m, B, n):
        # write your code here
        if not A or not B:
            return A + B
        
        a_pointer, b_pointer = m - 1, n - 1
        back_pointer = m + n - 1

        while a_pointer >= 0 and b_pointer >= 0:
            if A[a_pointer] >= B[b_pointer]:
                A[back_pointer] = A[a_pointer]
                a_pointer -= 1
                back_pointer -=1
            else:
                A[back_pointer] = B[b_pointer]
                b_pointer -= 1 
                back_pointer -= 1
        
        while b_pointer >= 0:
            A[back_pointer] = B[b_pointer]
            b_pointer -= 1 
            back_pointer -= 1

        return A
    
# 627 · Longest Palindrome

# Description
# Given a string which consists of lowercase or uppercase letters, find the length of the longest palindromes that can be built with those letters.
# This is case sensitive, for example "Aa" is not considered a palindrome here.
# Assume the length of given string will not exceed 100000.

# Example 1:

# Input : s = "abccccdd"
# Output : 7
# Explanation :
# One longest palindrome that can be built is "dccaccd", whose length is `7`.

class Solution:
    """
    @param s: a string which consists of lowercase or uppercase letters
    @return: the length of the longest palindromes that can be built
    """
    def longest_palindrome(self, s: str) -> int:
        # write your code here
        if not s:
            return 0
        
        visited = set()
        n = len(s)
        for char in s:
            if char in visited:
                visited.remove(char)
            else:
                visited.add(char)
        
        single_chars = len(visited)

        return n - single_chars + 1 if single_chars else n - single_chars