# 1525. Number of Good Ways to Split a String
# You are given a string s.

# A split is called good if you can split s into two non-empty strings sleft and sright where their concatenation is equal to s (i.e., sleft + sright = s) and the number of distinct letters in sleft and sright is the same.

# Return the number of good splits you can make in s.
# Example 1:

# Input: s = "aacaba"
# Output: 2
# Explanation: There are 5 ways to split "aacaba" and 2 of them are good. 
# ("a", "acaba") Left string and right string contains 1 and 3 different letters respectively.
# ("aa", "caba") Left string and right string contains 1 and 3 different letters respectively.
# ("aac", "aba") Left string and right string contains 2 and 2 different letters respectively (good split).
# ("aaca", "ba") Left string and right string contains 2 and 2 different letters respectively (good split).
# ("aacab", "a") Left string and right string contains 3 and 1 different letters respectively.
# Example 2:

# Input: s = "abcd"
# Output: 1
# Explanation: Split the string as follows ("ab", "cd").
class Solution:
    def numSplits(self, s: str) -> int:
        if len(s) <= 1:
            return 0
        if len(s) == 2:
            return 1
        
        left, right, res = {}, {}, 0
        
        for char in s:
            right[char] = right.get(char, 0) + 1
        
        for char in s:
            left[char] = left.get(char, 0) + 1
            if right[char] == 1:
                del right[char]
            
            else:
                right[char] -= 1
            
            if len(left) == len(right):
                res += 1
        
        return res
        
# 1509. Minimum Difference Between Largest and Smallest Value in Three Moves
# You are given an integer array nums. In one move, you can choose one element of nums and change it by any value.

# Return the minimum difference between the largest and smallest value of nums after performing at most three moves.
# Example 1:

# Input: nums = [5,3,2,4]
# Output: 0
# Explanation: Change the array [5,3,2,4] to [2,2,2,2].
# The difference between the maximum and minimum is 2-2 = 0.
# Example 2:

# Input: nums = [1,5,0,10,14]
# Output: 1
# Explanation: Change the array [1,5,0,10,14] to [1,1,0,1,1]. 
# The difference between the maximum and minimum is 1-0 = 1.

class Solution:
    def minDifference(self, nums: List[int]) -> int:
        if len(nums) < 4:
            return 0
        nums.sort()
        
        return min(nums[-1] - nums[3], nums[-2] - nums[2], nums[-3] - nums[1], nums[-4] - nums[0])


# 949. Largest Time for Given Digits
# Given an array arr of 4 digits, find the latest 24-hour time that can be made using each digit exactly once.

# 24-hour times are formatted as "HH:MM", where HH is between 00 and 23, and MM is between 00 and 59. The earliest 24-hour time is 00:00, and the latest is 23:59.

# Return the latest 24-hour time in "HH:MM" format. If no valid time can be made, return an empty string.
# Example 1:

# Input: arr = [1,2,3,4]
# Output: "23:41"
# Explanation: The valid 24-hour times are "12:34", "12:43", "13:24", "13:42", "14:23", "14:32", "21:34", "21:43", "23:14", and "23:41". Of these times, "23:41" is the latest.
# Example 2:

# Input: arr = [5,5,5,5]
# Output: ""
# Explanation: There are no valid 24-hour times as "55:55" is not valid.

from itertools import permutations

class Solution:
    def largestTimeFromDigits(self, arr: List[int]) -> str:
        newArr = list(permutations(sorted(arr, reverse=True)))
        
        for h1, h2, m1, m2 in newArr:
            if h1 * 10 + h2 < 24 and m1 * 10 + m2 < 60:
                return f'{h1}{h2}:{m1}{m2}'
        return ''

# 77 · Longest Common Subsequence
# Description
# Given two strings, find the longest common subsequence (LCS).

# Your code should return the length of LCS.

# Contact me on wechat to get Amazon、Google requent Interview questions . (wechat id : jiuzhang0607)


# What's the definition of Longest Common Subsequence?

# The longest common subsequence problem is to find the longest common subsequence in a set of sequences (usually 2). This problem is a typical computer science problem, which is the basis of file difference comparison program, and also has applications in bioinformatics.
# http://baike.baidu.com/view/2020307.htm
# Example
# Example 1:

# Input:

# A = "ABCD"
# B = "EDCA"
# Output:

# 1
# Explanation:

# LCS is 'A' or 'D' or 'C'
# Example 2:

# Input:

# A = "ABCD"
# B = "EACB"
# Output:

# 2
class Solution:
    """
    @param a: A string
    @param b: A string
    @return: The length of longest common subsequence of A and B
    """
    def longest_common_subsequence(self, a: str, b: str) -> int:
        # write your code here
        n, m = len(a), len(b)
        
        dp = [[0] * (m + 1) for _ in range (n + 1)]

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # print(i, j, dp)
                if a[i - 1] == b[j-1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i -1][j], dp[i][j -1])
                # print(dp)
        return dp[-1][-1]
