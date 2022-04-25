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

# 16. 3Sum Closest
# Given an integer array nums of length n and an integer target, find three integers in nums such that the sum is closest to target.

# Return the sum of the three integers.

# You may assume that each input would have exactly one solution.

 

# Example 1:

# Input: nums = [-1,2,1,-4], target = 1
# Output: 2
# Explanation: The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
# Example 2:

# Input: nums = [0,0,0], target = 1
# Output: 0

class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        if not nums:
            return False
        n = len(nums)
        nums.sort()
        dif = float('inf')
        res = 0
        for i in range(n - 2):
            left, right = i + 1, n - 1
            while left < right:
                cur_sum = nums[i] + nums[left] + nums[right]
                print(cur_sum)
                if cur_sum == target:
                    return cur_sum
                
                if cur_sum > target:
                    right -= 1
                elif cur_sum < target:
                    left += 1
                
                if abs(cur_sum - target) < dif:
                    dif = abs(cur_sum - target)
                    res = cur_sum
                
        return res 


# 15. 3Sum
# Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

# Notice that the solution set must not contain duplicate triplets.

 

# Example 1:

# Input: nums = [-1,0,1,2,-1,-4]
# Output: [[-1,-1,2],[-1,0,1]]
# Example 2:

# Input: nums = []
# Output: []
# Example 3:

# Input: nums = [0]
# Output: []

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        l = len(nums)
        res = []
        if l < 3:
            return res 
        
        for i in range(l):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            
            start, end = i + 1, len(nums) - 1
            while start < end:
                cur_sum = nums[start] + nums[end] + nums[i]
                if cur_sum < 0:
                    start += 1
                elif cur_sum > 0:
                    end -= 1
                else:
                    res.append((nums[i], nums[start], nums[end]))
                    while start < end and nums[start] == nums[start + 1]:
                        start += 1
                    while start < end and nums[end] == nums[end -1]:
                        end -= 1
                    start += 1
                    end -= 1
            
        return res

# 18. 4Sum
# Given an array nums of n integers, return an array of all the unique quadruplets [nums[a], nums[b], nums[c], nums[d]] such that:

# 0 <= a, b, c, d < n
# a, b, c, and d are distinct.
# nums[a] + nums[b] + nums[c] + nums[d] == target
# You may return the answer in any order.

 

# Example 1:

# Input: nums = [1,0,-1,0,-2,2], target = 0
# Output: [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
# Example 2:

# Input: nums = [2,2,2,2,2], target = 8
# Output: [[2,2,2,2]]

class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        def findNsum(nums, target, N, cur):
            if len(nums) < N or N < 2 or nums[0] * N > target or nums[-1] * N < target:  # if minimum possible sum (every element is first element) > target 
                return  # or maximum possible sum (every element is first element) < target, it's impossible to get target anyway          
            if N == 2:  # 2-sum problem
                l, r = 0, len(nums) - 1
                while l < r:
                    s = nums[l] + nums[r]
                    if s == target:
                        res.append(cur + [nums[l], nums[r]])
                        while l < r and nums[l] == nums[l - 1]:
                            l += 1
                        while l < r and nums[r] == nums[r - 1]:
                            r -= 1
                        l += 1
                        r -= 1
                    elif s < target:
                        l += 1
                    else:
                        r -= 1
            else:  # reduce to N-1 sum problem
                for i in range(len(nums) - N + 1):
                    if i == 0 or nums[i - 1] != nums[i]:
                        findNsum(nums[i + 1 :], target - nums[i], N - 1, cur + [nums[i]])

        res = []
        findNsum(sorted(nums), target, 4, [])
        return res
            
            
# 49. Group Anagrams
# Given an array of strings strs, group the anagrams together. You can return the answer in any order.

# An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

 

# Example 1:

# Input: strs = ["eat","tea","tan","ate","nat","bat"]
# Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
# Example 2:

# Input: strs = [""]
# Output: [[""]]
# Example 3:

# Input: strs = ["a"]
# Output: [["a"]]
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
       
        visited = {}
        for word in strs:
            new_word = ''.join(sorted(word))
            if new_word in visited:
                visited[new_word].append(word)
            else:
                visited[new_word] = [word]
        
        return visited.values()
            
# 56. Merge Intervals
# Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

 

# Example 1:

# Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
# Output: [[1,6],[8,10],[15,18]]
# Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].
# Example 2:

# Input: intervals = [[1,4],[4,5]]
# Output: [[1,5]]
# Explanation: Intervals [1,4] and [4,5] are considered overlapping.

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        
        intervals.sort(key = lambda x: x[0])
        res = [intervals[0]]
        for interval in intervals:
            start, end = interval[0], interval[1]
            if res[-1][0] <= start and res[-1][1] > end:
                continue 
            elif res[-1][1] < start:
                res.append(interval)
            else:
                res[-1][1] = end
        
        return res
