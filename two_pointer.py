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

# 75. Sort Colors
# Given an array nums with n objects colored red, white, or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white, and blue.

# We will use the integers 0, 1, and 2 to represent the color red, white, and blue, respectively.

# You must solve this problem without using the library's sort function.

 

# Example 1:

# Input: nums = [2,0,2,1,1,0]
# Output: [0,0,1,1,2,2]
# Example 2:

# Input: nums = [2,0,1]
# Output: [0,1,2]


class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        red, white, blue = 0, 0, len(nums) -1
        
        while white <= blue:
            
            if nums[white] == 0:
                nums[white], nums[red] = nums[red], nums[white]
                red += 1
                white += 1
            elif nums[white] == 1:
                white += 1
            else:
                nums[blue], nums[white] = nums[white], nums[blue]
                blue -= 1
           
# 88. Merge Sorted Array
# You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, and two integers m and n, representing the number of elements in nums1 and nums2 respectively.

# Merge nums1 and nums2 into a single array sorted in non-decreasing order.

# The final sorted array should not be returned by the function, but instead be stored inside the array nums1. To accommodate this, nums1 has a length of m + n, where the first m elements denote the elements that should be merged, and the last n elements are set to 0 and should be ignored. nums2 has a length of n.

 

# Example 1:

# Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
# Output: [1,2,2,3,5,6]
# Explanation: The arrays we are merging are [1,2,3] and [2,5,6].
# The result of the merge is [1,2,2,3,5,6] with the underlined elements coming from nums1.
# Example 2:

# Input: nums1 = [1], m = 1, nums2 = [], n = 0
# Output: [1]
# Explanation: The arrays we are merging are [1] and [].
# The result of the merge is [1].

class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
     
        while n:
        
            if m and nums1[m - 1] >= nums2[n - 1]:
                nums1[m + n -1] = nums1[m -1]
                m -= 1
            else:
                nums1[m + n -1] = nums2[n -1]
                n -= 1
            
# 143 · Sort Colors II QUICK-SORT O(NLOG K)
# Description
# Given an array of n objects with k different colors (numbered from 1 to k), sort them so that objects of the same color are adjacent, with the colors in the order 1, 2, ... k.

# You are not suppose to use the library's sort function for this problem.
# k <= n
# Example
# Example1

# Input: 
# [3,2,2,1,4] 
# 4
# Output: 
# [1,2,2,3,4]
# Example2

# Input: 
# [2,1,1,2,2] 
# 2
# Output: 
# [1,1,2,2,2]

from typing import (
    List,
)

class Solution:
    """
    @param colors: A list of integer
    @param k: An integer
    @return: nothing
    """
    def sort_colors2(self, colors: List[int], k: int):
        # write your code here
        self.sort(colors, 1, k, 0, len(colors) - 1)
        
    def sort(self, colors, color_from, color_to, index_from, index_to):
        if color_from == color_to or index_from == index_to:
            return
            
        color = (color_from + color_to) // 2
        
        left, right = index_from, index_to
        while left <= right:
            while left <= right and colors[left] <= color:
                left += 1
            while left <= right and colors[right] > color:
                right -= 1
            if left <= right:
                colors[left], colors[right] = colors[right], colors[left]
                left += 1
                right -= 1
        
        self.sort(colors, color_from, color, index_from, right)
        self.sort(colors, color + 1, color_to, left, index_to)

# 147. Insertion Sort List INSERTION-SORT O(N) -> O(N^2) Time Complexity
# Given the head of a singly linked list, sort the list using insertion sort, and return the sorted list's head.

# The steps of the insertion sort algorithm:

# Insertion sort iterates, consuming one input element each repetition and growing a sorted output list.
# At each iteration, insertion sort removes one element from the input data, finds the location it belongs within the sorted list and inserts it there.
# It repeats until no input elements remain.
# The following is a graphical example of the insertion sort algorithm. The partially sorted list (black) initially contains only the first element in the list. One element (red) is removed from the input data and inserted in-place into the sorted list with each iteration.

# Example 1:
# Input: head = [4,2,1,3]
# Output: [1,2,3,4]
# Example 2:
# Input: head = [-1,5,3,4,0]
# Output: [-1,0,3,4,5]

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def insertionSortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        cur, pre = head.next, head 
        
        while cur:
            
            if cur.val >= pre.val:
                pre, cur = cur, cur.next
                continue 
            
            temp = dummy
            while cur.val > temp.next.val:
                temp = temp.next
            
            pre.next = cur.next 
            cur.next = temp.next 
            temp.next = cur 
            cur = pre.next
        
        return dummy.next

# 148. Sort List  MERGE-SORT
# Given the head of a linked list, return the list after sorting it in ascending order.

# Example 1:
# Input: head = [4,2,1,3]
# Output: [1,2,3,4]
# Example 2:
# Input: head = [-1,5,3,4,0]
# Output: [-1,0,3,4,5]
# Example 3:
# Input: head = []
# Output: []

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head 
        
        slow, fast = head, head.next
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        temp = slow.next
        slow.next = None
        l, r = self.sortList(head), self.sortList(temp)
        return self.mergeSort(l, r)
        
    def mergeSort(self, l, r):
        dummy = cur = ListNode(0)
        
        while l and r:
            if l.val <= r.val:
                cur.next = l
                l = l.next
            
            else:
                cur.next = r
                r = r.next
            cur = cur.next 
            
        cur.next = l or r
        
        return dummy.next
    
# 164. Maximum Gap
# Given an integer array nums, return the maximum difference between two successive elements in its sorted form. If the array contains less than two elements, return 0.

# You must write an algorithm that runs in linear time and uses linear extra space.

# Example 1:

# Input: nums = [3,6,9,1]
# Output: 3
# Explanation: The sorted form of the array is [1,3,6,9], either (3,6) or (6,9) has the maximum difference 3.
# Example 2:

# Input: nums = [10]
# Output: 0
# Explanation: The array contains less than 2 elements, therefore return 0.

class Solution:
    def maximumGap(self, nums: List[int]) -> int:
        if len(nums) < 2: return 0
        hi, lo, ans = max(nums), min(nums), 0
        bsize = (hi - lo) // (len(nums) - 1) or 1
    
        buckets = [[] for _ in range(((hi - lo) // bsize) + 1)]
        for n in nums:
            print(n)
            buckets[(n - lo) // bsize].append(n)
        currhi = 0
      
        for b in buckets:
          
            if not len(b): continue
            prevhi, currlo = currhi or b[0], b[0]
            for n in b: 
                currhi, currlo = max(currhi, n), min(currlo, n)
            ans = max(ans, currlo - prevhi)
        return ans


# 169. Majority Element
# Given an array nums of size n, return the majority element.

# The majority element is the element that appears more than ⌊n / 2⌋ times. You may assume that the majority element always exists in the array.
# Example 1:

# Input: nums = [3,2,3]
# Output: 3
# Example 2:

# Input: nums = [2,2,1,1,1,2,2]
# Output: 2

class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        target = None 
        count = 0 
        
        for num in nums:
            if target == num:
                count += 1
            else:
                count -= 1
            
            if count <= 0:
                target = num
                count = 1
        return target

# 1202. Smallest String With Swaps
# ou are given a string s, and an array of pairs of indices in the string pairs where pairs[i] = [a, b] indicates 2 indices(0-indexed) of the string.

# You can swap the characters at any pair of indices in the given pairs any number of times.

# Return the lexicographically smallest string that s can be changed to after using the swaps.
# Example 1:

# Input: s = "dcab", pairs = [[0,3],[1,2]]
# Output: "bacd"
# Explaination: 
# Swap s[0] and s[3], s = "bcad"
# Swap s[1] and s[2], s = "bacd"
# Example 2:

# Input: s = "dcab", pairs = [[0,3],[1,2],[0,2]]
# Output: "abcd"
# Explaination: 
# Swap s[0] and s[3], s = "bcad"
# Swap s[0] and s[2], s = "acbd"
# Swap s[1] and s[2], s = "abcd"
# Example 3:

# Input: s = "cba", pairs = [[0,1],[1,2]]
# Output: "abc"
# Explaination: 
# Swap s[0] and s[1], s = "bca"
# Swap s[1] and s[2], s = "bac"
# Swap s[0] and s[1], s = "abc"

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        def dfs(i):
            visited[i] = True
            component.append(i)
            for j in adj_lst[i]:
                if not visited[j]:
                    dfs(j)
            
        n = len(s)
        adj_lst = [[] for _ in range(n)]
        print(adj_lst)
        for i, j in pairs:
            print(pairs, i , j)
            adj_lst[i].append(j)
            adj_lst[j].append(i)
            
        print(adj_lst)
        visited = [False for _ in range(n)]
        lst = list(s)
        print(lst)
        for i in range(n):
            # print(visited)
            if not visited[i]:
                component = []
                dfs(i)
                print(component)
                component.sort()
                print(component)
                chars = [lst[k] for k in component]
                chars.sort()
                for i in range(len(component)):
                    lst[component[i]] = chars[i]
        return ''.join(lst)