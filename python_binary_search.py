# Binary Search

# 457. Classical Binary Search
# Description
# Find any position of a target number in a sorted array. Return -1 if target does not exist.

# Example
# Example 1:
# Input: nums = [1,2,2,4,5,5], target = 2
# Output: 1 or 2

# Example 2:
# Input: nums = [1,2,2,4,5,5], target = 6
# Output: -1

def findPosition(self, nums, target):
    # write your code here
    if not nums:
        return -1
    start, end = 0 , len(nums) - 1
    while start + 1 < end:
        mid = (start + end) // 2

        if nums[mid] < target:
            start = mid 
        else:
            end = mid 
    if nums[start] == target:
        return start
    if nums[end] == target:
        return end 
    return -1 

# 460 · Find K Closest Elements
# Description
# Given target, a non-negative integer k and an integer array A sorted in ascending order, find the k closest numbers to 
# target in A, sorted in ascending order by the difference between the number and target. Otherwise, sorted in ascending 
# order by number if the difference is same.

# The value k is a non-negative integer and will always be smaller than the length of the sorted array.
# Length of the given array is positive and will not exceed 10^4
# Absolute value of elements in the array will not exceed 10^4
# Example 1:
# Input: A = [1, 2, 3], target = 2, k = 3
# Output: [2, 1, 3]

# Example 2:
# Input: A = [1, 4, 6, 8], target = 3, k = 3
# Output: [4, 1, 6]

class Solution:
    """
    @param A: an integer array
    @param target: An integer
    @param k: An integer
    @return: an integer array
    """
    def kClosestNumbers(self, A, target, k):
        # write your code here
        results = []
        right = self.findClosestRight(A, target)
        left = right - 1
        for _ in range(k):
            if self.isLeft(A, left, right, target):
                results.append(A[left])
                left -= 1
            else:
                results.append(A[right])
                right += 1

        return results 

    def findClosestRight(self, nums, target):
        start, end = 0, len(nums) - 1
        while start + 1 < end:
            mid = (start + end) // 2
            if nums[mid] < target:
                start = mid
            else:
                end = mid
        if nums[start] >= target:
            return start  
        if nums[end] >= target:
            return end  

        return len(nums)

    def isLeft(self, nums, left, right, target):
        if left < 0:
            return False
        if right >= len(nums):
            return True 
        return target - nums[left] <= nums[right] - target

# 159 · Find Minimum in Rotated Sorted Array
# Description
# Suppose a sorted array in ascending order is rotated at some pivot unknown to you beforehand.
# (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
# Find the minimum element.
# You can assume no duplicate exists in the array.

# Example 1:
# Input：[4, 5, 6, 7, 0, 1, 2]
# Output：0
# Explanation：
# The minimum value in an array is 0.

# Example 2:
# Input：[2,1]
# Output：1
# Explanation：
# The minimum value in an array is 1.

def findMin(self, nums):
        # write your code here
        if not nums:
            return -1
        start, end = 0, len(nums) - 1
        while start + 1 < end:
            mid = (start + end) // 2
            if nums[mid] > nums[end]:
                start = mid 
            else:
                end = mid
        return min(nums[start], nums[end])

# 75 · Find Peak Element
# Description
# There is an integer array which has the following features:
# The numbers in adjacent positions are different.
# A[0] < A[1] && A[A.length - 2] > A[A.length - 1].
# We define a position P is a peak if:
# A[P] > A[P-1] && A[P] > A[P+1]
# Find a peak element in this array. Return the index of the peak.
# It's guaranteed the array has at least one peak.
# The array may contain multiple peeks, find any of them.
# The array has at least 3 numbers in it.

# Example 1:
# Input:
# A = [1, 2, 1, 3, 4, 5, 7, 6]
# Output:
# 1
# Explanation:
# Returns the index of any peak element. 6 is also correct.
