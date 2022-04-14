# Binary Search

def sampleBinarySearchFunction(self, nums):
    start, end = 0, len(nums) - 1
    while start + 1 < end:
        mid = (start + end) // 2
        
        # find first target
        # if nums[mid] < target:  
        #     start = mid 
        # else:
        #     end = mid 
    # if nums[start] == target:
    #     return start
    # if nums[end] == target:
    #     return end 

        # find last target
        if nums[mid] > target:  
            end = mid 
        else:
            start = mid 
    if nums[end] == target:
        return end
    if nums[start] == target:
        return start     
    return -1 
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
def findPeak(self, A):
    # write your code here
    start, end = 1, len(A) - 2 
    while start + 1 < end:
        mid = (start + end) // 2
        if A[mid] < A[mid - 1]:
            end = mid 
        elif A[mid] < A[mid + 1]:
            start = mid
        else:
            return mid
    return end if A[start] < A[end] else start

# 585 · Maximum Number in Mountain Sequence
# Description
# Given a mountain sequence of n integers which increase firstly and then decrease, find the mountain top(Maximum).
# Arrays are strictly incremented, strictly decreasing

# Example 1:
# Input: nums = [1, 2, 4, 8, 6, 3] 
# Output: 8

# Example 2:
# Input: nums = [10, 9, 8, 7], 
# Output: 10

def mountainSequence(self, nums):
    # write your code here
    start, end = 0, len(nums) - 1
    while start + 1 < end:
        mid = (start + end) // 2
        if nums[mid] < nums[mid + 1]:
            start = mid
        elif nums[mid] < nums[mid - 1]:
            end = mid 
        else:
            return nums[mid]
    return nums[start] if nums[start] > nums[end] else nums[end]

# 183 · Wood Cut
# Description
# Given n pieces of wood with length L[i] (integer array). Cut them into small pieces to guarantee you could have 
# equal or more than k pieces with the same length. What is the longest length you can get from the n pieces of wood? 
# Given L & k, return the maximum length of the small pieces.

# The unit of length is centimeter.The length of the woods are all positive integers,you couldn't cut wood into float length.
# If you couldn't get >= k pieces, return 0.

# Example 1
# Input:
# L = [232, 124, 456]
# k = 7
# Output: 114
# Explanation: We can cut it into 7 pieces if any piece is 114cm long, however we can't cut it into 7 pieces if any piece is 115cm long.

# Example 2
# Input:
# L = [1, 2, 3]
# k = 7
# Output: 0
# Explanation: It is obvious we can't make it.

def woodCut(self, L, k):
    # write your code here
    if not L or sum(L) < k:
        return 0
    start, end = 1, min(max(L), sum(L)//k)
    while start + 1 < end:
        mid = (start + end) // 2
        if self.cut_counts(L, mid) >= k:
            start = mid
        else:
            end = mid 
    
    return end if self.cut_counts(L, end) >= k else start 

def cut_counts(self, L, woodLength):
    return sum([wood // woodLength for wood in L])

# 458 · Last Position of Target
# Description
# Find the last position of a target number in a sorted array. Return -1 if target does not exist.

# Example 1:
# Input: nums = [1,2,2,4,5,5], target = 2
# Output: 2

# Example 2:
# Input: nums = [1,2,2,4,5,5], target = 6
# Output: -1
def lastPosition(self, nums, target):
    # write your code here
    if not nums:
        return -1
    start, end = 0, len(nums) - 1
    while start + 1 < end:
        mid = (start + end) // 2
        if nums[mid] > target:
            end = mid 
        else:
            start = mid 
    if nums[end] == target:
        return end
    if nums[start] == target:
        return start     
    return -1 

# 14 · First Position of Target
# Description
# Given a sorted array (ascending order) and a target number, find the first index of this number in O(log n)O(logn) time complexity.
# If the target number does not exist in the array, return -1.

# Example 1:
# Input:
# tuple = [1,4,4,5,7,7,8,9,9,10]
# target = 1
# Output:
# 0
# Explanation:
# The first index of 1 is 0.
def binarySearch(self, nums, target):
    # write your code here
    if not nums:
        return -1
    start, end = 0, len(nums) - 1
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

# 447 · Search in a Big Sorted Array
# Description
# Given a big sorted array with non-negative integers sorted by non-decreasing order. The array is so big so that you can not get 
# the length of the whole array directly, and you can only access the kth number by ArrayReader.get(k) (or ArrayReader->get(k) for C++).
# Find the first index of a target number. Your algorithm should be in O(log k), where k is the first index of the target number.
# Return -1, if the number doesn't exist in the array.
# If you accessed an inaccessible index (outside of the array), ArrayReader.get will return 2,147,483,647.

# Example 1:
# Input: [1, 3, 6, 9, 21, ...], target = 3
# Output: 1

# Example 2:
# Input: [1, 3, 6, 9, 21, ...], target = 4
# Output: -1

def searchBigSortedArray(self, reader, target):
    # write your code here
    nums = 1
    while reader.get(nums - 1) < target:
        nums = nums * 2
    
    start, end = 0, nums - 1
    while start + 1 < end:
        mid = (start + end) // 2
        if reader.get(mid) < target:
            start = mid
        else:
            end = mid 
    if reader.get(start) == target:
        return start
    if reader.get(end) == target:
        return end 
    return - 1

# 62 · Search in Rotated Sorted Array
# Description
# Suppose a sorted array is rotated at some pivot unknown to you beforehand.
# (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
# You are given a target value to search. If found in the array return its index, otherwise return -1.
# You may assume no duplicate exists in the array.

# Example 1:
# Input:
# array = [4, 5, 1, 2, 3]
# target = 1
# Output:
# 2
# Explanation:
# 1 is indexed at 2 in the array.

def search(self, A, target):
    # write your code here
    if not A:
        return -1 
    start, end = 0, len(A) - 1
    while start + 1 < end:
        mid = (start + end) // 2
        if A[mid] > A[end]:
            if A[start] <= target <= A[mid]:
                end = mid
            else:
                start = mid
        else:
            if A[mid] <= target <= A[end]:
                start = mid 
            else:
                end = mid
    
    if A[start] == target:
        return start
    if A[end] == target:
        return end
    return -1 


# 14 · First Position of Target
# Description
# Given a sorted array (ascending order) and a target number, find the first index of this number in O(log n)O(logn) time complexity.
# If the target number does not exist in the array, return -1.

# Example 1:
# Input:
# tuple = [1,4,4,5,7,7,8,9,9,10]
# target = 1
# Output:
# 0
# Explanation:
# The first index of 1 is 0.

# Example 2:
# Input:
# tuple = [1, 2, 3, 3, 4, 5, 10]
# target = 3
# Output:
# 2
# Explanation:
# The first index of 3 is 2.

from typing import (
    List,
)

class Solution:
    """
    @param nums: The integer array.
    @param target: Target to find.
    @return: The first position of target. Position starts from 0.
    """
    def binary_search(self, nums: List[int], target: int) -> int:
        # write your code here
        if not nums:
            return -1
        start, end = 0, len(nums) -1
        
        while start + 1 < end:
            mid = (start + end ) // 2
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
# Given target, a non-negative integer k and an integer array A sorted in ascending order, find the k closest numbers to target in A, sorted in ascending order by the difference between the number and target. Otherwise, sorted in ascending order by number if the difference is same.
# The value k is a non-negative integer and will no more than the length of the sorted array.
# Length of the given array is positive and will not exceed 10^410 
# Absolute value of elements in the array will not exceed 10^410 
# Example
# Example 1:

# Input: A = [1, 2, 3], target = 2, k = 3
# Output: [2, 1, 3]
# Example 2:

# Input: A = [1, 4, 6, 8], target = 3, k = 3
# Output: [4, 1, 6]

from typing import (
    List,
)

class Solution:
    """
    @param a: an integer array
    @param target: An integer
    @param k: An integer
    @return: an integer array
    """
    def k_closest_numbers(self, a: List[int], target: int, k: int) -> List[int]:
        # write your code here
        res = []
        if not a:
            return res
        
        start, end = 0, len(a) -1

        while start + 1 < end:
            mid = (start + end) // 2
            if a[mid] < target:
                start = mid 
            else:
                end = mid 

        for _ in range(k):

            if self.is_left(start, end, target, a):
                res.append(a[start])
                start -= 1
            else:
                res.append(a[end])
                end += 1

        return res 

    def is_left(self, start, end, target, a):
        if start < 0:
            return False 
        if end >= len(a):
            return True 
        
        return abs(a[start] - target) <= abs(a[end] - target)