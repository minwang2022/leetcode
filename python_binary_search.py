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

