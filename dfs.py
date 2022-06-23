
# 822 · Reverse Order Storage
# Description
# Give a linked list, and store the values of linked list in reverse order into an array.

# Contact me on wechat to get Amazon、Google requent Interview questions . (wechat id : jiuzhang0607)


# You can not change the structure of the original linked list.
# ListNode have two elements: ListNode.val and ListNode.next
# Example
# Example1

# Input: 1 -> 2 -> 3 -> null
# Output: [3,2,1]
# Example2

# Input: 4 -> 2 -> 1 -> null
# Output: [1,2,4]

class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
# """

class Solution:
    # """
    # @param head: the given linked list
    # @return: the array that store the values in reverse order 
    # """
    def reverse_store(self, head: ListNode) -> List[int]:
        # write your code here
        ans = []
        self.helper(head, ans)
        return ans 
    
    def helper(self, head, ans):
        if head == None:
            return 
        
        self.helper(head.next, ans)
        ans.append(head.val)

# 771 · Double Factorial
# Description
# Given a number n, return the double factorial of the number.In mathematics, the product of all the integers from 1 up to some non-negative integer n that have the same parity (odd or even) as n is called the double factorial.

# Contact me on wechat to get Amazon、Google requent Interview questions . (wechat id : jiuzhang0607)


# We guarantee that the result does not exceed long.
# n is a positive integer
# Example
# Example1 :

# Input: n = 5
# Output: 15
# Explanation:
# 5!! = 5 * 3 * 1 = 15
# Example2:

# Input: n = 6
# Output: 48
# Explanation:
# 6!! = 6 * 4 * 2 = 48
class Solution:
    """
    @param n: the given number
    @return:  the double factorial of the number
    """

    def double_factorial(self, n: int) -> int:
        # Write your code here
        res = 1 
        
 
        return self.helper(n, res)
    
    def helper(self, n, res):
        if n <= 0:
            return res

        return self.helper(n - 2, res * n)

# 451 · Swap Nodes in Pairs
# Description
# Given a linked list, swap every two adjacent nodes and return its head.

# Contact me on wechat to get Amazon、Google requent Interview questions . (wechat id : jiuzhang0607)


# Example
# Example 1:

# Input: 1->2->3->4->null
# Output: 2->1->4->3->null
# Example 2:

# Input: 5->null
# Output: 5->null
# Definition of ListNode:
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
# """

class Solution:
    """
    @param head: a ListNode
    @return: a ListNode
    """
    def swap_pairs(self, head: ListNode) -> ListNode:
        # write your code here
        if not head or not head.next:
            return head
        newHead = head.next

        head.next = self.swap_pairs(newHead.next)

        newHead.next = head

        return newHead

# 464 · Sort Integers II
# Description
# Given an integer array, sort it in ascending order in place. Use quick sort, merge sort, heap sort or any O(nlogn) algorithm.

# Contact me on wechat to get Amazon、Google requent Interview questions . (wechat id : jiuzhang0607)


# Example
# Example1:

# Input: [3, 2, 1, 4, 5], 
# Output: [1, 2, 3, 4, 5].
# Example2:

# Input: [2, 3, 1], 
# Output: [1, 2, 3].

class Solution:
    """
    @param a: an integer array
    @return: nothing
    """
    def sort_integers2(self, a: List[int]):
        # write your code here
        # return self.quickSort(a, 0, len(a) -1)
        temp = [0] * len(a)
        self.mergeSort(a, 0, len(a) - 1, temp)
        return a
    def mergeSort(self, a, start, end, temp):
        if start >= end:
            return 
        
        mid = (start + end )// 2
        self.mergeSort(a, start, mid, temp)
        self.mergeSort(a, mid + 1, end, temp)

        left, right = start, mid + 1
        idx = start
        while left <= mid and right <= end:
           
            if a[left] < a[right]:
                temp[idx] = a[left]
                idx += 1
                left += 1
            else:
                temp[idx] = a[right]
                idx += 1
                right += 1
        
        while left <= mid:
            temp[idx] = a[left]
            idx += 1
            left += 1
        while right <= end:
            temp[idx] = a[right]
            idx += 1
            right += 1
        
        for i in range(start, end + 1):
            a[i] = temp[i]

    # def quickSort(self, a, start, end):
    #     if start >= end:
    #         return 
        
    #     left, right = start, end
    #     mid = a[(left + right) // 2]

    #     while left <= right:
    #         while left <= right and a[left] < mid:
    #             left += 1
    #         while left <= right and a[right] > mid:
    #             right -= 1
            
    #         if left <= right:
    #             a[left], a[right] = a[right], a[left]
    #             left += 1 
    #             right -= 1
        
    #     self.quickSort(a, start, right)
    #     self.quickSort(a, left, end)


# 97 · Maximum Depth of Binary Tree
# Description
# Given a binary tree, find its maximum depth.

# The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

# Contact me on wechat to get Amazon、Google requent Interview questions . (wechat id : jiuzhang0607)


# The answer will not exceed 5000

# Example
# Example 1:

# Input:

# tree = {}
# Output:

# 0
# Explanation:

# The height of empty tree is 0.

# Example 2:

# Input:

# tree = {1,2,3,#,#,4,5}
# Output:

# 3

"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: The root of binary tree.
    @return: An integer
    """
    def max_depth(self, root: TreeNode) -> int:
        # write your code here
        if not root:
            return 0 
        
        return max(self.max_depth(root.left), self.max_depth(root.right)) + 1