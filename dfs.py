
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
Definition of ListNode:
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
