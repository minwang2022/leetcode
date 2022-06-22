
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
