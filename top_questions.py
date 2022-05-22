# 780 · Remove Invalid Parentheses
# Description
# Remove the minimum number of invalid parentheses in order to make the input string valid. Return all possible results.

# The input string may contain letters other than the parentheses ( and ).

# Example 1:

# Input:
# "()())()"
# Ouput:
# ["(())()","()()()"]
# Example 2:

# Input:
# "(a)())()"
# Output:
#  ["(a)()()", "(a())()"]
# Example 3:

# Input:
# ")(" 
# Output:
#  [""]

from typing import (
    List,
)

class Solution:
    """
    @param s: The input string
    @return: Return all possible results
             we will sort your return value in output
    """
    def removeInvalidParentheses(self, s):
        res = []
        if len(s) == 0:
            res.append("")
            return res
        q = []
        S = set()
        q.append(s)
        S.add(s)
        flag = False
        # bfs
        while len(q) > 0:
            curStr = q.pop(0)
            # 若curStr有效，则找到答案
            if self.check(curStr):
                res.append(curStr)
                flag = True
            #不需要去找更短的串
            if flag:
                continue
            # 对于无效的字符串，依次删除一个字符压入队列
            for i in range(len(curStr)):
                if curStr[i] != '(' and curStr[i] != ')':
                    continue;
                str1 = curStr[:i]
                str2 = curStr[i + 1:]
                str3 = str1 + str2
                sizeSet = len(S)
                S.add(str3)
                # 如果这个字符串未被check过，则压入队列
                if sizeSet != len(S):
                    q.append(str3)
        return res

    def check(self,str):
        if len(str) == 0:
            return True
        # 记录count对
        count = 0
        for i in range(0,len(str)):
            if str[i] == '(':
                count += 1
            if str[i] == ')':
                count -= 1
            if count < 0:
                return False
        if count == 0:
            return True
        return False
    
# 653 · Expression Add Operators
# Description
# Given a string that contains only digits 0-9 and a target value, return all possibilities to add binary operators (not unary) +, -, or * between the digits so they evaluate to the target value.
# The number does not contain the leading 0.

# Example
# Example 1:

# Input:
# "123"
# 6
# Output: 
# ["1*2*3","1+2+3"]
# Example 2:

# Input:
# "232"
# 8
# Output: 
# ["2*3+2", "2+3*2"]
# Example 3:

# Input:
# "105"
# 5
# Output:
# ["1*0+5","10-5"]

from typing import (
    List,
)

class Solution:
    """
    @param num: a string contains only digits 0-9
    @param target: An integer
    @return: return all possibilities
             we will sort your return value in output
    """
    def add_operators(self, num: str, target: int) -> List[str]:
        # write your code here
        def dfs(idx, tmp, tot, last, res):
            if idx == len(num):
                if tot == target:
                    res.append(tmp)
                return
            for i in range(idx, len(num)):
                x = int(num[idx: i + 1])
                if idx == 0:
                    dfs(i + 1, str(x), x, x, res)
                else:
                    dfs(i + 1, tmp + "+" + str(x), tot + x, x, res)
                    dfs(i + 1, tmp + "-" + str(x), tot - x, -x, res)
                    dfs(i + 1, tmp + "*" + str(x), tot - last + last * x, last * x, res)
                if x == 0:
                    break
        res = []
        dfs(0, "", 0, 0, res)
        return res


# 86 · Binary Search Tree Iterator
# Description
# Design an iterator over a binary search tree with the following rules:
# Next() returns the next smallest element in the BST.

# Elements are visited in ascending order (i.e. an in-order traversal)
# next() and hasNext() queries run in O(1)O(1) time in average.
# Example
# Example 1:

# Input:

# tree = {10,1,11,#,6,#,12}
# Output:

# [1,6,10,11,12]

"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

Example of iterate a tree:
iterator = BSTIterator(root)
while iterator.hasNext():
    node = iterator.next()
    do something for node 
"""


class BSTIterator:
    """
    @param: root: The root of binary tree.
    """
    def __init__(self, root):
        # do intialization if necessary
        self.stack = []
        self.curt = root

    #@return: True if there has next node, or false
    def hasNext(self):
        return self.curt is not None or len(self.stack) > 0

    #@return: return next node
    def _next(self):                                #[10]
        while self.curt is not None:                #none, 
            self.stack.append(self.curt)            #[10], 
            self.curt = self.curt.left              # none 
            
        self.curt = self.stack.pop()                #[],  10
        nxt = self.curt                             #nxt = 10
        self.curt = self.curt.right                 #[], 11
        return nxt                                  #1, 6, 10


# 1704 · Range Sum of BST
# Description
# Given the root node of a binary search tree, return the sum of values of all nodes with value between L and R (inclusive).

# The binary search tree is guaranteed to have unique values.

# The number of nodes in the tree is at most 10000.
# The final answer is guaranteed to be less than 2^31.

# Example 1:

# Input: root = [10,5,15,3,7,null,18], L = 7, R = 15
# Output: 32
# Example 2:

# Input: root = [10,5,15,3,7,13,18,1,null,6], L = 6, R = 10
# Output: 23

from lintcode import (
    TreeNode,
)

"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: the root node
    @param l: an integer
    @param r: an integer
    @return: the sum
    """
    def range_sum_b_s_t(self, root: TreeNode, l: int, r: int) -> int:
        # write your code here.
        
        def dfs(node):
            if node:
                if l <= node.val <= r:
                    self.tot += node.val
                if l < node.val:
                    dfs(node.left)
                if r > node.val:
                    dfs(node.right)
        
        self.tot = 0
        dfs(root)
        return self.tot