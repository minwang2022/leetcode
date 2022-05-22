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

        # if not root:
        #     return 0
        
        # if root.val < l:
        #     return self.range_sum_b_s_t(root.right, l, r)
            
        # elif root.val > r:
        #     return self.range_sum_b_s_t(root.left, l, r)
            
        # else: 
        #     return root.val + self.range_sum_b_s_t(root.left, l, r) + self.range_sum_b_s_t(root.right, l, r)


# 1506 · All Nodes Distance K in Binary Tree

# Description
# We are given a binary tree (with root node root), a target node, and an integer value K.

# Return a list of the values of all nodes that have a distance K from the target node. The answer can be returned in any order.

# The given tree is non-empty and has k nodes at least.
# Each node in the tree has unique values 0 <= node.val <= 500.
# The target node is a node in the tree.
# 0 <= K <= 1000.
# Example
# Example 1:

# Input:
# {3,5,1,6,2,0,8,#,#,7,4}
# 5
# 2

# Output: [7,4,1]

# Explanation: 
# The nodes that are a distance 2 from the target node (with value 5)
# have values 7, 4, and 1.

from typing import (
    List,
)
from lintcode import (
    TreeNode,
)
from collections import deque
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: the root of the tree
    @param target: the target
    @param k: the given K
    @return: All Nodes Distance K in Binary Tree
             we will sort your return value in output
    """
    def distance_k(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        # Write your code here
        graph = {}
        self.build_graph(graph, None, root)
        
        # bfs
        queue = deque([target])
        visited = set([target])
        for _ in range(k):
            size = len(queue)
            for i in range(size):
                node = queue.popleft()
                for neighbor in graph[node]:
                    if neighbor in visited: continue
                    queue.append(neighbor)
                    visited.add(neighbor)
        return [x.val for x in queue]

    def build_graph(self, graph, parent, root):
        if root is None:
            return
        self.add_edge(graph, root, root.left)
        self.add_edge(graph, root, root.right)
        self.add_edge(graph, root, parent)
        self.build_graph(graph, root, root.left)
        self.build_graph(graph, root, root.right)

    def add_edge(self, graph, a, b):
        # add an edge a->b
        if b is None:
            return
        if a not in graph:
            graph[a] = set()
        graph[a].add(b)
    
# 1871 · Maximum moment
# Description
# Give you a 24-hour time (00: 00-23: 59), where one or more numbers of the four numbers are question marks. Question mark can be replaced with any number, then what is the maximum time you can represent.

# Example 1:

# Input: 
# time = "2?:00"
# Output: 
# "23:00"
# Example 2:

# Input: 
# time = "??:??"
# Output: 
# "23:59"

class Solution:
    """
    @param time: a string of Time
    @return: The MaximumMoment
    """
    def maximum_moment(self, time: str) -> str:
        # Write your code here.
        answer = ""
        if(time[0] == '?'):
            if(time[1] <= '9' and time[1] >= '4'):
                answer += '1'
            else:
                answer += '2'
        else:
            answer += time[0]
        if(time[1] == '?'):
            if(answer[0] != '2'):
                answer += '9'
            else:
                answer += '3'
        else:
            answer += time[1]
        answer += ':'
        if(time[3] == '?'):
            answer += '5'
        else:
            answer += time[3]
        if(time[4] == '?'):
            answer += '9'
        else:
            answer += time[4]
        return answer

# 407 · Plus One
# Description
# Given a non-negative number represented as an array of digits, plus one to the number.Returns a new array.

# The number is arranged according to the number of digits, with the highest digit at the top of the list.

# Example
# Example 1:

# Input: [1,2,3]
# Output: [1,2,4]
# Example 2:

# Input: [9,9,9]
# Output: [1,0,0,0]

from typing import (
    List,
)

class Solution:
    """
    @param digits: a number represented as an array of digits
    @return: the result
    """
    #O(n) time, O(n) space
    def plus_one(self, digits: List[int]) -> List[int]:
        # write your code here
        num = int("".join([str(digit) for digit in digits]))
        num += 1 
        return [int(ele) for ele in str(num)]

    #O(n) time , O(1) space
    def plus_one1(self, digits: List[int]) -> List[int]:    
        # write your code here
        n = len(digits)
        for i in range(n - 1, -1, -1):
            if digits[i] != 9:
                digits[i] += 1
                for j in range(i + 1, n):
                    digits[j] = 0
                return digits

        # 元素均为 9 的话需要向前进一位
        return [1] + [0] * n

# 514 · Paint Fence
# Description
# There is a fence with n posts, each post can be painted with one of the k colors.
# You have to paint all the posts such that no more than two adjacent fence posts have the same color.
# Return the total number of ways you can paint the fence.

# n and k are non-negative integers.

# Example
# Example 1:

# Input: n=3, k=2  
# Output: 6
# Explanation:
#           post 1,   post 2, post 3
#     way1    0         0       1 
#     way2    0         1       0
#     way3    0         1       1
#     way4    1         0       0
#     way5    1         0       1
#     way6    1         1       0

class Solution:
    """
    @param n: non-negative integer, n posts
    @param k: non-negative integer, k colors
    @return: an integer, the total number of ways
    """
    def num_ways(self, n: int, k: int) -> int:
        # write your code here
        if n == 0: return 0
        if n == 1: return k 
        if n == 2: return k*k
        if k == 1: return 0
        
        lo, hi = k, k*k 
        for _ in range(n-2):
            lo, hi = hi, (k-1)*hi + (k-1)*lo 
        return hi

# 888 · Valid Word Square
# Description
# Given a sequence of words, check whether it forms a valid word square.

# A sequence of words forms a valid word square if the k^th row and column read the exact same string, where 0 ≤ k < max(numRows, numColumns).

# The number of words given is at least 1 and does not exceed 500.
# Word length will be at least 1 and does not exceed 500.
# Each word contains only lowercase English alphabet a-z.
# Example
# Example1

# Input:  
# [
#   "abcd",
#   "bnrt",
#   "crmy",
#   "dtye"
# ]
# Output: true
# Explanation:
# The first row and first column both read "abcd".
# The second row and second column both read "bnrt".
# The third row and third column both read "crmy".
# The fourth row and fourth column both read "dtye".

# Therefore, it is a valid word square.

class Solution:
    """
    @param words: a list of string
    @return: a boolean
    """
    def valid_word_square(self, words: List[str]) -> bool:
        # Write your code here
        for i in range(len(words)):
            for j in range(i + 1, len(words[i])):
                if words[i][j] != words[j][i]:
                    return False
        return True
        
# 914 · Flip Game
# Description
# You are playing the following Flip Game with your friend: Given a string that contains only two characters: + and -, you can flip two consecutive "++" into "--", you can only flip one time. Please find all strings that can be obtained after one flip.

# Write a program to find all possible states of the string after one valid move.

# Example
# Example1

# Input:  s = "++++"
# Output: 
# [
#   "--++",
#   "+--+",
#   "++--"
# ]
# Example2

# Input: s = "---+++-+++-+"
# Output: 
# [
# 	"---+++-+---+",
# 	"---+++---+-+",
# 	"---+---+++-+",
# 	"-----+-+++-+"
# ]

class Solution:
    """
    @param s: the given string
    @return: all the possible states of the string after one valid move
             we will sort your return value in output
    """
    def generate_possible_next_moves(self, s: str) -> List[str]:
        # write your code here
        res, n = [], len(s)

        for i in range(n - 1):
            if s[i] == "+" and s[i + 1] == "+":
                res.append(s[:i] + "--" + s[i+ 2:])
        
        return res 