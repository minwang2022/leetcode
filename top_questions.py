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
                    continue
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


# 669 · Coin Change
# Description
# You are given coins of different denominations and a total amount of money amount. Write a function to compute the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

# You may assume that you have an infinite number of each kind of coin.
# It is guaranteed that the num of money will not exceed 10000.
# And the num of coins wii not exceed 500，The denomination of each coin will not exceed 100

# Example
# Example1

# Input: 
# [1, 2, 5]
# 11
# Output: 3
# Explanation: 11 = 5 + 5 + 1
# Example2

# Input: 
# [2]
# 3
# Output: -1

class Solution:
    """
    @param coins: a list of integer
    @param amount: a total amount of money amount
    @return: the fewest number of coins that you need to make up
    """
    def coin_change(self, coins: List[int], amount: int) -> int:
        # write your code here
        INF = 0x3f3f3f3f
                                                              #[ [0],  [1],  [2], [3],  [4],  [5],   [6],  [7],  [8],  [9], [10], [11]]
        dp = [INF for _ in range(amount + 1)]  # 边界条件      #[[inf],[inf],[inf],[inf],[inf],[inf],[inf],[inf],[inf],[inf],[inf],[inf]]

        dp[0] = 0 # 初始化0                                    #[[0],  [1],[inf],[inf],[inf],[inf],[inf],[inf],[inf],[inf],[inf],[inf]]

        for i in range(1, amount + 1):                      # 1 - 11 i = 1

            for coin in coins:                              #[1, 2, 5] , 5

                if i >= coin and dp[i - coin] != INF:       #i = 1, coin = 5 and dp[1-1] = dp[0] == 0

                    dp[i] = min(dp[i], dp[i - coin] + 1)    #dp[1] = 1
                print(dp)
        # 如果不存任意的方案 返回-1

        if dp[amount] == INF:

            return -1

        return dp[amount]

# 257 · Longest String Chain
# Description
# Given a list of words, each word consists of English lowercase letters.
# Let's say word1 is a predecessor of word2 if and only if we can add exactly one letter anywhere in word1 to make it equal to word2.  For example, "abc" is a predecessor of "abac".
# A word chain is a sequence of words [word_1, word_2, ..., word_k] with k >= 1, where word_1 is a predecessor of word_2, word_2 is a predecessor of word_3, and so on.
# Return the longest possible length of a word chain with words chosen from the given list of words.

# 1 <= words.length <= 1000
# 1 <= words[i].length <= 16
# words[i] only consists of English lowercase letters.
# Example
# Exmple 1

# Input: ["ba","a","b","bca","bda","bdca"]
# Output: 4
# Explanation: one of the longest word chain is "a","ba","bda","bdca".

class Solution:
    """
    @param words: the list of word.
    @return: the length of the longest string chain.
    """
    def pre_word(self, a, b):
        if len(a) + 1 != len(b):
            return False
        i = 0
        j = 0
        while i < len(a) and j < len(b):
            if a[i] == b[j]:
                i += 1
            j += 1
        if(i == len(a)):
            return True
        return False
    
    def longestStrChain(self, words):
        dp = [0 for i in range(len(words))]
        ans = 0
        words = sorted(words, key=lambda x: len(x))
        for i in range(len(words)):
            for j in range(i):
                if self.pre_word(words[j], words[i]):
                    dp[i] = int(max(dp[i], dp[j] + 1))
                    ans = int(max(ans, dp[i]))
        return ans + 1

# 941 · Sliding Puzzle
# Description
# On a 2x3 board, there are 5 tiles represented by the integers 1 through 5, and an empty square represented by 0.

# A move consists of choosing 0 and a 4-directionally adjacent number and swapping it.

# The state of the board is solved if and only if the board is [[1,2,3],[4,5,0]].

# Given a puzzle board, return the least number of moves required so that the state of the board is solved. If it is impossible for the state of the board to be solved, return -1.

# board will be a 2 x 3 array as described above.
# board[i][j] will be a permutation of [0, 1, 2, 3, 4, 5].
# Example
# Example 1:

# Given board = `[[1,2,3],[4,0,5]]`, return `1`.

# Explanation: 
# Swap the 0 and the 5 in one move.
# Example 2：

# Given board = `[[1,2,3],[5,4,0]]`, return `-1`.

# Explanation: 
# No number of moves will make the board solved.
# Example 3:

# Given board = `[[4,1,2],[5,0,3]]`, return `5`.

# Explanation: 
# 5 is the smallest number of moves that solves the board.
# An example path:
# After move 0: [[4,1,2],[5,0,3]]
# After move 1: [[4,1,2],[0,5,3]]
# After move 2: [[0,1,2],[4,5,3]]
# After move 3: [[1,0,2],[4,5,3]]
# After move 4: [[1,2,0],[4,5,3]]
# After move 5: [[1,2,3],[4,5,0]]

class Solution:
    """
    @param board: the given board
    @return:  the least number of moves required so that the state of the board is solved
    """
    NEIGHBORS = [[1, 3], [0, 2, 4], [1, 5], [0, 4], [1, 3, 5], [2, 4]]

    def sliding_puzzle(self, board: List[List[int]]) -> int:
        # write your code here
        # 枚举 status 通过一次交换操作得到的状态
        def get(status: str):
            s = list(status)
            x = s.index("0")
            for y in Solution.NEIGHBORS[x]:
                s[x], s[y] = s[y], s[x]
                yield "".join(s)
                s[x], s[y] = s[y], s[x]

        initial = "".join(str(num) for num in sum(board, []))
        if initial == "123450":
            return 0

        q = collections.deque([(initial, 0)])
        seen = {initial}
        while q:
            status, step = q.popleft()
            for next_status in get(status):
                if next_status not in seen:
                    if next_status == "123450":
                        return step + 1
                    q.append((next_status, step + 1))
                    seen.add(next_status)
        
        return -1

# 53 · Reverse Words in a String
# Description
# Given an input string, reverse the string word by word.

# What constitutes a word?
# A sequence of non-space characters constitutes a word and some words have punctuation at the end.
# Could the input string contain leading or trailing spaces?
# Yes. However, your reversed string should not contain leading or trailing spaces.
# How about multiple spaces between two words?
# Reduce them to a single space in the reversed string.
# Example
# Example 1:

# Input:

# s = "the sky is blue"
# Output:

# "blue is sky the"
# Explanation:

# return a reverse the string word by word.
# Example 2:

# Input:

# s = "hello world"
# Output:

# "world hello"

class Solution:
    """
    @param s: A string
    @return: A string
    """
    def reverse_words(self, s: str) -> str:
        # write your code here
        if not s:
            return ""
        s = [word for word in s.split()]
        s.reverse()

        return " ".join(s)

# 1299 · Bulls and Cows
# Description
# You are playing the following Bulls and Cows game with your friend: You write down a number and ask your friend to guess what the number is. Each time your friend makes a guess, you provide a hint that indicates how many digits in said guess match your secret number exactly in both digit and position (called "bulls") and how many digits match the secret number but locate in the wrong position (called "cows"). Your friend will use successive guesses and hints to eventually derive the secret number.

# Write a function to return a hint according to the secret number and friend's guess, use Ato indicate the bulls and B to indicate the cows.

# Please note that both secret number and friend's guess may contain duplicate digits.

# You may assume that the secret number and your friend's guess only contain digits, and their lengths are always equal.

# Example
# Example 1:

# Input：secret = "1807", guess = "7810"
# Output："1A3B"
# Explanation：1 bull and 3 cows. The bull is 8, the cows are 0, 1 and 7.
# Example 2:

# Input：secret = "1123", guess = "0111"
# Output："1A1B"
# Explanation：The 1st 1 in friend's guess is a bull, the 2nd or 3rd 1 is a cow.
class Solution:
    """
    @param secret: An string
    @param guess: An string
    @return: An string
    """
    def get_hint(self, secret: str, guess: str) -> str:
        # write your code here
        bulls = 0
        cntS, cntG = [0] * 10, [0] * 10
        for s, g in zip(secret, guess):
            if s == g:
                bulls += 1
            else:
                cntS[int(s)] += 1
                cntG[int(g)] += 1
        cows = sum(min(s, g) for s, g in zip(cntS, cntG))
        return f'{bulls}A{cows}B'

# 615 · Course Schedule
# Description
# There are a total of n courses you have to take, labeled from 0 to n - 1.

# Before taking some courses, you need to take other courses. For example, to learn course 0, you need to learn course 1 first, which is expressed as [0,1].

# Given the total number of courses and a list of prerequisite pairs, is it possible for you to finish all courses?

# prerequisites may appear duplicated

# Example
# Example 1:

# Input: n = 2, prerequisites = [[1,0]] 
# Output: true
# Example 2:

# Input: n = 2, prerequisites = [[1,0],[0,1]] 
# Output: false

class Solution:
    """
    @param num_courses: a total of n courses
    @param prerequisites: a list of prerequisite pairs
    @return: true if can finish all courses or false
    """
    def can_finish(self, num_courses: int, prerequisites: List[List[int]]) -> bool:
        # write your code here
        indegree = [0] * num_courses
        edges = collections.defaultdict(list)

        for i, j in prerequisites:
            indegree[i] += 1
            edges[j].append(i)

        que = collections.deque()
        visited = 0

        for i in range(num_courses):
            if indegree[i] == 0:
                que.append(i)

        while que:
            crs_num = que.popleft()
            visited += 1
            for course in edges[crs_num]:
                indegree[course] -= 1
                if indegree[course] == 0:
                    que.append(course)

        return visited == num_courses

# 156 · Merge Intervals
# Description
# Given a collection of intervals, merge all overlapping intervals.

# Example
# Example 1:

# Input: [(1,3)]
# Output: [(1,3)]
# Example 2:

# Input:  [(1,3),(2,6),(8,10),(15,18)]
# Output: [(1,6),(8,10),(15,18)]

"""
Definition of Interval:
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
"""

class Solution:
    """
    @param intervals: interval list.
    @return: A new interval list.
    """
    def merge(self, intervals: List[Interval]) -> List[Interval]:
        # write your code here
        intervals = sorted(intervals, key=lambda x: x.start)
        result = []
        for interval in intervals:
            if len(result) == 0 or result[-1].end < interval.start:
                result.append(interval)
            else:
                result[-1].end = max(result[-1].end, interval.end)
        return result

# 860 · Number of Distinct Islands
# Description
# Given a non-empty 2D array grid of 0's and 1's, an island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical). You may assume all four edges of the grid are surrounded by water.

# Count the number of distinct islands. An island is considered to be the same as another if and only if one island has the same shape as another island (and not rotated or reflected).

# Notice that:

# 11
# 1
# and

#  1
# 11
# are considered different island, because we do not consider reflection / rotation.

# The length of each dimension in the given grid does not exceed 50.

# Example
# Example 1:

# Input: 
#   [
#     [1,1,0,0,1],
#     [1,0,0,0,0],
#     [1,1,0,0,1],
#     [0,1,0,1,1]
#   ]
# Output: 3
# Explanation:
#   11   1    1
#   1        11   
#   11
#    1
# Example 2:

# Input:
#   [
#     [1,1,0,0,0],
#     [1,1,0,0,0],
#     [0,0,0,1,1],
#     [0,0,0,1,1]
#   ]
# Output: 1

import collections

DIRECTION = [(1, 0), (-1,0), (0, 1), (0, -1)]


class Solution:
    """
    @param grid: a list of lists of integers
    @return: return an integer, denote the number of distinct islands
    """
    def numberof_distinct_islands(self, grid: List[List[int]]) -> int:
        # write your code here
        n, m = len(grid), len(grid[0])
        que = collections.deque()
        count = 0
        visited = set()
        paths = set()
        for i in range(n):
            for j in range(m):
                if (i, j) in visited: continue 
                if grid[i][j] == 1:
                    que.append((i,j))
                    visited.add((i,j))
                
                    path = ""
                        
                    while que:
                        x, y = que.popleft()
                        for delta_x, delta_y in DIRECTION:
                            new_x, new_y = x + delta_x, y + delta_y
                            
                            if (new_x, new_y) in visited: continue 
                            if self.is_valid(grid, new_x, new_y):
                                visited.add((new_x, new_y))
                                que.append((new_x, new_y))
                                path += str(new_x - i) + str(new_y -j)
                    paths.add(path)

        return len(paths)                            

    def is_valid(self, grid, x, y):
        row, col = len(grid), len(grid[0])
        return x >= 0 and x < row and y >= 0 and y < col and grid[x][y] == 1

