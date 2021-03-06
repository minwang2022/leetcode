
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

# 68 · Binary Tree Postorder Traversal
# Description
# Given a binary tree, return the postorder traversal of its nodes’ values.
# Input:

# binary tree = {1,2,3}
# Output:

# [2,3,1]

class Solution:
    """
    @param root: A Tree
    @return: Preorder in ArrayList which contains node values.
    """
    def preorder_traversal(self, root: TreeNode) -> List[int]:
        # write your code here
        def helper(root):
            if not root:
                return 
            res.append(root.val)
            helper(root.left)
            helper(root.right)
        
        res = []
        helper(root)
        return res  

# 1596 · Possible Bipartition
# Description
# Given a set of N people (numbered 1, 2, ..., N), we would like to split everyone into two groups of any size.

# Each person may dislike some other people, and they should not go into the same group.

# Formally, if dislikes[i] = [a, b], it means it is not allowed to put the people numbered a and b into the same group.

# Return true if and only if it is possible to split everyone into two groups in this way.Otherwise, return false.

# Contact me on wechat to get Amazon、Google requent Interview questions . (wechat id : jiuzhang0607)


# 1 <= N <= 2000
# 0 <= dislikes.length <= 10000
# 1 <= dislikes[i][j] <= N
# dislikes[i][0] < dislikes[i][1]
# There does not exist i != j for which dislikes[i] == dislikes[j].

# Example
# Example 1:

# Input: N = 4, dislikes = [[1,2],[1,3],[2,4]]
# Output: true
# Explanation: group1 [1,4], group2 [2,3]
# Example 2:

# Input: N = 3, dislikes = [[1,2],[1,3],[2,3]]
# Output: false
# Example 3:

# Input: N = 5, dislikes = [[1,2],[2,3],[3,4],[4,5],[1,5]]
# Output: false

class Solution:
    """
    @param n:  sum of the set
    @param dislikes: dislikes peoples
    @return:  if it is possible to split everyone into two groups in this way
    """
    def possible_bipartition(self, n: int, dislikes: List[List[int]]) -> bool:
        # Write your code here.
        graph = collections.defaultdict(list)
        for u, v in dislikes:
            graph[u].append(v)
            graph[v].append(u)

        color = {}
        def dfs(node, c = 0):
            if node in color:
                return color[node] == c
            color[node] = c
            for nei in graph[node]:
                if not dfs(nei, c ^ 1):
                    return False 
            return True  

        for node in range(1, n+1):
            if node not in color and not dfs(node):
                return False 
        
        return True 
        
# 246 · Binary Tree Path Sum II
# Description
# Given a binary tree and a target value, design an algorithm to find all paths in the binary tree that sum to that target value. The path can start and end at any node, but it needs to be a route that goes all the way down. That is, the hierarchy of nodes on the path is incremented one by one.

# Contact me on wechat to get Amazon、Google requent Interview questions . (wechat id : jiuzhang0607)


# Example
# Example 1:

# Input:
# {1,2,3,4,#,2}
# 6
# Output:
# [[2, 4],[1, 3, 2]]
# Explanation:
# The binary tree is like this:
#     1
#    / \
#   2   3
#  /   /
# 4   2
# for target 6, it is obvious 2 + 4 = 6 and 1 + 3 + 2 = 6.
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: the root of binary tree
    @param target: An integer
    @return: all valid paths
             we will sort your return value in output
    # """
    def binary_tree_path_sum2(self, root: TreeNode, target: int) -> List[List[int]]:
        # write your code here
        res = []
        
        def helper(root, target, path, length):
            if not root:
                return 
            path.append(root.val)
            temp = target
            for i in range(length, -1, -1):
                temp -= path[i]
                if temp == 0:
                    res.append(path[i:])
                
            helper(root.left, target, path, length + 1 )
            helper(root.right, target, path, length + 1 )
            path.pop()
            
                
        helper(root, target, [], 0)

        return res  

# 376 · Binary Tree Path Sum
# Description
# Given a binary tree, find all paths that sum of the nodes in the path equals to a given number target.

# A valid path is from root node to any of the leaf nodes.

# Contact me on wechat to get Amazon、Google requent Interview questions . (wechat id : jiuzhang0607)


# Example
# Example 1:

# Input:
# {1,2,4,2,3}
# 5
# Output: [[1, 2, 2],[1, 4]]
# Explanation:
# The tree is look like this:
# 	     1
# 	    / \
# 	   2   4
# 	  / \
# 	 2   3
# For sum = 5 , it is obviously 1 + 2 + 2 = 1 + 4 = 5
# Example 2:

# Input:
# {1,2,4,2,3}
# 3
# Output: []
# Explanation:
# The tree is look like this:
# 	     1
# 	    / \
# 	   2   4
# 	  / \
# 	 2   3
# Notice we need to find all paths from root node to leaf nodes.
# 1 + 2 + 2 = 5, 1 + 2 + 3 = 6, 1 + 4 = 5 
# There is no one satisfying it.

class Solution:
    """
    @param root: the root of binary tree
    @param target: An integer
    @return: all valid paths
             we will sort your return value in output
    """
    def binary_tree_path_sum(self, root: TreeNode, target: int) -> List[List[int]]:
        # write your code here
        res = []
        
        def dfs(root,target, path, l, res):
            if not root:
                return 
            
            path.append(root.val)
            
            if target - root.val == 0:
                res.append(path[:])

            dfs(root.left, target - root.val, path, l + 1,res )
            dfs(root.right, target - root.val, path, l + 1, res)

            path.pop()
        
        dfs(root, target, [], 0, res)
        return res 

# 94 · Binary Tree Maximum Path Sum

# Description
# Given a binary tree, find the maximum path sum.
# The path may start and end at any node in the tree.
# (Path sum is the sum of the weights of nodes on the path between two nodes.)

# Contact me on wechat to get Amazon、Google requent Interview questions . (wechat id : jiuzhang0607)


# About the tree representation

# Example
# Example 1:

# Input:

# tree = {2}
# Output:

# 2
# Explanation:

# There is only one node 2
# Example 2:

# Input:

# tree = {1,2,3}
# Output:

# 6
class Solution:
    """
    @param root: The root of binary tree.
    @return: An integer
    """

    def max_path_sum(self, root: TreeNode) -> int:
        # write your code here
        max_num = [float("-inf")]
        def dfs(root):
            if not root:
                return 0 
            left = max(dfs(root.left), 0)
            right = max(dfs(root.right), 0)

            max_num[0] = max(max_num[0], left + right + root.val)

            return max(root.val + left, root.val + right)

        
        dfs(root)

        return max_num[0]

# 72 · Construct Binary Tree from Inorder and Postorder Traversal
# Description
# Given inorder and postorder traversal of a tree, construct the binary tree.

# Contact me on wechat to get Amazon、Google requent Interview questions . (wechat id : jiuzhang0607)


# You may assume that duplicates do not exist in the tree.

# Example
# Example 1:

# Input:

# inorder traversal = []
# postorder traversal = []
# Output:

# {}
# Explanation:

# Binary tree is empty

# Example 2:

# Input:

# inorder traversal = [1,2,3]
# postorder traversal = [1,3,2]
# Output:

# {2,1,3}

"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param inorder: A list of integers that inorder traversal of a tree
    @param postorder: A list of integers that postorder traversal of a tree
    @return: Root of a tree
    """
    def build_tree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        # write your code here
        if not inorder: return None 
        root = TreeNode(postorder[-1])
        rootIndex = inorder.index(postorder[-1])

        root.left = self.build_tree(inorder[:rootIndex], postorder[:rootIndex])
        root.right = self.build_tree(inorder[rootIndex + 1:], postorder[rootIndex:-1])
        return root

# 73 · Construct Binary Tree from Preorder and Inorder Traversal
class Solution:
    """
    @param preorder: A list of integers that preorder traversal of a tree
    @param inorder: A list of integers that inorder traversal of a tree
    @return: Root of a tree
    """
    def build_tree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        # write your code here
        if not preorder and not inorder:
            return None 
        
        root = TreeNode(preorder[0])
        rootPos = inorder.index(preorder[0])
        root.left = self.build_tree(preorder[1:rootPos + 1], inorder[:rootPos])
        root.right = self.build_tree(preorder[rootPos + 1:], inorder[rootPos + 1:])
        return root

#22 · Flatten List
# Description
# Given a list, each element in the list can be a list or an integer.Flatten it into a simply list with integers.

# Contact me on wechat to get Amazon、Google requent Interview questions . (wechat id : jiuzhang0607)


# If the element in the given list is a list, it can contain list too.

# Example
# Example 1:

# Input:

# list = [[1,1],2,[1,1]]
# Output:

# [1,1,2,1,1]
# Explanation:

# Flatten it into a simply list with integers.

class Solution(object):

    # @param nestedList a list, each element in the list 
    # can be a list or integer, for example [1,2,[1,2]]
    # @return {int[]} a list of integer
    def flatten(self, nestedList):
        # Write your code here
        res = []

        for i in range(len(nestedList)):
            if isinstance(nestedList[i], int):
                res.append(nestedList[i])
            else:
                res.extend(self.flatten(nestedList[i]))
        return res 

# 10 · String Permutation II
# Description
# Given a string, find all permutations of it without duplicates.

# Contact me on wechat to get Amazon、Google requent Interview questions . (wechat id : jiuzhang0607)


# Example
# Example 1:

# Input:

# s = "abb"
# Output:

# ["abb", "bab", "bba"]
# Explanation:

# There are six kinds of full arrangement of abb, among which there are three kinds after removing duplicates.

# Example 2:

# Input:

# s = "aabb"
# Output:

# ["aabb", "abab", "baba", "bbaa", "abba", "baab"]

class Solution:
    """
    @param str: A string
    @return: all permutations
             we will sort your return value in output
    """
    def string_permutation2(self, str: str) -> List[str]:
        # write your code here

        def dfs(chars, path):
            if len(chars) == len(path):
                res.append("".join(path))
                return  
            n = len(chars)
            
            for i in range(n):
                if visited[i]:
                    continue
                if i > 0 and chars[i] == chars[i - 1] and not visited[i - 1]:
                    continue
                
                path.append(chars[i])
                visited[i] = True

                dfs(chars, path)

                path.pop()
                visited[i] = False

        res = []
        chars = sorted(list(str))
        visited = [False] * len(chars)

        dfs(chars, [])
        return res
   
   
#    371 · Print Numbers by Recursion
# Description
# Print numbers from 1 to the largest number with N digits by recursion.

# Contact me on wechat to get Amazon、Google requent Interview questions . (wechat id : jiuzhang0607)


# It's pretty easy to do recursion like:

# recursion(i) {
#     if i > largest number:
#         return
#     results.add(i)
#     recursion(i + 1)
# }
# however this cost a lot of recursion memory as the recursion depth maybe very large. Can you do it in another way to recursive with at most N depth?

# Example
# Example 1:

# Input : N = 1 
# Output :[1,2,3,4,5,6,7,8,9]
# Example 2:

# Input : N = 2 
# Output :[1,2,3,4,5,6,7,8,9,10,11,12,...,99]

class Solution:
    """
    @param n: An integer
    @return: An array storing 1 to the largest number with n digits.
    """
    def numbers_by_recursion(self, n: int) -> List[int]:
        # write your code here

        res = []
        
        def dfs(n, idx):
            if idx >=  n:
                return 
            
            for i in range(10 ** idx, 10 ** (idx + 1)):
                res.append(i)
            dfs(n, idx + 1)
        
        dfs(n, 0)
        return res

# 135 · Combination Sum
# Description
# Given a set of candidate numbers candidates and a target number target. Find all unique combinations in candidates where the numbers sums to target.

# The same repeated number may be chosen from candidates unlimited number of times.

# Contact me on wechat to get Amazon、Google requent Interview questions . (wechat id : jiuzhang0607)


# All numbers (including target) will be positive integers.
# Numbers in a combination a1, a2, … , ak must be in non-descending order. (ie, a1 ≤ a2 ≤ … ≤ ak)
# Different combinations can be in any order.
# The solution set must not contain duplicate combinations.
# Example
# Example 1:

# Input: candidates = [2, 3, 6, 7], target = 7
# Output: [[7], [2, 2, 3]]
# Example 2:

# Input: candidates = [1], target = 3
# Output: [[1, 1, 1]]
class Solution:
    """
    @param candidates: A list of integers
    @param target: An integer
    @return: A list of lists of integers
             we will sort your return value in output
    """
    def combination_sum(self, candidates: List[int], target: int) -> List[List[int]]:
        # write your code here
        combinations = []
        newCandidates = sorted(list(set(candidates)))
        def dfs(newCandidates,target, start, combination):
            if target < 0:
                return 
            if target == 0:
                return combinations.append(combination[:])

            for i in range(start, len(newCandidates)):
                combination.append(newCandidates[i])
                dfs(newCandidates, target - newCandidates[i], i, combination)
                combination.pop() 
        
        dfs(newCandidates, target, 0, [])
        return combinations

# 153 · Combination Sum II
# Description
# Given an array num and a number target. Find all unique combinations in num where the numbers sum to target.
class Solution:
    """
    @param num: Given the candidate numbers
    @param target: Given the target number
    @return: All the combinations that sum to target
             we will sort your return value in output
    """
    def combinationSum2(self, num, target):
        # write your code here
        nums = sorted(num)
        subset,result = [],[]
        start_index = 0
        self.dfs(nums, target, start_index,subset,result)
        return result
        
    def dfs(self, nums, target, start_index,subset,result):
        # combination sum 例题的出口
        if target ==0:
            result.append(subset[:])
            return 
        
        for i in range(start_index, len(nums)):

            if nums[i]>target:
                return 
            # print("subset",subset)
            # subset with duplicate 的判定
            if nums[i] == nums[i-1] and i > start_index:
                continue 
            subset.append(nums[i])

            self.dfs(nums, target-nums[i], i+1,subset,result)
            subset.pop()

# 33 · N-Queens
# Description
# The N-queens puzzle is the problem of placing n queens on an n×nn×n chessboard, and the queens can not attack each other(Any two queens can't be in the same row, column, diagonal line).Given an integer n, return all distinct solutions to the N-queens puzzle.Each solution contains a distinct board configuration of the N-queens' placement, where 'Q' and '.' each indicate a queen and an empty space respectively.

# Contact me on wechat to get Amazon、Google requent Interview questions . (wechat id : jiuzhang0607)


# n <= 10n<=10

# Example
# Example 1:

# Input:

# n = 1
# Output:

# "Q"
# Explanation:

# There is only one solution.

# Example 2:

# Input:

# n = 4
# Output:

# [
#   // Solution 1
#   [".Q..",
#    "...Q",
#    "Q...",
#    "..Q."
#   ],
#   // Solution 2
#   ["..Q.",
#    "Q...",
#    "...Q",
#    ".Q.."
#   ]
# ]
# Explanation:

# There are two solutions.
class Solution:
    """
    @param n: The number of queens
    @return: All distinct solutions
             we will sort your return value in output
    """
    def solveNQueens(self, n):
        #result用于存储答案
        results = []
        self.search(n, [], results)
        return results
    #search函数为搜索函数，n表示已经放置了n个皇后，col表示每个皇后所在的列
    def search(self, n, col, results):
        row = len(col)
        #若已经放置了n个皇后表示出现了一种解法，绘制后加入答案result
        if row == n:
            print(row, n, col)
            results.append(self.Draw(col))
            return
        #枚举当前皇后放置的列，若不合法则跳过
        for now_col in range(n):
            if not self.isValid(col, row, now_col):
                continue
            #若合法则递归枚举下一行的皇后
            col.append(now_col)
            self.search(n, col, results)
            col.pop()
    #isValid函数为合法性判断函数
    def isValid(self, cols, row, now_col):
        for r, c in enumerate(cols):
            #若有其他皇后在同一列或同一斜线上则不合法
            if c == now_col:
                return False
            if abs( r - row ) == abs( c - now_col ):
                return False
        return True
    #Draw函数为将col数组转换为答案的绘制函数
    def Draw(self, cols):
        n = len(cols)
        board = []
        for i in range(n):
            row = ['Q' if j == cols[i] else '.' for j in range(n)]
            board.append(''.join(row))
        return board

# 17 · Subsets
# Description
# Given a set with distinct integers, return all possible subsets.

# Contact me on wechat to get Amazon、Google requent Interview questions . (wechat id : jiuzhang0607)


# Elements in a subset must be in non-descending order.
# The solution set must not contain duplicate subsets.
# Example
# Example 1:

# Input:

# nums = [0] 
# Output:

# [ 
#   [], 
#   [0] 
# ] 



class Solution:
    """
    @param nums: A set of numbers
    @return: A list of lists
             we will sort your return value in output
    """
    def subsets(self, nums: List[int]) -> List[List[int]]:
        # write your code here
        

        def dfs(nums, idx, path, res):
 
            res.append(path[:])

            for i in range(idx, len(nums)):
                path.append(nums[i])
                dfs(nums, i + 1, path, res)
                path.pop()
                
        nums = sorted(nums)
        res = []
        dfs(nums, 0, [], res)
        return res      