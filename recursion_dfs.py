# DFS

# 480 · Binary Tree Paths
# Description
# Given a binary tree, return all root-to-leaf paths.

# Example 1:
# Input：{1,2,3,#,5}
# Output：["1->2->5","1->3"]
# Explanation：
#    1
#  /   \
# 2     3
#  \
#   5

# Example 2:
# Input：{1,2}
# Output：["1->2"]
# Explanation：
#    1
#  /   
# 2     

"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: the root of the binary tree
    @return: all root-to-leaf paths
    """
    def binaryTreePaths(self, root):
        # write your code here
        if not root:
            return []
        
        paths = []
        self.find_paths(root, [root], paths)
        return paths
    
    def find_paths(self, node, path, paths):
        if not node:
            return
        
        if not node.left and not node.right:
            paths.append('->'.join([str(n.val) for n in path]))
            return

        path.append(node.left)
        self.find_paths(node.left, path, paths)
        path.pop()
        
        path.append(node.right)
        self.find_paths(node.right, path, paths)
        path.pop()

# 376 · Binary Tree Path Sum
# Description
# Given a binary tree, find all paths that sum of the nodes in the path equals to a given number target.
# A valid path is from root node to any of the leaf nodes.

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

"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""


class Solution:
    """
    @param: root: the root of binary tree
    @param: target: An integer
    @return: all valid paths
    """
    def binaryTreePathSum(self, root, target):
        # write your code here
        if not root:
            return []
        result = []
        self.dfs(root, target, [], result)
        return result
    
    def dfs(self, root, target, path, result):
        if root is None:
            return 
        path.append(root.val)
        if not root.left and not root.right:
            if root.val == target:
                result.append(path[:])
            del path[-1]
            return 

        self.dfs(root.left, target - root.val, path, result)
        self.dfs(root.right, target - root.val, path, result)
        del path[-1]

# implicit pop and push node
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""


class Solution:
    """
    @param: root: the root of binary tree
    @param: target: An integer
    @return: all valid paths
    """
    def binaryTreePathSum(self, root, target):
        # write your code here
        if not root:
            return []
        result = []
        self.dfs(root, target, [root.val], result)
        return result
    
    def dfs(self, root, target, path, result):
        if root is None:
            return 
       
        if not root.left and not root.right:
            if sum(path) == target:
                result.append(path[:])
            
            return 
        if root.left:
            self.dfs(root.left, target, path + [root.left.val], result)
        if root.right:
            self.dfs(root.right, target, path + [root.right.val], result)


# 453 · Flatten Binary Tree to Linked List
# Description
# Flatten a binary tree to a fake "linked list" in pre-order traversal.
# Here we use the right pointer in TreeNode as the next pointer in ListNode.
# Don't forget to mark the left child of each node to null. Or you will get Time Limit Exceeded or Memory Limit Exceeded.

# Example 1:

# Input:{1,2,5,3,4,#,6}
# Output：{1,#,2,#,3,#,4,#,5,#,6}
# Explanation：
#      1
#     / \
#    2   5
#   / \   \
#  3   4   6

# 1
# \
#  2
#   \
#    3
#     \
#      4
#       \
#        5
#         \
#          6

"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: a TreeNode, the root of the binary tree
    @return: nothing
    """
    def flatten(self, root):
        # write your code here
        # if not root:
        #     return 
    # divide and conquer
        # self.flatten(root.left)
        # self.flatten(root.right)
        # # Connect
        # if root.left:
        #     tmp = root.right 
        #     root.right = root.left
        #     root.left = None 
        #     cur = root.right
        #     while cur.right:
        #         cur = cur.right
        #     cur.right = tmp

    # stack

        if not root:
            return
        
        stack = collections.deque([root])
        
        while stack:
            node = stack.pop()
            
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
            
            node.left = None
            
            if stack:
                node.right = stack[-1]
            else:
                node.right = None
            
        
# 468 · Symmetric Binary Tree
# Description
# Given a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).
# Example 1:

# Input: {1,2,2,3,4,4,3}
# Output: true
# Explanation:
#          1
#         / \
#        2   2
#       / \ / \
#       3 4 4 3

# is a symmetric binary tree.

"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: the root of binary tree.
    @return: true if it is a mirror of itself, or false.
    """
    def isSymmetric(self, root):
        # write your code here
        if not root:
            return True
        return self.helper(root.left, root.right)
    
    
    def helper(self, left, right):
        if not left and not right:
            return True
        elif not (left and right):
            return False
        return left.val == right.val and \
        self.helper(left.left, right.right) and \
        self.helper(left.right, right.left)
        
# 469 · Same Tree
# Description
# Check if two binary trees are identical. Identical means the two binary trees have the same structure and every identical 
# position has the same value.

# Example 1:
# Input:{1,2,2,4},{1,2,2,4}
# Output:true
# Explanation:
#         1                   1
#        / \                 / \
#       2   2   and         2   2
#      /                   /
#     4                   4

# are identical.
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param a: the root of binary tree a.
    @param b: the root of binary tree b.
    @return: true if they are identical, or false.
    """
    def isIdentical(self, a, b):
        # write your code here
        return self.helper(a, b)
    
    def helper(self, root_a, root_b):
        if not root_a and not root_b:
            return True 
        elif not (root_a and root_b):
            return False 
        return root_a.val == root_b.val and \
        self.helper(root_a.left, root_b.left) and \
        self.helper(root_a.right, root_b.right)

# 470 · Tweaked Identical Binary Tree
# Description
# Check whether two binary trees are equivalent after several twists. Twist is defined as exchanging left and right subtrees 
# of any node. The definition of equivalence is that two binary trees must have the same structure, and the values of nodes 
# in corresponding positions must be equal.

# There is no two nodes with the same value in the tree.

# Example 1:

# Input:{1,2,3,4},{1,3,2,#,#,#,4}
# Output:true
# Explanation:
#         1             1
#        / \           / \
#       2   3   and   3   2
#      /                   \
#     4                     4

# are identical.

"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param a: the root of binary tree a.
    @param b: the root of binary tree b.
    @return: true if they are tweaked identical, or false.
    """
    def isTweakedIdentical(self, a, b):
        # write your code here
        if not a and not b:
            return True 

        if a and b and a.val == b.val:
            return self.isTweakedIdentical(a.left, b.left) and \
                    self.isTweakedIdentical(a.right, b.right) or \
                    self.isTweakedIdentical(a.left, b.right) and \
                    self.isTweakedIdentical(a.right, b.left)
        else:
            return False

# 481 · Binary Tree Leaf Sum
# Description
# Given a binary tree, calculate the sum of leaves.

# Example 1:
# Input：{1,2,3,4}
# Output：7
# Explanation：
#     1
#    / \
#   2   3
#  /
# 4
# 3+4=7

"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""
# dfs
class Solution:
    """
    @param root: the root of the binary tree
    @return: An integer
    """
    def leafSum(self, root):
        # write your code here
        if not root:
           return 0 
        total = []
        
        self.helper(root, total)
        return sum(total) 
    
    def helper(self, root, total):
        if not root:
            return 
        if not root.left and not root.right:
            total.append(root.val)
            return 
        self.helper(root.left, total)
        self.helper(root.right, total)

# divide and conquer
    def leafSum(self, root):
        # write your code here
        if not root:
           return 0 
        
        if not root.left and not root.right:
            return root.val 
        
        return self.leafSum(root.left) + self.leafSum(root.right)

# 482 · Binary Tree Level Sum
# Description
# Given a binary tree and an integer which is the depth of the target level.

# Calculate the sum of the nodes in the target level.

# Example 1:

# Input：{1,2,3,4,5,6,7,#,#,8,#,#,#,#,9},2
# Output：5 
# Explanation：
#      1
#    /   \
#   2     3
#  / \   / \
# 4   5 6   7
#    /       \
#   8         9
# 2+3=5

"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: the root of the binary tree
    @param level: the depth of the target level
    @return: An integer
    """
    def levelSum(self, root, level):
        # write your code here
        if not root:
            return 0
        count = 1
        total = []
        self.helper(root, level, count, total)
        return sum(total)

    def helper(self, root, level, count, total):
        if not root:
            return 
        if count == level:
            total.append(root.val)
        self.helper(root.left, level, count + 1, total)
        self.helper(root.right, level, count + 1, total)
        
# 595 · Binary Tree Longest Consecutive Sequence
# Description
# Given a binary tree, find the length of the longest consecutive sequence path.

# The path refers to any sequence of nodes from some starting node to any node in the tree along 
# the parent-child connections. The longest consecutive path need to be from parent to child (cannot be the reverse).

# Example 1:

# Input:
#    1
#     \
#      3
#     / \
#    2   4
#         \
#          5
# Output:3
# Explanation:
# Longest consecutive sequence path is 3-4-5, so return 3.
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
    @return: the length of the longest consecutive sequence path
    """
    def longestConsecutive(self, root):
        # write your code here
        if not root:
            return 0 
       
        return self.helper(root, None, 0)
        
    
    def helper(self, root, parent, count):
        if not root:
            return count
        if parent != None and root.val == parent.val + 1:
            count += 1
        else:
            count = 1
        return max(count, self.helper(root.left, root, count),\
        self.helper(root.right, root, count))


# 597 · Subtree with Maximum Average
# Description
# Given a binary tree, find the subtree with maximum average. Return the root of the subtree.

# LintCode will print the subtree which root is your return node.
# It's guaranteed that there is only one subtree with maximum average.

# Example 1
# Input：
# {1,-5,11,1,2,4,-2}
# Output：11
# Explanation:
# The tree is look like this:
#      1
#    /   \
#  -5     11
#  / \   /  \
# 1   2 4    -2 
# The average of subtree of 11 is 4.3333, is the maximun.

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
    @return: the root of the maximum average of subtree
    """
    def findSubtree2(self, root):
        # write your code here
        maximum, sub_node, _, _ = self.helper(root)
        return sub_node
    
    def helper(self, root):
        if not root:
            return -sys.maxsize, None, 0, 0
        

        left_max, left_node, left_size, left_sum = self.helper(root.left)
        right_max, right_node, right_size, right_sum  = self.helper(root.right)
        
        sub_size, sub_sum = left_size + right_size + 1, left_sum + right_sum + root.val  
        sub_avg = sub_sum / sub_size

        if left_max == max(left_max, right_max, sub_avg):
            return left_max, left_node, sub_size, sub_sum
        if right_max == max(left_max, right_max, sub_avg):
            return right_max, right_node, sub_size, sub_sum 
        
        return sub_avg, root, sub_size, sub_sum

# 596 · Minimum Subtree
# Description
# Given a binary tree, find the subtree with minimum sum. Return the root of the subtree.
# The range of input and output data is in int.

# LintCode will print the subtree which root is your return node.
# It's guaranteed that there is only one subtree with minimum sum and the given binary tree is not an empty tree.

# Example
# Example 1:

# Input:
# {1,-5,2,1,2,-4,-5}
# Output:1
# Explanation:
# The tree is look like this:
#      1
#    /   \
#  -5     2
#  / \   /  \
# 1   2 -4  -5 
# The sum of whole tree is minimum, so return the root.

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
    @return: the root of the minimum subtree
    """
    def findSubtree(self, root):
        # write your code here
        minimum, sub_tree, root_sum = self.helper(root)
        return sub_tree
    
    def helper(self, root):
        if not root:
            return sys.maxsize, None, 0
        
        left_min, left_node, left_sum = self.helper(root.left)
        right_min, right_node, right_sum = self.helper(root.right)

        root_sum = left_sum + right_sum + root.val
        
        if left_min == min(left_min, right_min, root_sum):
            return left_min, left_node, root_sum
        if right_min == min(left_min, right_min, root_sum):
            return right_min, right_node, root_sum
        
        return root_sum, root, root_sum