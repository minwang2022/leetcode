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
