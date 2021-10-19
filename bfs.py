# Breath First Search

# BFS sample formule:
# step 1: puting head into deque, label distance as initial range as 0 in dict
def sampleBFS(self, graph):
    queue = colections.deque([node])
    distance = {node: 0}
# step 2: looping and poping
    while queue:
        node = queue.popleft()
# step 3: visting next node, and adding node, and store distance
        for neighbor in node.get_neighbors():
            if neighbor in distance:
                continue
            distance[neighbor] = distance[node] + 1 
            queue.append(neighbor)

# 433 · Number of Islands
# Description
# Given a boolean 2D matrix, 0 is represented as the sea, 1 is represented as the island. If two 1 is adjacent, 
# we consider them in the same island. We only consider up/down/left/right adjacent.
# Find the number of islands.

# Example 1:
# Input:
# [
#   [1,1,0,0,0],
#   [0,1,0,0,1],
#   [0,0,0,1,1],
#   [0,0,0,0,0],
#   [0,0,0,0,1]
# ]
# Output:
# 3

# Example 2:
# Input:
# [
#   [1,1]
# ]
# Output:
# 1

def numIslands(self, grid):
    # write your code here
    if not grid or not grid[0]:
        return 0
    islands = 0
    visited = set()
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] and (i, j) not in visited:
                self.bfs(grid, i, j, visited)
                islands += 1

    return islands 

def bfs(self, grid, x, y, visited):
    queue = collections.deque([(x, y)])
    visited.add((x, y))
    while queue:
        curX, curY = queue.popleft()
        for deltaX, deltaY in DIRECTIONS:
            newX, newY = curX + deltaX, curY + deltaY
            if not self.is_valid(grid,newX, newY, visited):
                continue
            queue.append((newX,newY))
            visited.add((newX,newY))

def is_valid(self, grid, x, y, visited):
    n, m = len(grid), len(grid[0])
    if not (0 <= x <n and 0 <= y < m):
        return False
    if (x,y) in visited:
        return False
    return grid[x][y]





# 137 · Clone Graph
# Description
# Clone an undirected graph. Each node in the graph contains a label and a list of its neighbors. Nodes are labeled uniquely.
# You need to return a deep copied graph, which has the same structure as the original graph, and any changes to the new graph 
# will not have any effect on the original graph.

# You need return the node with the same label as the input node.
# How we represent an undirected graph: http://www.lintcode.com/help/graph/

# Example1
# Input:
# {1,2,4#2,1,4#4,1,2}
# Output: 
# {1,2,4#2,1,4#4,1,2}
# Explanation:
# 1------2  
#  \     |  
#   \    |  
#    \   |  
#     \  |  
#       4   

def cloneGraph(self, node):
    # write your code here
    if not node:
        return None
    nodes = self.copy_nodes(node)
    mapping = self.mapping_labels(nodes)
    self.copy_neighbors(nodes, mapping)
    return mapping[node]

def copy_nodes(self, node):
    queue = collections.deque([node])
    visited = set([node])
    while queue:
        curNode = queue.popleft()
        for neighbor in curNode.neighbors:
            if neighbor in visited:
                continue 
            queue.append(neighbor)
            visited.add(neighbor)
    return list(visited)

def mapping_labels(self, nodes):
    mapping = {}
    for node in nodes:
        mapping[node] = UndirectedGraphNode(node.label)
    return mapping

def copy_neighbors(self, nodes, mapping): 
    for node in nodes:
        curNode = mapping[node]
        for neighbor in node.neighbors:
            newNeighbor = mapping[neighbor]
            curNode.neighbors.append(newNeighbor)


# 611 · Knight Shortest Path
# Description
# Given a knight in a chessboard (a binary matrix with 0 as empty and 1 as barrier) with a source position, find the shortest 
# path to a destination position, return the length of the route.
# Return -1 if destination cannot be reached.
# source and destination must be empty.
# Knight can not enter the barrier.
# Path length refers to the number of steps the knight takes.
# If the knight is at (x, y), he can get to the following positions in one step:

# (x + 1, y + 2)
# (x + 1, y - 2)
# (x - 1, y + 2)
# (x - 1, y - 2)
# (x + 2, y + 1)
# (x + 2, y - 1)
# (x - 2, y + 1)
# (x - 2, y - 1)

# Example 1:
# Input:
# [[0,0,0],
#  [0,0,0],
#  [0,0,0]]
# source = [2, 0] destination = [2, 2] 
# Output: 2
# Explanation:
# [2,0]->[0,1]->[2,2]

# Example 2:
# Input:
# [[0,1,0],
#  [0,0,1],
#  [0,0,0]]
# source = [2, 0] destination = [2, 2] 
# Output:-1

"""
Definition for a point.
class Point:
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b
"""
DIRECTIONS = [(1, 2),
(1, - 2),
(- 1, 2),
(- 1, -2),
(2, 1),
(2, - 1),
(- 2, 1),
(- 2, - 1)]

class Solution:
    """
    @param grid: a chessboard included 0 (false) and 1 (true)
    @param source: a point
    @param destination: a point
    @return: the shortest path 
    """
    def shortestPath(self, grid, source, destination):
        # write your code here

        queue = collections.deque([(source.x, source.y)])
        visited = {(source.x, source.y): 0}
        
        while queue:
            curX, curY = queue.popleft()
            if (curX, curY) == (destination.x, destination.y):
                return visited[(curX,curY)]
            for deltaX, deltaY in DIRECTIONS:
                newX, newY = curX + deltaX, curY + deltaY
                
                if not self.is_valid(grid, newX, newY, visited):
                    continue 
                queue.append((newX, newY))
                visited[(newX, newY)] = visited[(curX, curY)] + 1
        return -1 
        
    def is_valid(self, grid, x, y, visited):
        n, m = len(grid), len(grid[0])
        if not (0 <= x < n and 0 <= y < m):
            return False 
        if (x, y) in visited:
            return False
        return grid[x][y] != 1

# 127 · Topological Sorting
# Description
# Given an directed graph, a topological order of the graph nodes is defined as follow:
# For each directed edge A -> B in graph, A must before B in the order list.
# The first node in the order can be any node in the graph with no nodes direct to it.
# Find any topological order for the given graph.
# You can assume that there is at least one topological order in the graph.
# Learn more about representation of graphs
# The number of graph nodes <= 5000

# Example 1:
# Input:
# graph = {0,1,2,3#1,4#2,4,5#3,4,5#4#5}
# Output:
# [0, 1, 2, 3, 4, 5]
"""
class DirectedGraphNode:
     def __init__(self, x):
         self.label = x
         self.neighbors = []
"""

class Solution:
    """
    @param graph: A list of Directed graph node
    @return: Any topological order for the given graph.
    """
    def topSort(self, graph):
        # write your code here
        node_to_indegree = self.node_degree_hash(graph)
        queue = collections.deque([headNode for headNode in node_to_indegree if node_to_indegree[headNode] == 0])
        order = []
        while queue:
            curNode = queue.popleft()
            order.append(curNode)
            for neighbor in curNode.neighbors:
                node_to_indegree[neighbor] -= 1
                if node_to_indegree[neighbor] == 0:
                    queue.append(neighbor)
        
        return order

    def node_degree_hash(self, graph):
        node_to_indegree = {node: 0 for node in graph}
        for node in node_to_indegree:
            for neighbor in node.neighbors:
                node_to_indegree[neighbor] += 1

        return node_to_indegree

# 120 · Word Ladder
# Description
# Given two words (start and end), and a dictionary, find the shortest transformation sequence from start to end, 
# output the length of the sequence.
# Transformation rule such that:
# Only one letter can be changed at a time
# Each intermediate word must exist in the dictionary. (Start and end words do not need to appear in the dictionary )
# Return 0 if there is no such transformation sequence.
# All words have the same length.
# All words contain only lowercase alphabetic characters.
# You may assume no duplicates in the dictionary.
# You may assume beginWord and endWord are non-empty and are not the same.
# len(dict) <= 5000, len(start) < 5len(dict)<=5000,len(start)<5

# Example 1:
# Input:
# start = "a"
# end = "c"
# dict =["a","b","c"]
# Output:
# 2
# Explanation:
# "a"->"c"

# Example 2:
# Input:
# start ="hit"
# end = "cog"
# dict =["hot","dot","dog","lot","log"]
# Output:
# 5
# Explanation:
# "hit"->"hot"->"dot"->"dog"->"cog"

class Solution:
    """
    @param: start: a string
    @param: end: a string
    @param: dict: a set of string
    @return: An integer
    """
    def ladderLength(self, start, end, dict):
        # write your code here
        dict.add(end)
        queue, visited = collections.deque([start]), {start: 1}

        while queue:
            cur_word = queue.popleft()
            if cur_word == end:
                return visited[cur_word]
            for word in self.get_words(cur_word,dict):
                if word in visited:
                    continue
                queue.append(word)
                visited[word] = visited[cur_word] + 1
        return 0 

    def get_words(self, word, dict):
        words = []
        for i in range(len(word)):
            left, right = word[:i], word[i + 1:]
            for char in "abcdefghijklmnopqrstuvwxyz":
                if char == word[i]:
                    continue

                new_word = left + char + right 
                if new_word in dict:
                     words.append(new_word)

        return words


# 615 · Course Schedule
# Description
# There are a total of n courses you have to take, labeled from 0 to n - 1.
# Before taking some courses, you need to take other courses. For example, to learn course 0, you need to learn course 1 first, which is expressed as [0,1].
# Given the total number of courses and a list of prerequisite pairs, is it possible for you to finish all courses?

# Example 1:
# Input: n = 2, prerequisites = [[1,0]] 
# Output: true

# Example 2:
# Input: n = 2, prerequisites = [[1,0],[0,1]] 
# Output: false

class Solution:
    """
    @param numCourses: a total of n courses
    @param prerequisites: a list of prerequisite pairs
    @return: true if can finish all courses or false
    """
    def canFinish(self, numCourses, prerequisites):
        # write your code here
        edges = {i:[] for i in range(numCourses)}
        degree = [0 for _ in range(numCourses)]
        for after, pre in prerequisites:
            edges[pre].append(after)
            degree[after] += 1
        
        queue = collections.deque([i for i in range(numCourses) if degree[i] == 0])
        count = 0 
        while queue:
            node = queue.popleft()
            count += 1
            for course in edges[node]:
                degree[course] -= 1
                if degree[course] == 0:
                    queue.append(course)
        
        return count == numCourses

# 616 · Course Schedule II
# Description
# There are a total of n courses you have to take, labeled from 0 to n - 1.
# Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]
# Given the total number of courses and a list of prerequisite pairs, return the ordering of courses you should take to finish all courses.
# There may be multiple correct orders, you just need to return one of them. If it is impossible to finish all courses, return an empty array.

# Example 1:
# Input: n = 2, prerequisites = [[1,0]] 
# Output: [0,1]

# Example 2:
# Input: n = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]] 
# Output: [0,1,2,3] or [0,2,1,3]

class Solution:
    """
    @param: numCourses: a total of n courses
    @param: prerequisites: a list of prerequisite pairs
    @return: the course order
    """
    def findOrder(self, numCourses, prerequisites):
        # write your code here
        edges = {i:[] for i in range(numCourses)}
        degree = [0 for _ in range(numCourses)]
        for course, prep in prerequisites:
            edges[prep].append(course)
            degree[course] += 1
        
        queue = collections.deque([i for i in range(numCourses) if degree[i] == 0])
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for course in edges[node]:
                degree[course] -= 1
                if degree[course] == 0:
                    queue.append(course)
                   
        if len(order) == numCourses:
            return order
        return []

# 630 · Knight Shortest Path II
# Description
# Given a knight in a chessboard n * m (a binary matrix with 0 as empty and 1 as barrier). 
# the knight initialze position is (0, 0) and he wants to reach position (n - 1, m - 1), 
# Knight can only be from left to right. Find the shortest path to the destination position, 
# return the length of the route. Return -1 if knight can not reached.
# If the knight is at (x, y), he can get to the following positions in one step:
# (x + 1, y + 2)
# (x - 1, y + 2)
# (x + 2, y + 1)
# (x - 2, y + 1)

# Example 1:
# Input:
# [[0,0,0,0],[0,0,0,0],[0,0,0,0]]
# Output:
# 3
# Explanation:
# [0,0]->[2,1]->[0,2]->[2,3]

# Example 2:
# Input:
# [[0,1,0],[0,0,1],[0,0,0]]
# Output:
# -1
DIRECTIONS = [(1, 2),
(- 1, 2),
(2, 1),
(-2, 1)]

class Solution:
    """
    @param grid: a chessboard included 0 and 1
    @return: the shortest path
    """
    def shortestPath2(self, grid):
        # write your code here
        queue = collections.deque([(0,0)])
        steps = {(0, 0): 0}
        n, m = len(grid), len(grid[0])
        while queue:
            cur_x, cur_y = queue.popleft()
            if (cur_x, cur_y) == (n - 1, m - 1):
                return steps[(cur_x, cur_y)]
            for d_x, d_y in DIRECTIONS:
                new_x, new_y = cur_x + d_x, cur_y + d_y

                if not self.is_valid(grid, new_x, new_y, n, m):
                    continue
                if (new_x, new_y) in steps:
                    continue 

                queue.append((new_x, new_y))
                steps[(new_x, new_y)] = steps[(cur_x, cur_y)] + 1

        return -1
                    
    def is_valid(self, grid, x, y, n, m):
        if not (0 <= x < n and 0 <= y < m):
            return False 
        if grid[x][y] == 1:
            return False 
        return not grid[x][y]
