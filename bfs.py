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

