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

# 618 · Search Graph Nodes
# Description
# Given a undirected graph, a node and a target, return the nearest node to given node which value of it is target,
# return NULL if you can't find.
# There is a mapping store the nodes' values in the given parameters.
# It's guaranteed there is only one available solution

# Example 1:
# Input:
# {1,2,3,4#2,1,3#3,1,2#4,1,5#5,4}
# [3,4,5,50,50]
# 1
# 50
# Output:
# 4
# Explanation:
# 2------3  5
#  \     |  | 
#   \    |  |
#    \   |  |
#     \  |  |
#       1 --4
# Give a node 1, target is 50

# there a hash named values which is [3,4,10,50,50], represent:
# Value of node 1 is 3
# Value of node 2 is 4
# Value of node 3 is 10
# Value of node 4 is 50
# Value of node 5 is 50

# Return node 4

# Example 2:
# Input:
# {1,2#2,1}
# [0,1]
# 1
# 1
# Output:
# 2

"""
Definition for a undirected graph node
class UndirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []
"""


class Solution:
    """
    @param: graph: a list of Undirected graph node
    @param: values: a hash mapping, <UndirectedGraphNode, (int)value>
    @param: node: an Undirected graph node
    @param: target: An integer
    @return: a node
    """
    def searchNode(self, graph, values, node, target):
        # write your code here
        queue = collections.deque([node])
        visited = set([node])
        while queue:
            cur_node = queue.popleft()
            if values[cur_node] == target:
                return cur_node
            for neighbor in cur_node.neighbors:
                if neighbor in visited:
                    continue 
                queue.append(neighbor)
                visited.add(neighbor)
        return None

# 605 · Sequence Reconstruction
# Description
# Check whether the original sequence org can be uniquely reconstructed from the sequences in seqs. 
# The org sequence is a permutation of the integers from 1 to n, with 1 \leq n \leq 10^4 1≤n≤10^4. 
# Reconstruction means building a shortest common supersequence of the sequences in seqs (i.e., a 
# shortest sequence so that all sequences in seqs are subsequences of it). Determine whether there is 
# only one sequence that can be reconstructed from seqs and it is the org sequence.


# Example 1:
# Input:org = [1,2,3], seqs = [[1,2],[1,3]]
# Output: false
# Explanation:
# [1,2,3] is not the only one sequence that can be reconstructed, because [1,3,2] is also a valid sequence that can be reconstructed.

# Example 2:
# Input: org = [1,2,3], seqs = [[1,2]]
# Output: false
# Explanation:
# The reconstructed sequence can only be [1,2].

# Example 3:
# Input: org = [1,2,3], seqs = [[1,2],[1,3],[2,3]]
# Output: true
# Explanation:
# The sequences [1,2], [1,3], and [2,3] can uniquely reconstruct the original sequence [1,2,3].

# Example 4:
# Input:org = [4,1,5,2,6,3], seqs = [[5,2,6,3],[4,1,5,2]]
# Output:true

class Solution:
    """
    @param org: a permutation of the integers from 1 to n
    @param seqs: a list of sequences
    @return: true if it can be reconstructed only one or false
    """
    def sequenceReconstruction(self, org, seqs):
        graph = self.build_graph(seqs)
        topo_order = self.topological_sort(graph)
        return topo_order == org
             
    def build_graph(self, seqs):
        # initialize graph
        graph = {}
        for seq in seqs:
            for node in seq:
                if node not in graph:
                    graph[node] = set()
        
        for seq in seqs:
            for i in range(1, len(seq)):
                graph[seq[i - 1]].add(seq[i])

        return graph
    
    def get_indegrees(self, graph):
        indegrees = {
            node: 0
            for node in graph
        }
        
        for node in graph:
            for neighbor in graph[node]:
                indegrees[neighbor] += 1
                
        return indegrees
        
    def topological_sort(self, graph):
        indegrees = self.get_indegrees(graph)
        
        queue = []
        for node in graph:
            if indegrees[node] == 0:
                queue.append(node)
        
        topo_order = []
        while queue:
            if len(queue) > 1:
                # there must exist more than one topo orders
                return None
                
            node = queue.pop()
            topo_order.append(node)
            for neighbor in graph[node]:
                indegrees[neighbor] -= 1
                if indegrees[neighbor] == 0:
                    queue.append(neighbor)
                    
        if len(topo_order) == len(graph):
            return topo_order
            
        return None

# 598 · Zombie in Matrix
# Description
# Give a two-dimensional grid, each grid has a value, 2 for wall, 1 for zombie, 0 for human (numbers 0, 1, 2).
# Zombies can turn the nearest people(up/down/left/right) into zombies every day, but can not through wall. 
# How long will it take to turn all people into zombies? Return -1 if can not turn all people into zombies.

# Example 1:
# Input:
# [[0,1,2,0,0],
#  [1,0,0,2,1],
#  [0,1,0,0,0]]
# Output:
# 2

# Example 2:
# Input:
# [[0,0,0],
#  [0,0,0],
#  [0,0,1]]
# Output:
# 4

DIRECTIONS = [(1, 0),(-1, 0),(0, -1), (0, 1)]
class Solution:
    """
    @param grid: a 2D integer grid
    @return: an integer
    """
    def zombie(self, grid):
        # write your code here
        if not grid:
            return -1 

        n, m = len(grid), len(grid[0])
        zoombies_pos = []
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    zoombies_pos.append((i,j))

        queue = collections.deque(zoombies_pos)
        visited = set()
        time = 0
        while queue:
            time += 1 
            for i in range(len(queue)):
                cur_x, cur_y = queue.popleft()
                for d_x, d_y in DIRECTIONS:
                    new_x, new_y = cur_x + d_x, cur_y + d_y
                    if not self.is_valid(grid, new_x, new_y, n, m):
                        continue 
                    if (new_x, new_y) in visited:
                        continue
                    queue.append((new_x, new_y))
                    visited.add((new_x, new_y))
            

        humans = sum(x.count(0) for x in grid)
        if humans == len(visited):
            return time - 1
        return -1

    def is_valid(self, grid, x, y, n, m):
        if not (0 <= x < n and 0 <= y < m):
            return False

        return grid[x][y] == 0

# 178 · Graph Valid Tree
# Description
# Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge is a pair of nodes), write a function to check whether these edges make up a valid tree.
# You can assume that no duplicate edges will appear in edges. Since all edges are undirected, [0, 1] is the same as [1, 0] and thus will not appear together in edges.

# Example 1:
# Input: n = 5 edges = [[0, 1], [0, 2], [0, 3], [1, 4]]
# Output: true

# Example 2:
# Input: n = 5 edges = [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]]
# Output: false.

class Solution:
    """
    @param n: An integer
    @param edges: a list of undirected edges
    @return: true if it's a valid tree, or false
    """
    def validTree(self, n, edges):
        # write your code here
        numOfEdge = len(edges)
        if numOfEdge != n - 1:
            return False
        adjacent = [[0] * n for _ in range(n)] 
        for i in range(numOfEdge):
            u = edges[i][0]
            v = edges[i][1]
            adjacent[u][v] = adjacent[v][u] = 1
        visit = [0] * n
        visit[0] = 1
        root, numOfVisited = 0, 1
        q = collections.deque()
        q.append(root)
        while len(q) != 0:
            root = q.popleft()
            for i in range(n):
                if adjacent[root][i] != 1:
                    continue
                if visit[i] == 0:
                    visit[i] = 1
                    numOfVisited += 1
                    q.append(i)
        if numOfVisited == n:
            return True
        return False
    
# 892 · Alien Dictionary
# Description
# There is a new alien language which uses the latin alphabet. However, the order among letters are unknown to you. You receive a list of non-empty words from the dictionary, where words are sorted lexicographically by the rules of this new language. Derive the order of letters in this language.

# You may assume all letters are in lowercase.
# The dictionary is invalid, if a is prefix of b and b is appear before a.
# If the order is invalid, return an empty string.
# There may be multiple valid order of letters, return the smallest in normal lexicographical order

# Example 1:
# Input：["wrt","wrf","er","ett","rftt"]
# Output："wertf"
# Explanation：
# from "wrt"and"wrf" ,we can get 't'<'f'
# from "wrt"and"er" ,we can get 'w'<'e'
# from "er"and"ett" ,we can get 'r'<'t'
# from "ett"and"rftt" ,we can get 'e'<'r'
# So return "wertf"

# Example 2:
# Input：["z","x"]
# Output："zx"
# Explanation：
# from "z" and "x"，we can get 'z' < 'x'
# So return "zx"

from heapq import heapify, heappop, heappush

class Solution:
    """
    @param words: a list of words
    @return: a string which is correct order
    """
    def alienOrder(self, words):
        # Write your code here
        graph = self.build_graph(words)
        if not graph:
            return ""
        return self.topological_sort(graph)
    
    def build_graph(self, words):
        graph = {}
        for word in words:
            for c in word:
                if c not in graph:
                    graph[c] = set()

        for i in range(len(words) - 1):
            for j in range(min(len(words[i]), len(words[i + 1]))):
                if words[i][j] != words[i + 1][j]:
                    graph[words[i][j]].add(words[i + 1][j])
                    break
                if j == min(len(words[i]), len(words[i + 1])) - 1:
                    if len(words[i]) > len(words[i + 1]):
                        return None
        print("graph",graph)
        return graph
    
    def get_indegree(self, graph):
        indegree = {node: 0 for node in graph}
        for node in graph:
            for neighbor in graph[node]:
                indegree[neighbor] += 1
        
        return indegree 

    def topological_sort(self, graph):
        indegree = self.get_indegree(graph)
        print("indegree",indegree)
        queue = [node for node in graph if indegree[node] == 0]
        heapify(queue)

        new_word = ""
        while queue:
            node = heappop(queue)
            new_word += node 
            for neighbor in graph[node]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    heappush(queue, neighbor)

        return new_word if len(new_word) == len(graph) else ""

        