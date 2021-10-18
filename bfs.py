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
