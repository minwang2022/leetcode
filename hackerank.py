
#You have just arrived in a new city and would like to see its sights. Each sight is located in a square and you have assigned each a beauty value. Each road to a square takes an amount of time to travel, and you have limited time for sightseeing. Detemine the maximum value of beauty that you can visit during your time in the city. Start and finish at your hotel, the location of sight zero.
# Constraints

# 1 <= n <= 1000
# 1 <= m <= 2000
# 10 <= max_t <= 100
# 0 <= u[i], v[i] <= n-1
# u[i] != v[i]
# 10 <= t[i] <= 100
# 0 <= beauty[i] <= 10^8
# No more than 4 roads connect a single square with others
# Two roads can be connected by at most 1 road
# Example

# n = 4
# m = 3
# max_t = 30
# beauty = [5, 10, 15, 20]
# u = [0, 1, 0]
# v = [1, 2, 3]
# t = [6, 7, 10]

# Output
# 43

# Explanation : 0 -> 3 -> 0

from collections import defaultdict
def findBestPath(n, m, max_t, beauty, u, v, t):
    def prepare_graph():
        graph = defaultdict(list)
        for i in range(len(u)):
            graph[u[i]].append([v[i], t[i]])
            graph[v[i]].append([u[i], t[i]])
        return graph

    def dfs_helper(node, curr_val, curr_time, visited):
        if curr_time > max_t:
            return

        if node == 0:
            max_beaty[0] = max(max_beaty[0], curr_val)

        for nei in graph[node]:
            new_node, new_node_time = nei[0], nei[1]

            new_node_val = beauty[new_node]

            if new_node in visited:
                new_node_val = 0

            dfs_helper(new_node, curr_val + new_node_val, curr_time + new_node_time, visited | set([new_node]))

    max_beaty = [float('-inf')]
    graph = prepare_graph()

    dfs_helper(0, beauty[0], 0, set([0]))

    return max_beaty[0]




#degree of an array 
#degree of array, find shortest length
# 5    →   arr[] size n = 5
# 1    →   arr = [1, 2, 2, 3, 1]

# 2 options for items with highest degree
# [1, 2, 2, 3, 1], [2, 2]

#output -> 2 [2, 2]

import collections
def degreeOfArray(arr):
    # Write your code here
    degrees = collections.Counter(arr)
    max_degree = degrees.most_common(1)[-1][-1]
    nums = []
    for num, degree in degrees.items():
        if degree == max_degree:
            nums.append(num)
      
    min_len = float("inf")
    for item in nums:
        min_len = min(min_len, find_length(item, arr))
    
    return min_len

def find_length(num, arr):
    left = arr.index(num)
    right = len(arr) - arr[::-1].index(num)


#array reduction

# STDIN    Function
# -----    --------
# 3    →   num[] size n = 3
# 1    →   num = [1, 2, 3]
# outpu 9

# STDIN    Function
# -----    --------
# 4    →   num[] size n = 4
# 1    →   num = [1, 2, 3, 4]
# outpu 19
import heapq
def reductionCost(num):
    # Write your code here
    heap = num
    heapq.heapify(heap)
    
    res = 0 
    while heap:
        cur = 0 
        for _ in range(2):
            if heap:
                cur += heapq.heappop(heap)
        if heap:       
            heapq.heappush(heap, cur)
        res += cur
    return res