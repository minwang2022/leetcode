# questions: You are given an integer array arr. You can choose a set of integers and remove all the occurrences of these integers in the array.
# Return the minimum size of the set so that at least half of the integers of the array are removed.
#ex. [1,1,1,1,2,2,2,2,2,3,3,3,3] remove 3, and 1 , return 2 



# In a town, there are n people labeled from 1 to n. There is a rumor that one of these people is secretly the town judge.
# If the town judge exists, then:
# The town judge trusts nobody.
# Everybody (except for the town judge) trusts the town judge.
# There is exactly one person that satisfies properties 1 and 2.
# You are given an array trust where trust[i] = [ai, bi] representing that the person labeled ai trusts the person labeled bi.
# Return the label of the town judge if the town judge exists and can be identified, or return -1 otherwise.
# # N - 1 people trust len n - 1
# n =2 [[4,1]] => 1
# [[1, 3],[2, 5],[3,5],[4,5], [1,5]] => 5
#O(N) TIME, O(N) SPACE

def findJudge(n, array):
	trusted = {} #trusted_people 
	Trusting = {} # not_judge
	for i in range(len(array)):
		cur = array[i][0]
		trust = array[i][1]
		if  trust in trusted:
			Trusted[trust] += 1
		else:
			Trusted[trust] = 1
		if  cur in Trusting:
			Trusting[cur] += 1
		else:
			Trusting[cur] = 1
	judge = max(trusted, key= trusted.get())		#max(trusted, key = lambda x:)
	if judge in trusting and trusted[judge] != n - 1:
		Return -1
	return judge

	
# characters in the most inner layer, parentheses are vaild
# ex: "a((b))" => "b"

def charsInString(string):
    if not "(" in string:
        return string
    front_bracket, cur_layers = 0, 0
    visited = {}
    for char in string:
        if char == "(":
            front_bracket += 1
            cur_layers = front_bracket
            continue 
        if char == ")" and front_bracket != 0:
            cur_layers -= 1
            front_bracket -= 1
            continue
        if cur_layers in visited:
            visited[cur_layers].append(char)
        else:
            visited[cur_layers] = [char]
    
    max_layer = max(visited)
    return visited[max_layer]


# 155. Min Stack
# Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

# Implement the MinStack class:

# MinStack() initializes the stack object.
# void push(int val) pushes the element val onto the stack.
# void pop() removes the element on the top of the stack.
# int top() gets the top element of the stack.
# int getMin() retrieves the minimum element in the stack.
 

# Example 1:

# Input
# ["MinStack","push","push","push","getMin","pop","top","getMin"]
# [[],[-2],[0],[-3],[],[],[],[]]

# Output
# [null,null,null,null,-3,null,0,-2]

# Explanation
# MinStack minStack = new MinStack();
# minStack.push(-2);
# minStack.push(0);
# minStack.push(-3);
# minStack.getMin(); // return -3
# minStack.pop();
# minStack.top();    // return 0
# minStack.getMin(); // return -2

class MinStack:

    def __init__(self):
        self.stack = []

    def push(self, val: int) -> None:
        if not self.stack:
            self.stack.append((val, val))
        else:
            self.stack.append((val, min(val, self.stack[-1][1])))

    def pop(self) -> None:
        if self.stack: 
            self.stack.pop()
       
        

    def top(self) -> int:
        if self.stack:
            return self.stack[-1][0]
        else:
            return None 

    def getMin(self) -> int:
        if self.stack:
            return self.stack[-1][1]
        else:
            return None

# 430. Flatten a Multilevel Doubly Linked List
# You are given a doubly linked list, which contains nodes that have a next pointer, a previous pointer, and an additional child pointer. This child pointer may or may not point to a separate doubly linked list, also containing these special nodes. These child lists may have one or more children of their own, and so on, to produce a multilevel data structure as shown in the example below.

# Given the head of the first level of the list, flatten the list so that all the nodes appear in a single-level, doubly linked list. Let curr be a node with a child list. The nodes in the child list should appear after curr and before curr.next in the flattened list.

# Return the head of the flattened list. The nodes in the list must have all of their child pointers set to null.

#  Input: head = [1,2,3,4,5,6,null,null,null,7,8,9,10,null,null,11,12]
# Output: [1,2,3,7,8,11,12,9,10,4,5,6]
# Explanation: The multilevel linked list in the input is shown.

# Input: head = [1,2,null,3]
# Output: [1,3,2]
# Explanation: The multilevel linked list in the input is shown.

"""
# Definition for a Node.
class Node:
    def __init__(self, val, prev, next, child):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child
"""

class Solution:
    def flatten(self, head: 'Optional[Node]') -> 'Optional[Node]':
        
        if not head: return None
        stack = [head]
        previous = None 
        while stack:
            node = stack.pop()
            if previous:
                previous.next = node
                node.prev = previous
            previous = node
            
            if node.next:
                stack.append(node.next)
            
            if node.child:
                stack.append(node.child)
                node.child = None
        return head

# recursive
    # if not head: return None
    
    #     def travel(node):
    #         while node:
    #             q = node.next
    #             if not q: tail = node
    #             if node.child:
    #                 node.next = node.child
    #                 node.child.prev = node
    #                 t = travel(node.child)
    #                 if q:
    #                     q.prev = t
    #                 t.next= q
    #                 node.child = None
    #             node = node.next
    #         return tail
        
    #     travel(head)
    #     return head

# 117. Populating Next Right Pointers in Each Node II
# Given a binary tree

# struct Node {
#   int val;
#   Node *left;
#   Node *right;
#   Node *next;
# }
# Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.

# Initially, all next pointers are set to NULL.

# Example 1:

# Input: root = [1,2,3,4,5,null,7]
# Output: [1,#,2,3,#,4,5,7,#]
# Explanation: Given the above binary tree (Figure A), your function should populate each next pointer to point to its next right node, just like in Figure B. The serialized output is in level order as connected by the next pointers, with '#' signifying the end of each level.
# Example 2:

# Input: root = []
# Output: []

"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""
from collections import deque

class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if not root:
            return None
        que = deque([root])
        while que:
            n = len(que)
            for i in range(len(que)):
                cur = que.popleft()
                if cur.left:
                    que.append(cur.left)
                if cur.right:
                    que.append(cur.right)
                
                if i == n - 1:
                    cur.next = None 
                    continue 
                
                cur.next = que[0]
        return root

# 229. Majority Element II
# Given an integer array of size n, find all elements that appear more than ⌊ n/3 ⌋ times.

# Example 1:

# Input: nums = [3,2,3]
# Output: [3]
# Example 2:

# Input: nums = [1]
# Output: [1]
# Example 3:
# Input: nums = [1,2]
# Output: [1,2]

import collections


class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        count = collections.defaultdict(int)
        
        for num in nums:
            
            count[num] += 1
        
        res = []
        majority = len(nums) // 3 
        for key in count:
            if count[key] > majority:
                res.append(key)
        
        return res 
            
import collections

#O(n) time, O(k) space
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]: 
        ctr = collections.Counter()
        for n in nums:

            ctr[n] += 1
            if len(ctr) == 3:
                
                ctr -= collections.Counter(set(ctr))
               
        return [n for n in ctr if nums.count(n) > len(nums)/3]

#O(n) time, O(1) space
import collections
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]: 
        ctr = collections.Counter()
        for n in nums:

            ctr[n] += 1
            if len(ctr) == 3:
                
                ctr -= collections.Counter(set(ctr))
               
        return [n for n in ctr if nums.count(n) > len(nums)/3]

#finding occurence of target string in a given string 
#ex: string = "abcbcbcaawe"
# target = "cbc"
# output = 2 

def countStr(string, target):
    n = len(target)
    count = 0 
    for i in range(len(string) - n + 1):
        if string[i: i + n] == target:
            count += 1
    return count
    
# 23. Merge k Sorted Lists

import heapq
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
#         if not lists:
#             return None
#         if len(lists) == 1:
#             return lists[0]
#         mid = len(lists) // 2
#         l, r = self.mergeKLists(lists[:mid]), self.mergeKLists(lists[mid:])
#         print(l, r)
#         return self.merge(l, r)
    
#     def merge(self, l, r):
#         dummy = p = ListNode()
#         while l and r:
#             if l.val < r.val:
#                 p.next = l
#                 l = l.next
#             else:
#                 p.next = r
#                 r = r.next
#             p = p.next
#         p.next = l or r
#         return dummy.next

            h = []
            head = tail = ListNode(0)
            for i in range(len(lists)):
                if lists[i]:
                    heapq.heappush(h, (lists[i].val, i, lists[i]))

            while h:                                                        #[(1),(1),(2)]
                node = heapq.heappop(h)                                     # (1)
                node = node[2]                                              # [1,4,5]
                tail.next = node                                            #tail  node(0) -> 1
                tail = tail.next                                            # tail = 1 
                if node.next:                                               # 0 ->1 -> 4 -> 5   next = 4
                    i+=1                                                  # 3
                    # print(i)
                    heapq.heappush(h, (node.next.val, i,  node.next))        #
                # print(h)

            return head.next

# 78. Subsets
# Given an integer array nums of unique elements, return all possible subsets (the power set).

# The solution set must not contain duplicate subsets. Return the solution in any order.

# Example 1:

# Input: nums = [1,2,3]
# Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
# Example 2:

# Input: nums = [0]
# Output: [[],[0]]

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
#         res = [[]]
#         for i in range(len(nums)):
#             res += [num + [nums[i]] for num in res]
        
#         return res
        
#         res = []
        
#         self.backTrack(nums, res, [])
#         return res 
    
#     def backTrack(self, nums, res, path):
        
#         res.append(path)
#         print("nums",nums, path)
#         print(res)
#         for i in range(len(nums)):
#             print(i, nums)
#             self.backTrack(nums[i +1 :], res, path + [nums[i]])

        res = []
        self.dfs(sorted(nums), 0, [], res)
        return res

    def dfs(self, nums, index, path, res):
        res.append(path)
        print("path", path)
        print("res", res)
        for i in range(index, len(nums)):
            print(i, path)
            self.dfs(nums, i+1, path+[nums[i]], res)

#         def backtrack(first = 0, curr = []):
#             # if the combination is done
#             if len(curr) == k:  
#                 print(curr)
#                 output.append(curr[:])
#                 print(output)
#                 return
#             for i in range(first, n):
#                 print(i)
#                 # add nums[i] into the current combination
#                 curr.append(nums[i])
#                 # use next integers to complete the combination
#                 backtrack(i + 1, curr)
#                 # backtrack
#                 curr.pop()
        
#         output = []
#         n = len(nums)
#         for k in range(n + 1):
#             backtrack()
#         return output



# 3. Longest Substring Without Repeating Characters
# Given a string s, find the length of the longest substring without repeating characters.
# Example 1:

# Input: s = "abcabcbb"
# Output: 3
# Explanation: The answer is "abc", with the length of 3.
# Example 2:

# Input: s = "bbbbb"
# Output: 1
# Explanation: The answer is "b", with the length of 1.
# Example 3:

# Input: s = "pwwkew"
# Output: 3
# Explanation: The answer is "wke", with the length of 3.
# Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        start = longest = 0 
        visited = {}
        
        for i in range(len(s)):
            if s[i] in visited and start <= visited[s[i]]:
                start = visited[s[i]] + 1 
            else:
                longest = max(longest, i - start + 1)
            
            visited[s[i]] = i
        
        return longest

# 1169. Invalid Transactions
# A transaction is possibly invalid if:

# the amount exceeds $1000, or;
# if it occurs within (and including) 60 minutes of another transaction with the same name in a different city.
# You are given an array of strings transaction where transactions[i] consists of comma-separated values representing the name, time (in minutes), amount, and city of the transaction.

# Return a list of transactions that are possibly invalid. You may return the answer in any order.

# Example 1:

# Input: transactions = ["alice,20,800,mtv","alice,50,100,beijing"]
# Output: ["alice,20,800,mtv","alice,50,100,beijing"]
# Explanation: The first transaction is invalid because the second transaction occurs within a difference of 60 minutes, have the same name and is in a different city. Similarly the second one is invalid too.
# Example 2:

# Input: transactions = ["alice,20,800,mtv","alice,50,1200,mtv"]
# Output: ["alice,50,1200,mtv"]
# Example 3:

# Input: transactions = ["alice,20,800,mtv","bob,50,1200,mtv"]
# Output: ["bob,50,1200,mtv"]

import collections

class Solution:
    def invalidTransactions(self, transactions: List[str]) -> List[str]:
        
        r = {}
                
        inv = []        
        for i in transactions:
            split = i.split(",")
            name = str(split[0])
            time = int(split[1])
            amount = int(split[2])
            city = str(split[3])
            
            if time not in r:
                r[time] = {
                    name: [city]
                }
            else:
                if name not in r[time]:
                    r[time][name]=[city]
                else:
                    r[time][name].append(city)
                    
        
        for i in transactions:
            split = i.split(",")
            name = str(split[0])
            time = int(split[1])
            amount = int(split[2])
            city = str(split[3])
            
            
            if amount > 1000:
                inv.append(i)
                continue
            
            for j in range(time-60, time+61):
                if j not in r:
                    continue
                if name not in r[j]:
                    continue
                if len(r[j][name]) > 1 or (r[j][name][0] != city):
                    inv.append(i)
                    break
                                        
        return inv   


# 1169 · Permutation in String
# Description
# Given two strings s1 and s2, write a function to return true if s2 contains the permutation of s1. In other words, one of the first string's permutations is the substring of the second string.

# The input strings only contain lower case letters.
# The length of both given strings is in range [1, 10,000].
# Example
# Example 1:

# Input:s1 = "ab" s2 = "eidbaooo"
# Output:true
# Explanation: s2 contains one permutation of s1 ("ba").
# Example 2:

# Input:s1= "ab" s2 = "eidboaoo"
# Output: false

class Solution:
    """
    @param s1: a string
    @param s2: a string
    @return: if s2 contains the permutation of s1
    """
    def check_inclusion(self, s1: str, s2: str) -> bool:
        # write your code here
        l1 = len(s1)
        need = collections.Counter(s1)
        missing = l1
        for i,c in enumerate(s2):
            if c in need: 
                if need[c] > 0: missing -= 1    
                need[c] -= 1                    
            if i>=l1 and s2[i-l1] in need:      
                need[s2[i-l1]] += 1            
                if need[s2[i-l1]]>0: missing += 1  
            if missing == 0:
                return True
        return False

#ex: [0,3,2,5], 5 -> 2
# [1,2,1,1], 3  ->3
# [1,2,1,1,2] -> 6
def twoPairSUm(arr, target):
    if not arr and target is None:
        return 0
    
    arr = sorted(arr)
    
    start, end = 0, len(arr) - 1
    count = 0
    while start < end:                          #[1, 1, 1, 2] 3 , 0 , 3
        if arr[start] + arr[end] < target:         #5
            start += 1                               
        elif arr[start] + arr[end] > target:        # 5
            end -= 1                                
        else:       
            start_dup, end_dup = 1, 1                              
            while start < end and arr[start] == arr[start +1]:
                start_dup += 1
                start += 1
            while start < end and arr[end] == arr[end - 1]:      # 3 * 1 = 3
                end_dup += 1
                end -= 1
                
            count += start_dup * end_dup
            
    return count


# 1281 · Top K Frequent Elements
# Description
# Given a non-empty array of integers, return the k most frequent elements.

# You may assume k is always valid, 1 ≤ k ≤ number of unique elements.
# Your algorithm's time complexity must be better than O(n log n), where n is the array's size.
# Example
# Example 1:

# Input: nums = [1,1,1,2,2,3], k = 2
# Output: [1,2]
# Example 2:

# Input: nums = [1], k = 1
# Output: [1]

import collections 
class Solution:
    """
    @param nums: the given array
    @param k: the given k
    @return: the k most frequent elements
             we will sort your return value in output
    """
    def top_k_frequent(self, nums: List[int], k: int) -> List[int]:
        # Write your code here
        counter = collections.Counter(nums)
        targets = counter.most_common(k)
        res = [num for num, target in targets]
        return res 