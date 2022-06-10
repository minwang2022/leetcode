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


# 919 · Meeting Rooms II
# Description
# Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei), find the minimum number of conference rooms required.)

# Contact me on wechat to get Amazon、Google requent Interview questions . (wechat id : jiuzhang15)


# (0,8),(8,10) is not conflict at 8

# Example
# Example1

# Input: intervals = [(0,30),(5,10),(15,20)]
# Output: 2
# Explanation:
# We need two meeting rooms
# room1: (0,30)
# room2: (5,10),(15,20)
# Example2

# Input: intervals = [(2,7)]
# Output: 1
# Explanation: 
# Only need one meeting room

import collections 
"""
Definition of Interval:
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
"""

class Solution:
    """
    @param intervals: an array of meeting time intervals
    @return: the minimum number of conference rooms required
    """
    def min_meeting_rooms(self, intervals: List[Interval]) -> int:
        # Write your code here
        # visited = collections.defaultdict(int)
        # count = 0

        # for interval in intervals:
        #     start, end = interval.start, interval.end
        #     while start < end:
                
        #         visited[start] = visited.get(start, 0) +1 
                
        #         count = max(count, visited[start])
                    
        #         start += 1
        # # print(visited)
        # return count
        points = []
        for interval in intervals:
            points.append((interval.start, 1))
            points.append((interval.end, -1))
        print(sorted(points))
        meeting_rooms = 0
        ongoing_meetings = 0
        for _, delta in sorted(points):
            ongoing_meetings += delta
            meeting_rooms = max(meeting_rooms, ongoing_meetings)
            
        return meeting_rooms

# 1897 · Meeting Room III
# Description
# you have a list intervals of current meetings, and some meeting rooms with start and end timestamp.When a stream of new meeting ask coming in, judge one by one whether they can be placed in the current meeting list without overlapping..A meeting room can only hold one meeting at a time. Each inquiry is independent.

# The meeting asked can be splited to some times. For example, if you want to ask a meeting for [2, 4], you can split it to [2,3] and [3, 4].

# Ensure that Intervals can be arranged in rooms meeting rooms
# The start and end times of any session are guaranteed to take values in the range [1, 50000]
# |Intervals| <= 50000
# |ask| <= 50000
# 1 <= rooms <= 20

# Example
# Example 1:

# Input:
# Intervals:[[1,2],[4,5],[8,10]], rooms = 1, ask: [[2,3],[3,4]]
# Output: 
# [true,true]
# Explanation:
# For the ask of [2,3], we can arrange a meeting room room0.
# The following is the meeting list of room0:
# [[1,2], [2,3], [4,5], [8,10]]
# For the ask of [3,4], we can arrange a meeting room room0.
# The following is the meeting list of room0:
# [[1,2], [3,4], [4,5], [8,10]]
import collections
class Solution:
    """
    @param intervals: the intervals
    @param rooms: the sum of rooms
    @param ask: the ask
    @return: true or false of each meeting
    """
    def meeting_room_i_i_i(self, intervals: List[List[int]], rooms: int, ask: List[List[int]]) -> List[bool]:
        # Write your code here.
        visited = collections.defaultdict(int)
        for start, end in intervals:

            i, j = start, end
            visited[i] += 1
            
            while j - i > 1: 
                i += 1
                visited[i] += 1
        # print(visited)
        res = []
        for start, end in ask:

            start_time, end_time = start, end
            
            if visited[start_time] < rooms:
                flag = True
            else:
                flag = False 

            if flag:
                while end_time - start_time >= 1 :           # 19 - 18 = 1
                    if start_time in visited and visited[start_time] >= rooms:
                        flag = False 
                        break 
                    
                    start_time += 1

            res.append(flag)

        return res

# 300 · Meeting Room IV
# Description
# Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei) and the value of each meeting. You can only attend a meeting at the same time. Please calculate the most value you can get.

# Contact me on wechat to get Amazon、Google requent Interview questions . (wechat id : jiuzhang15)


# 0 \leq len(meeting) = len(value) \leq 10\,0000≤len(meeting)=len(value)≤10000
# 1 \leq meeting[i][0] < meeting[i][1] \leq 50\,0001≤meeting[i][0]<meeting[i][1]≤50000
# value_i \leq 10\,000value 
# i
# ​
#  ≤10000
# (0,8),(8,10) is not conflict at 8

# Example
# Example 1

# Input:
# meeting = [[10,40],[20,50],[30,45],[40,60]]
# value = [3,6,2,4]
# Output: 7
# Explanation: You can take the 0th meeting and the 3th meeting, you can get 3 + 4 = 7.
# Example 2

# Input:
# meeting = [[10,20],[20,30]]
# value = [2,4]
# Output: 6
# Explanation: You can take the 0th meeting and the 1st meeting, you can get 2 + 4 = 6.

MAX_TIME = 50000
class Solution:
    """
    @param meeting: the meetings
    @param value: the value
    @return: calculate the max value
    """
    def maxValue(self, meetings, values):
        meeting_end_time_to_index_value = collections.defaultdict(list)
        for i in range(len(meetings)):
            meeting_end_time_to_index_value[meetings[i][1]].append((meetings[i][0], values[i]))
        print(meeting_end_time_to_index_value)
        dp = [0] * (MAX_TIME + 1)
        for i in range(1, MAX_TIME + 1):
            dp[i] = dp[i - 1]
            print(meeting_end_time_to_index_value[i])
            for j in range(len(meeting_end_time_to_index_value[i])):
                start = meeting_end_time_to_index_value[i][j][0]
                value = meeting_end_time_to_index_value[i][j][1]
                print("i,j",i, j)
                print("start",start, value)
                dp[i] = max(dp[i], dp[start] + value)
        
        return max(dp)


# 127. Word Ladder
# A transformation sequence from word beginWord to word endWord using a dictionary wordList is a sequence of words beginWord -> s1 -> s2 -> ... -> sk such that:

# Every adjacent pair of words differs by a single letter.
# Every si for 1 <= i <= k is in wordList. Note that beginWord does not need to be in wordList.
# sk == endWord
# Given two words, beginWord and endWord, and a dictionary wordList, return the number of words in the shortest transformation sequence from beginWord to endWord, or 0 if no such sequence exists.

 

# Example 1:

# Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
# Output: 5
# Explanation: One shortest transformation sequence is "hit" -> "hot" -> "dot" -> "dog" -> cog", which is 5 words long.
# Example 2:

# Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]
# Output: 0
# Explanation: The endWord "cog" is not in wordList, therefore there is no valid transformation sequence.
class Solution:
    DICTIONARY = ["a", "b", "c","d", "e", "f","g", "h", "i","j", "k", "l","m", "n", "o","p", "q", "r","s", "t", "u","v", "w", "x","y", "z"]
    import collections
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if not beginWord or not endWord:
            return 0 
        
        que = collections.deque([beginWord])
        visited = {beginWord: 1}
        new_dict = {}
        for word in wordList:
            new_dict[word] = 1
        while que:
            word = que.popleft()
            if word == endWord:
                return visited[word]
            
            if self.generate_words(new_dict, visited, que, word):
                self.generate_words(new_dict, visited, que, word)
            
        return 0 
    
    def generate_words(self, wordList, visited, que, word):
        DICTIONARY = ["a", "b", "c","d", "e", "f","g", "h", "i","j", "k", "l","m", "n", "o","p", "q", "r","s", "t", "u","v", "w", "x","y", "z"]
        for i in range(len(word)):
            for ele in DICTIONARY:
                if word[i] == ele:
                    continue 
                
                new_word = word[:i] + ele + word[i + 1:]
                if not new_word in visited and new_word in wordList:
                    visited[new_word] = visited[word] + 1
                    que.append(new_word)
                
        return True 

class Solution:
    import collections
    def ladderLength1(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if not beginWord or not endWord or not wordList:
            return 0 
        new_dict = self.constructList(wordList)

        return self.findWordLayer(new_dict, beginWord, endWord)

    def constructList(self, wordList):
        
        dic = collections.defaultdict(list)
        
        for word in wordList:
            for i in range(len(word)):
                newWord = word[:i] + "_" + word[i + 1:]
                dic[newWord].append(word)
        
        
        return dic
    
    def findWordLayer(self, dic, begin, end):
        que = collections.deque([begin])
        visited = {begin: 1}
        
        while que:
            word = que.popleft()
            if word == end:
                return visited[word]
            
            for i in range(len(word)):
                newWord = word[:i] + "_" + word[i + 1:]
                if newWord not in dic:
                    continue 
                for neigh in dic[newWord]:
                    if neigh in visited:
                        continue
                    que.append(neigh)
                    visited[neigh] = visited[word] + 1
        return 0

# 126. Word Ladder II
# A transformation sequence from word beginWord to word endWord using a dictionary wordList is a sequence of words beginWord -> s1 -> s2 -> ... -> sk such that:

# Every adjacent pair of words differs by a single letter.
# Every si for 1 <= i <= k is in wordList. Note that beginWord does not need to be in wordList.
# sk == endWord
# Given two words, beginWord and endWord, and a dictionary wordList, return all the shortest transformation sequences from beginWord to endWord, or an empty list if no such sequence exists. Each sequence should be returned as a list of the words [beginWord, s1, s2, ..., sk].

 

# Example 1:

# Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
# Output: [["hit","hot","dot","dog","cog"],["hit","hot","lot","log","cog"]]
# Explanation: There are 2 shortest transformation sequences:
# "hit" -> "hot" -> "dot" -> "dog" -> "cog"
# "hit" -> "hot" -> "lot" -> "log" -> "cog"
# Example 2:

# Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]
# Output: []
# Explanation: The endWord "cog" is not in wordList, therefore there is no valid transformation sequence.

class Solution:
    import collections
    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        if not beginWord or not endWord or not wordList or beginWord == endWord or endWord not in wordList:
            return []
        
        new_dict = collections.defaultdict(list)
        length = len(beginWord)
        for word in wordList:
            for i in range(length):
                new_word = word[:i] + "*" + word[i + 1:]
                new_dict[new_word].append(word)

        res = []
        que = collections.deque()
        que.append((beginWord, [beginWord]))
        visited = set([beginWord])
        
        while que and not res:
            size = len(que)
            local_set = set()
            # print(que)
            for _ in range(size):
                word, path = que.popleft()
                
                for i in range(length):
                    new_word = word[:i] + "*" + word[i + 1:]
                    for neighbor in new_dict[new_word]:
                        if neighbor == endWord:
                            res.append(path + [neighbor])
                            
                        if neighbor not in visited:
                            local_set.add(neighbor)
                            que.append((neighbor, path + [neighbor]))
            visited = visited.union(local_set)
        return res