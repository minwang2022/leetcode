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