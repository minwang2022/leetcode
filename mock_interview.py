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


