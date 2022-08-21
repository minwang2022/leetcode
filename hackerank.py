
# Website Pagnation 
# Complete the 'fetchItemsToDisplay' function below.
#
# The function is expected to return a STRING_ARRAY.
# The function accepts following parameters:
#  1. 2D_STRING_ARRAY items
#  2. INTEGER sortParameter
#  3. INTEGER sortOrder
#  4. INTEGER itemsPerPage
#  5. INTEGER pageNumber
#

def fetchItemsToDisplay(items, sortParameter, sortOrder, itemsPerPage, pageNumber):
    # Write your code here
    a = sorted(items, key=lambda item: int(item[sortParameter]) if sortParameter else item[sortParameter], reverse=sortOrder)
    print(a)
    n = len(a)  
    count = 0
    res = []
    idx = itemsPerPage * pageNumber 
    while count < itemsPerPage and idx < n:
        res.append(a[idx][0])
        idx += 1
        count += 1
    return res


#order check 
# Complete the 'countStudents' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY height as parameter.
#

def countStudents(height):
    # Write your code here
    sorted_height = sorted(height)
    count = 0 
    for i in range(len(height)):
        if height[i] != sorted_height[i]:
            count += 1
    
    return count

#array game

# Complete the 'countMoves' function below.
#
# The function is expected to return a LONG_INTEGER.
# The function accepts INTEGER_ARRAY numbers as parameter.
#

def countMoves(numbers):
    # Write your code here
    return sum(numbers) - (len(numbers) * min(numbers))

# number of moves  (knight move from start position to end position)

# Complete the 'minMoves' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER startRow
#  3. INTEGER startCol
#  4. INTEGER endRow
#  5. INTEGER endCol
#
import collections 
DIRECTIONS = [(-2, -1), (-2, 1), (-1, 2), (1, 2),
                (2, 1), (2, -1), (1, -2), (-1, -2)]
def minMoves(n, startRow, startCol, endRow, endCol):
    # Write your code here
    que = collections.deque([(startRow, startCol)])
    distance = {(startRow, startCol): 0}
    
    while que:
        x, y = que.popleft()
        print((x, y))
        if (x,y) == (endRow, endCol):
            return distance[(x, y)]
        
        for delta_x, delta_y in DIRECTIONS:
            new_x, new_y = x + delta_x, y + delta_y
            if not isValid(n,new_x,new_y):
                continue 
            if (new_x, new_y) in distance:
                continue 
            que.append((new_x, new_y))
            
            distance[(new_x, new_y)] = distance[(x,y)] + 1
    return -1       
def isValid( n, i, j):
    
    if not (0 <= i < n and 0 <= j < n):
        return False
    return True 



# Given the root of a complete binary tree, return the number of the nodes in the tree.
# According to Wikipedia, every level, except possibly the last, is completely filled in a complete binary tree, and all nodes in the last level are as far left as possible. It can have between 1 and 2h nodes inclusive at the last level h.

# 		1
# 	2		3
# 4		5 6		7
# 8	9	10 	11



Input: root = [1,2,3,4,5,6]
Output: 6
Input: root = []
Output: 0
Input: root = [1]
Output: 1
def findNumNodes( root):
	if  not root: 
		return 0
	left  = findnumnodes(root.left)
	Right = findnumnodes(root.right)
	
	return left + right + 1

def findNumNodes( root):
	leftPointer, rightPointer, leftHeight, rightHeight = root,left, root.right, 1, 1
	
	return left


    # let leftPointer = root.left
    # let leftHeight = 1
    # let rightPointer = root.right
    # let rightHeight = 1

    # while (leftPointer !== null) {
        
    #     leftHeight++
    #     leftPointer = leftPointer.left
        
        
    # }
    
    # while (rightPointer !== null) {
        
    #     rightHeight++
    #     rightPointer = rightPointer.right
        
    # }

    # if (leftHeight === rightHeight) {
    #     return 2 ** leftHeight - 1
    # }
    
    # return 1 + findNumNodes(root.left) + findNumNodes(root.right)

