
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