# 780 · Remove Invalid Parentheses
# Description
# Remove the minimum number of invalid parentheses in order to make the input string valid. Return all possible results.

# The input string may contain letters other than the parentheses ( and ).

# Example 1:

# Input:
# "()())()"
# Ouput:
# ["(())()","()()()"]
# Example 2:

# Input:
# "(a)())()"
# Output:
#  ["(a)()()", "(a())()"]
# Example 3:

# Input:
# ")(" 
# Output:
#  [""]

from typing import (
    List,
)

class Solution:
    """
    @param s: The input string
    @return: Return all possible results
             we will sort your return value in output
    """
    def removeInvalidParentheses(self, s):
        res = []
        if len(s) == 0:
            res.append("")
            return res
        q = []
        S = set()
        q.append(s)
        S.add(s)
        flag = False
        # bfs
        while len(q) > 0:
            curStr = q.pop(0)
            # 若curStr有效，则找到答案
            if self.check(curStr):
                res.append(curStr)
                flag = True
            #不需要去找更短的串
            if flag:
                continue
            # 对于无效的字符串，依次删除一个字符压入队列
            for i in range(len(curStr)):
                if curStr[i] != '(' and curStr[i] != ')':
                    continue;
                str1 = curStr[:i]
                str2 = curStr[i + 1:]
                str3 = str1 + str2
                sizeSet = len(S)
                S.add(str3)
                # 如果这个字符串未被check过，则压入队列
                if sizeSet != len(S):
                    q.append(str3)
        return res
        
    def check(self,str):
        if len(str) == 0:
            return True
        # 记录count对
        count = 0
        for i in range(0,len(str)):
            if str[i] == '(':
                count += 1
            if str[i] == ')':
                count -= 1
            if count < 0:
                return False
        if count == 0:
            return True
        return False