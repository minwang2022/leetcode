Dynamic Programming - Memoization 

107 · Word Break

# Description
# Given a string s and a dictionary of words dict, determine if s can be broken into a space-separated sequence of one or more dictionary words.
# Because we have used stronger data, the ordinary DFS method can not pass this question now.
# Example 1:

# Input:
# s = "lintcode"
# dict = ["lint", "code"]
# Output:

# true
# Explanation:

# Lintcode can be divided into lint and code.
# Example 2:
# Input:

# s = "a"
# dict = ["a"]
# Output:

# true
# Explanation:

# a is in the dict.
def wordBreak(self, s, wordSet):
        # write your code here
        if not s:
            return True 
        if not wordSet:
            return False 
        max_len = len(max(wordSet, key = len))

        return self.dfs(s, 0, max_len, wordSet, {})
    
def dfs(self, s, index, max_len, dict, memo):
    if index in memo:
        return memo[index]
    if index == len(s):
        return True
    for end in range(index + 1, len(s) + 1):
        if (end - index) > max_len:
            break 
        word = s[index: end]
        # print(word, index, end)
        if word not in dict:
            # print("word nit in dict", word)
            continue
        if self.dfs(s, end, max_len, dict, memo):
            return True
            
    memo[index] = False
    # print(memo[index])
    return False
683 · Word Break III
# Description
# Give a dictionary of words and a sentence with all whitespace removed, return the number of sentences you can /
# form by inserting whitespaces to the sentence so that each word can be found in the dictionary.

# Ignore case

# Example
# Example1

# Input:
# "CatMat"
# ["Cat", "Mat", "Ca", "tM", "at", "C", "Dog", "og", "Do"]
# Output: 3
# Explanation:
# we can form 3 sentences, as follows:
# "CatMat" = "Cat" + "Mat"
# "CatMat" = "Ca" + "tM" + "at"
# "CatMat" = "C" + "at" + "Mat"
# Example1

# Input:
# "a"
# []
# Output: 
# 0
def wordBreak3(self, s, dict):
        # Write your code here
        if not s or not dict:
            return 0
        
        max_len, lower_dict = self.initialize(dict)
        return self.dfs(s.lower(), 0, max_len, lower_dict, {})

def initialize(self, dict):
    max_len = 0 
    lower_dict = set()
    for word in dict:
        lower_dict.add(word.lower())
        max_len = max(max_len, len(word))
    
    return max_len, lower_dict

def dfs(self, s, index, max_len, dict, memo):
    if index in memo:
        return memo[index]
    if index == len(s):
        return 1 
    
    result = 0
    for end in range(index + 1, len(s) + 1):
        if (end - index) > max_len:
            break 
        word = s[index: end]
        if word not in dict:
            continue
        result += self.dfs(s, end, max_len, dict, memo)

    memo[index] = result
    return memo[index]

582 · Word Break II

# Description
# Given a string s and a dictionary of words dict, add spaces in s to construct a sentence where each word is a valid dictionary word.

# Return all such possible sentences.

# Example
# Example 1:

# Input："lintcode"，["de","ding","co","code","lint"]
# Output：["lint code", "lint co de"]
# Explanation：
# insert a space is "lint code"，insert two spaces is "lint co de".
# Example 2:

# Input："a"，[]
# Output：[]
# Explanation：dict is null.

# Solution:
def wordBreak(self, s, wordDict):
        # write your code here
        if not s or not wordDict:
            return []
        max_len = len(max(wordDict, key = len))

        return self.dfs(s, max_len, wordDict, {})
    
def dfs(self, s, max_len, dict, memo):
    if s in memo:
        return memo[s]
    if len(s) == 0:
        print("here end here", len(s))
        return []
    
    partitions = []

    for prefix in range(1, len(s)):
        if prefix > max_len:
            break 
        word = s[:prefix]
        print("prefix and word", prefix, word)
        if word not in dict:
            print("skip word", word)
            continue 
        sub_partitions = self.dfs(s[prefix:], max_len, dict, memo)
        print("sub_partitions = ", sub_partitions)

        for partition in sub_partitions:
            print("each partition", partition)
            partitions.append(word + " " + partition)
            print("partitions.append(word + " " + partition)", word, partition, partitions)

    if s in dict:
        partitions.append(s)
    memo[s] = partitions
    print("memo", memo)
    print("end partitions =", partitions )
    return partitions


829 Word Pattern II
# Description
# Given a pattern and a string str, find if str follows the same pattern.

# Here follow means a full match, such that there is a bijection between a letter in pattern and a non-empty substring in str.(i.e if a corresponds to s, then b cannot correspond to s. For example, given pattern = "ab", str = "ss", return false.)

# You may assume both pattern and str contains only lowercase letters.

# Example
# Example 1

# Input:
# pattern = "abab"
# str = "redblueredblue"
# Output: true
# Explanation: "a"->"red","b"->"blue"
# Example 2

# Input:
# pattern = "aaaa"
# str = "asdasdasdasd"
# Output: true
# Explanation: "a"->"asd"
# Example 3

# Input:
# pattern = "aabb"
# str = "xyzabcxzyabc"
# Output: false

def wordPatternMatch(self, pattern, str):
    # write your code here
    if not pattern or not str:
        return False
    return self.isMatch(pattern, str, {}, set())

def isMatch(self, pattern, string, mapping, used):
    if not pattern:
        return not string 
    char = pattern[0]
    if char in mapping:
        word = mapping[char]
        if not string.startswith(word):
            return False 
        return self.isMatch(pattern[1:], string[len(word):], mapping, used)
    
    for i in range(len(string)):
        word = string[:i + 1]
        if word in used:
            continue
        mapping[char] = word
        used.add(word)
        if self.isMatch(pattern[1:], string[i + 1:], mapping, used):
            return True 
        del mapping[char]
        used.remove(word)
    return False

