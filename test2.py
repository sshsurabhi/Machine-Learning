#!/bin/python3

import math
import os
import random
import re
import sys



#
# Complete the 'longestVowelSubsequence' function below.
#
# The function is expected to return an INTEGER.
# The function accepts STRING s as parameter.
#

def longestVowelSubsequence(s):
    # Write your code here
    vowels = 'aeiou'
    checkIndex = 0
    sequence = ''
    for vowel in vowels:
        if s.find(vowel) >= 0:
            continue
        else:
            return 0

    for i in s:
        if len(sequence) == 0 and vowels.find(i) > 0:
            continue
        if vowels.find(i) >= checkIndex:
            sequence += i
            checkIndex = vowels.find(i)
    return len(sequence)

if __name__ == '__main__':
    text = 'aeiaaiooaa'
    result = longestVowelSubsequence(text)
    print(result)