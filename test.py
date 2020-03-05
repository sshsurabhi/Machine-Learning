#!/bin/python3

import math
import os
import random
import re
import sys



#
# Complete the 'calculateScore' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. STRING text
#  2. STRING prefixString
#  3. STRING suffixString
#

def getPrefixScore(text, prefixString):
    matchedString = ''
    for i in range(len(prefixString)):
        subString = prefixString[-(i+1):]
        if(subString in text):
            matchedString = subString
            continue
        break
    return matchedString, len(matchedString)


def getSuffixScore(text, suffixString):
    matchedString = ''
    for i in range(len(suffixString)):
        subString = suffixString[:(i+1)]
        if(subString in text):
            matchedString = subString
            continue
        break
    return matchedString, len(matchedString)

def calculateScore(text, prefixString, suffixString):
    # Write your code here
    prefix, prefixScore = getPrefixScore(text, prefixString)
    suffix, suffixScore = getSuffixScore(text, suffixString)
    totalScore = prefixScore + suffixScore
    prefixStartsAt = text.find(prefix)
    suffixStartsAt = text.find(suffix)
    return text[prefixStartsAt:suffixStartsAt+suffixScore]


if __name__ == '__main__':
    text = 'engine'
    prefixString = 'raven'
    suffixString = 'ginkgo'
    result = calculateScore(text, prefixString, suffixString)
    print(result)