#!/bin/python3

import math
import os
import random
import re
import sys



#
# Complete the 'getTimes' function below.
#
# The function is expected to return an INTEGER_ARRAY.
# The function accepts following parameters:
#  1. INTEGER_ARRAY time
#  2. INTEGER_ARRAY direction
#

get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]

def getTimes(time, direction):
    # Write your code here
    waiting_enter = []
    waiting_exit = []
    last_action = 1
    lastEntryTime = time[-1]
    exit_at = {}
    for t in range(lastEntryTime+1):
        person_waiting_at_t = get_indexes(t, time)
        for person in person_waiting_at_t:
            if direction[person] == 1:
                waiting_exit.append(person)
            else:
                waiting_enter.append(person)
        if last_action == 0 and len(waiting_enter) > 0:
            exit_at[waiting_enter.pop()] = t
            last_action = 0
        elif len(waiting_exit) > 0:
            exit_at[waiting_exit.pop()] = t
            last_action = 1
        elif len(waiting_enter) > 0:
            exit_at[waiting_enter.pop()] = t
            last_action = 0
    keys = exit_at.keys()
    values = list(exit_at.values())
    final_exit_times = [None] * len(time)
    for key in keys:
        final_exit_times[key] = values[key]
    return final_exit_times


if __name__ == '__main__':
    time = [0, 0, 1, 5]
    direction = [0, 1, 1, 0]
    result = getTimes(time, direction)
    print(result)