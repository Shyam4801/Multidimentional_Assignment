import sys
import json
import numpy as np
from time import perf_counter_ns
# from lap import lapjv
from numba import njit, prange

@njit()
def auction(X, eps=0.33):
    num_persons, num_objects = X.shape
    
    cost = np.zeros(num_objects)
    
    idx1 = np.zeros(num_persons, dtype=np.int32) - 1
    idx2 = np.zeros(num_persons, dtype=np.int32) - 1
    val1 = np.zeros(num_persons) - 1
    val2 = np.zeros(num_persons) - 1
    
    highest_bid   = np.zeros(num_objects) - 1
    high_bidder = np.zeros(num_objects, dtype=np.int32) - 1

    object2person = np.zeros(num_objects,  dtype=np.int32) - 1
    person2object = np.zeros(num_persons, dtype=np.int32) - 1
    
    unassigned_person = num_persons
    while unassigned_person > 0:
        
        idx1.fill(-1)
        idx2.fill(-1)
        val1.fill(-1)
        val2.fill(-1)
        highest_bid.fill(-1)
        high_bidder.fill(-1)
        
        #Bidding phase 
        for bidder in range(num_persons):
            if person2object[bidder] != -1: continue
            
            for item in range(num_objects):
                val = X[bidder, item] - cost[item]
                if val > val1[bidder]:
                    idx2[bidder] = idx1[bidder]
                    idx1[bidder] = item
                    
                    val2[bidder] = val1[bidder]
                    val1[bidder] = val
                    
                elif val > val2[bidder]:
                    idx2[bidder] = item
                    val2[bidder] = val
        
        # Competing phase 
        for bidder in range(num_persons):
            if person2object[bidder] != -1: continue
            
            bid = val1[bidder] - val2[bidder] + eps
            
            if bid > highest_bid[idx1[bidder]]:
                highest_bid[idx1[bidder]]   = bid
                high_bidder[idx1[bidder]] = bidder
        
        # Assignment phase
        for item in range(num_objects):
            if highest_bid[item] == -1: continue
            
            cost[item] += highest_bid[item]
            
            if object2person[item] != -1:
                person2object[object2person[item]] = -1
                unassigned_person += 1
            
            object2person[item]              = high_bidder[item]
            person2object[high_bidder[item]] = item
            unassigned_person -= 1
    
    return person2object, cost

