# Base heuristic 
import numpy as np 

X=np.array([[5,2,6],[1,4,7],[9,8,2]])

def three_dim_ass(arr,eps=0.1):
  lm_arr = np.zeros((3,3))
  jl_arr = np.zeros((3,3))

  j=0
  for l in range(3):
    for m in range(3):
      lm_arr[l][m] = arr[:,l,m].min()
  
  bidder2item, cost = auction(lm_arr)
  auc_score = lm_arr[(np.arange(lm_arr.shape[0]), bidder2item)].sum()
  print(lm_arr,bidder2item,auc_score,cost)

  for j in range(3):
    for l in range(3):
      jl_arr[l][j] = arr[j,l,bidder2item[l]]

  print('jobs to l machine')
  print(jl_arr)

  jobstomachine, cost = auction(jl_arr,eps=eps)
  auc_score = jl_arr[(np.arange(jl_arr.shape[0]), jobstomachine)].sum()
  print(jl_arr,jobstomachine,auc_score,cost)

  # 3 dim cost
  final_cost = arr[0,jobstomachine[0],bidder2item[0]] + arr[1,jobstomachine[1],bidder2item[1]] + arr[2,jobstomachine[2],bidder2item[2]]
  print(arr[0,jobstomachine[0],bidder2item[0]] ,arr[1,jobstomachine[1],bidder2item[1]] , arr[2,jobstomachine[2],bidder2item[2]])
  return bidder2item,jobstomachine,final_cost