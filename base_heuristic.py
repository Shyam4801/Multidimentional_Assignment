# Base heuristic 

from auction import auction
import numpy as np


X=np.array([[5,2,6],[1,4,7],[9,8,2]])

def three_dim_ass(arr,eps=0.1):
  lm_arr = np.zeros((3,3))
  jl_arr = np.zeros((3,3))

  j=0
  for l in range(3):
    for m in range(3):
      lm_arr[l][m] = arr[:,l,m].min()
  
  # row_ind, col_ind = linear_sum_assignment(lm_arr)
  # cost = lm_arr[row_ind, col_ind].sum()
  # machine2worker = col_ind
  machine2worker, cost = auction(lm_arr)
  auc_score = lm_arr[(np.arange(lm_arr.shape[0]), machine2worker)].sum()
  print(lm_arr,machine2worker,auc_score,cost)

  for j in range(3):
    for l in range(3):
      jl_arr[l][j] = arr[j,l,machine2worker[l]]

  print('jobs to l machine')
  print(jl_arr)

  # row_ind, col_ind = linear_sum_assignment(jl_arr)
  # cost = jl_arr[row_ind, col_ind].sum()
  # jobstomachine = col_ind
  jobstomachine, cost = auction(jl_arr,eps=eps)
  auc_score = jl_arr[(np.arange(jl_arr.shape[0]), jobstomachine)].sum()
  print(jl_arr,jobstomachine,auc_score,cost)

  # 3 dim cost
  final_cost = arr[0,jobstomachine[0],machine2worker[jobstomachine[0]]] + arr[1,jobstomachine[jobstomachine[1]],machine2worker[1]] + arr[2,jobstomachine[2],machine2worker[jobstomachine[2]]]
  print(arr[0,jobstomachine[0],machine2worker[jobstomachine[0]]] ,arr[1,jobstomachine[1],machine2worker[jobstomachine[1]]] , arr[2,jobstomachine[2],machine2worker[jobstomachine[2]]])
  return jobstomachine,machine2worker,final_cost


np.random.seed(123)
c = np.random.randint(0, 10, size=(3, 3, 3))#[0][0]
jobstomachine,machine2worker,final_cost = three_dim_ass(c)
print(jobstomachine,"|",machine2worker,"|",final_cost)