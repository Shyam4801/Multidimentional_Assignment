# Rollout 
import numpy as np
from base_heuristic import three_dim_ass
from auction import auction

jl_arr = np.zeros((3,3))
fixed = 0

def three_dim_rollout(arr,fixed_l,fixed_j):
  lm_arr = []
  j_index = []
  for l in range(3): 
    for m in range(3): 
      if l == fixed_l:
        lm_arr.append(arr[fixed_j,fixed_l,m])
      elif l in final_l_ass:
        lm_arr.append(arr[np.where(final_l_ass[0] == l)[0][0],l,m])
      else:
        if fixed_j == 2:
          lm_arr.append(arr[fixed_j:,l,m].min())
        else:
          lm_arr.append(arr[fixed_j+1:,l,m].min())
  mat_size = int(np.sqrt(len(lm_arr)))
  lm_arr = np.asarray(lm_arr).reshape((mat_size,mat_size))
  
  # row_ind, col_ind = linear_sum_assignment(lm_arr)
  # cost = lm_arr[row_ind, col_ind].sum()
  # machine2worker = col_ind
  machine2worker, cost = auction(lm_arr)
  auc_score = lm_arr[(np.arange(lm_arr.shape[0]), machine2worker)].sum()
  print(lm_arr,'l to worker',machine2worker,auc_score,cost)

  tmp_arr = []
  index_arr = []
  j_idx=[]

  if fixed_j != 2:
    fixed_j += 1

  for j in range(fixed_j,3): 
    for l in range(3): 
      if l != fixed_l and l not in final_l_ass:
        tmp_arr.append(arr[j,l,machine2worker[l]])
        index_arr.append(l)
        j_idx.append(j)
        # print('tmp_arr: ',tmp_arr,'index_arr: ',index_arr)
   
  sq_root = int(np.sqrt(len(tmp_arr)))
  tmp_arr = np.asarray(tmp_arr).reshape((sq_root,sq_root))
  index_arr = np.asarray(index_arr).reshape((sq_root,sq_root))

  print('jobs to l machine')
  print(tmp_arr)

  # row_ind, col_ind = linear_sum_assignment(tmp_arr)
  # cost = tmp_arr[row_ind, col_ind].sum()
  # jobstomachine = col_ind
  # print(tmp_arr, jobstomachine)
  jobstomachine, cost = auction(tmp_arr)
  auc_score = tmp_arr[(np.arange(tmp_arr.shape[0]), jobstomachine)].sum()
  print(tmp_arr,jobstomachine,'jtom index: ',index_arr,auc_score,cost)

  j2m = final_l_ass[0][np.where(final_l_ass[0] != -1)[0]]
  j2m = np.append(j2m,fixed_l)
  for i in range(len(jobstomachine)):
    j2m = np.append(j2m,index_arr[i,jobstomachine[i]])
  m2w = machine2worker[j2m] 
  s_cost = arr[(np.arange(arr.shape[0]),j2m,m2w)].sum()
  print(s_cost,(np.arange(arr.shape[0]),j2m,m2w) )
  return s_cost, m2w,j2m


# c = np.array([[5,1,2],[5,5,6],[5,5,7],[2,6,8],[9,2,9],[0,3,3],[4,1,9],[4,9,8],[1,3,1]]).reshape(3,3,3)
np.random.seed(123)
c = np.random.randint(0, 10, size=(3, 3, 3))
print()
print('cost matrix: ',c)
print()
print()
final_l_ass = np.full((1,3),fill_value=-1)
final_m_ass = np.full((1,3),fill_value=-1)
final_3d_ass = np.full((3,3,3),fill_value=0)

final_l = -1

base_mtow ,base_jtom ,base_cost = three_dim_ass(c)
print('BASE COST :',base_cost, 'base_mtow ,base_jtom: ,',base_mtow ,base_jtom)
for j in range(3):  
  st = []
  for l in range(3):
    if l not in final_l_ass:
      print('#################################################################### job: ',j,' machine: ',l)
      auc_score,mtow,jtom = three_dim_rollout(c,l,j)
      st.append(auc_score)
      print('BASE AUC SCORE : ',base_cost)
      if auc_score <= base_cost:
        base_cost = auc_score
        base_mtow = mtow
        base_jtom = jtom
        

      print('############### end of ',j,'|',l,'st: ',st,'base_cost: ',base_cost)
      print('final l ass: ',final_l_ass,'base_mtow: ',base_mtow,'base_jtom: ',base_jtom)
  final_3d_ass[j,base_jtom[j],base_mtow[j]] = 1
  final_l_ass[0][j] = base_jtom[j]
  final_m_ass[0][j] = base_mtow[j]
  print('+++++++++++++++++++++++++++++++++')
  print('++++++++ END of STAGE {} ++++++++'.format(j+1))
  print('+++++++++++++++++++++++++++++++++')
  print(final_3d_ass)
  print('final_l_ass: ',final_l_ass,'final_m_ass: ',final_m_ass)
  print('+++++++++++++++++++++++++++++++++')
  print('+++++++++++++++++++++++++++++++++')
  print('+++++++++++++++++++++++++++++++++')
