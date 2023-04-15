# Rollout 
import numpy as np
from base_heuristic import three_dim_ass
from auction import auction

jl_arr = np.zeros((3,3))
fixed = 0
# def append_dict(d,x):
#   if x in d:

def three_dim_rollout(arr,fixed_l,fixed_j):
  lm_arr = [] #np.zeros((3-(fixed_j),3-(fixed_j)))
  # lm_dict = {}
  j_index = []
  for l in range(3): #for a,l in zip(range(lm_arr.shape[0]),range(3)):
    for m in range(3): #for b,m in zip(range(lm_arr.shape[0]),range(3)):
      # if m not in final_m_ass and l not in final_l_ass:
      if l == fixed_l:
        lm_arr.append(arr[fixed_j,fixed_l,m])
      elif l in final_l_ass:
        # print('j idx: ,',l,'|',final_l_ass,'|',np.where(final_l_ass[0] == l)[0])
        lm_arr.append(arr[np.where(final_l_ass[0] == l)[0][0],l,m])
      else:
        if fixed_j == 2:
          lm_arr.append(arr[fixed_j:,l,m].min())
        else:
          lm_arr.append(arr[fixed_j+1:,l,m].min())
  print('LMARR: ',lm_arr)
  mat_size = int(np.sqrt(len(lm_arr)))
  lm_arr = np.asarray(lm_arr).reshape((mat_size,mat_size))
  
  # row_ind, col_ind = linear_sum_assignment(lm_arr)
  # cost = lm_arr[row_ind, col_ind].sum()
  # bidder2item = col_ind
  bidder2item, cost = auction(lm_arr)
  auc_score = lm_arr[(np.arange(lm_arr.shape[0]), bidder2item)].sum()
  print(lm_arr,'l to worker',bidder2item,auc_score,cost)

  tmp_arr = []#np.zeros((3-(fixed_j+1),3-(fixed_j+1)))
  index_arr = []
  j_idx=[]

  if fixed_j != 2:
    fixed_j += 1
  # print('tmp_arr.shape[0]',tmp_arr.shape[0])
  for j in range(fixed_j,3): #for c,j in zip(range(tmp_arr.shape[0]),range(3)):
    for l in range(3): #for d,l in zip(range(tmp_arr.shape[0]),range(3)):
      if l != fixed_l and l not in final_l_ass:# and j in np.where(final_l_ass == -1)[0]:
        # print('jl',j,l,bidder2item[l])
        tmp_arr.append(arr[j,l,bidder2item[l]])
        index_arr.append(l)
        j_idx.append(j)
        print('tmp_arr: ',tmp_arr,'index_arr: ',index_arr)
   
  sq_root = int(np.sqrt(len(tmp_arr)))
  # if (sq_root*sq_root) == len(tmp_arr):
  tmp_arr = np.asarray(tmp_arr).reshape((sq_root,sq_root))
  index_arr = np.asarray(index_arr).reshape((sq_root,sq_root))
  # else:
  #   res = np.zeros((2,2))
  #   res[0,0] = tmp_arr[0]
  #   res[1,1] = tmp_arr[1]
  #   tmp_arr = res
  
  

  print('jobs to l machine')
  print(tmp_arr)

  # row_ind, col_ind = linear_sum_assignment(tmp_arr)
  # cost = tmp_arr[row_ind, col_ind].sum()
  # jobstomachine = col_ind
  # print(tmp_arr, jobstomachine)
  jobstomachine, cost = auction(tmp_arr)
  auc_score = tmp_arr[(np.arange(tmp_arr.shape[0]), jobstomachine)].sum()
  print(tmp_arr,jobstomachine,'jtom index: ',index_arr,auc_score,cost,'final l :',np.where(final_l_ass[0] != -1)[0])

  # cost2d = arr[0,jobstomachine[0],bidder2item[0]] + arr[1,jobstomachine[1],bidder2item[1]] + arr[2,jobstomachine[2],bidder2item[2]] 
  # s_cost = arr[fixed_j,fixed_l,bidder2item[fixed_l]] + arr[fixed_j+1,index_arr[0,jobstomachine[0]],bidder2item[index_arr[0,jobstomachine[0]]]] + arr[fixed_j+2,index_arr[1,jobstomachine[1]],bidder2item[index_arr[1,jobstomachine[1]]]]
  j2m = final_l_ass[0][np.where(final_l_ass[0] != -1)[0]]
  j2m = np.append(j2m,fixed_l)
  print('below app: ',fixed_l,j2m)
  for i in range(len(jobstomachine)):
    j2m = np.append(j2m,index_arr[i,jobstomachine[i]])
  # j2m = np.array([fixed_l,index_arr[0,jobstomachine[0]],index_arr[1,jobstomachine[1]]])
  print('J2M:    ',j2m)
  m2w = bidder2item[j2m] #np.array([bidder2item[fixed_l],bidder2item[index_arr[0,jobstomachine[0]]],bidder2item[index_arr[1,jobstomachine[1]]]])
  s_cost = arr[(np.arange(arr.shape[0]),j2m,m2w)].sum()
  print(s_cost,(np.arange(arr.shape[0]),j2m,m2w) )
  return s_cost, m2w,j2m#bidder2item, jobstomachine#auc_score #+ arr[]


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
      # best_cost = min(17,min(st))
      print('BASE AUC SCORE : ',base_cost)
      if auc_score <= base_cost:
        final_l = l
        base_cost = auc_score
        base_mtow = mtow
        base_jtom = jtom
        # final_l_ass[0][j] = l
        # final_m_ass[0][l] = base_mtow[base_jtom[j]]
        

      print('############### end of ',j,'|',l,'st: ',st,'base_cost: ',base_cost,'final_l: ',final_l)
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
