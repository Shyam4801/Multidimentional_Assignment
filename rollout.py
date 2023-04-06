# Rollout 
import numpy as np


jl_arr = np.zeros((3,3))
fixed = 0
# def append_dict(d,x):
#   if x in d:

def three_dim_rollout(arr,fixed_l,fixed_j):
  lm_arr = []    # to store input lm cost to auction()
  m_index = []    # To store the corresponding indices coz after stage 1 we need to not use the assigned worker of stage 1
  l_idx = {}
  
  # Iterate over machine and worker 
  for l in range(3): 
    lcount = 0
    for m in range(3): 
      if m not in final_m_ass and l not in final_l_ass:     # make sure the assigned worker and machine of previous stages are not taken into account
        l_idx[l] = lcount
        if l == fixed_l:   # make sure job is fixed 
          lm_arr.append(arr[fixed_j,fixed_l,m])
          
        else:   # take min among the unassigned jobs 
          if fixed_j == 2:  # used to set the range of jobs among which min is calculated 
            lm_arr.append(arr[fixed_j:,l,m].min())  # if its the last job then use that alone 
          else:
            lm_arr.append(arr[fixed_j+1:,l,m].min()) # else use the job index excluding the fixed one
        
        # print('inside lm loop: ',l_idx)
        m_index.append(m)  # to store indices of worker coz after stage 1 we need to not use the assigned worker of stage 1
        lcount+=1
  mat_size = int(np.sqrt(len(lm_arr)))      # choose size based on len of list
  lm_arr = np.asarray(lm_arr).reshape((mat_size,mat_size))    # convert to array
  m_index = np.asarray(m_index).reshape((mat_size,mat_size))
      
  bidder2item, cost = auction(lm_arr)         # input lm mat to auction()
  auc_score = lm_arr[(np.arange(lm_arr.shape[0]), bidder2item)].sum()
  print(lm_arr,'l to worker',bidder2item,'m_index: ',m_index,auc_score,cost)

  tmp_arr = []      # to store the jl cost to input to auction()
  index_arr = [] # to store the respective indices 
  j_idx=[]        
  
  if fixed_j != 2:    # To make sure we start from the unassigned jobs , if its not the last job start from the next coz previous is assigned in previous stages
    fixed_j+=1
  
  for j in range(fixed_j,3):          # To start from unassigned jobs 
    count=0
    for l in range(3):                   
      # Make sure that the l is not already assigned and not the fixed one 
      if l == fixed_l:
        count+=1
      if l != fixed_l and l not in final_l_ass:# and j in np.where(final_l_ass == -1)[0]:
        print('j,l,m_index[l,bidder2item[l]]]: ',j,l)#,m_index[l,bidder2item[l]])
        # act_l = np.where(l_idx==l)
        print('act_l: ',count,l)
        tmp_arr.append(arr[j,l,m_index[count,bidder2item[count]]])         # store the jl cost to input to auction()
        index_arr.append(l)                           # store their indices 
        j_idx.append(j)
        count +=1 
   
  # if len(tmp_arr) != 1:
  # store and reshape the array based on len of jl cost matrix 
  mat_size = int(np.sqrt(len(tmp_arr)))
  tmp_arr = np.asarray(tmp_arr).reshape((mat_size,mat_size))
  index_arr = np.asarray(index_arr).reshape((mat_size,mat_size))

  # else:
  #   tmp_arr = np.asarray([tmp_arr])
  #   index_arr = np.asarray([index_arr])
  print('jobs to l machine')
  # print('tmp arr: ',tmp_arr,index_arr)

  jobstomachine, cost = auction(tmp_arr)
  auc_score = tmp_arr[(np.arange(tmp_arr.shape[0]), jobstomachine)].sum()
  print('j2m array: ',tmp_arr,'j2m ass: ',jobstomachine,'jtom index: ',index_arr,'score: ',auc_score,cost)

  # cost2d = arr[0,jobstomachine[0],bidder2item[0]] + arr[1,jobstomachine[1],bidder2item[1]] + arr[2,jobstomachine[2],bidder2item[2]] 
  # s_cost = arr[fixed_j,fixed_l,bidder2item[fixed_l]] + arr[fixed_j+1,index_arr[0,jobstomachine[0]],bidder2item[index_arr[0,jobstomachine[0]]]] + arr[fixed_j+2,index_arr[1,jobstomachine[1]],bidder2item[index_arr[1,jobstomachine[1]]]]
  j2m=[]        # to store assigned j2m indices 
  m2w=final_m_ass.copy()        # to store assigned m2w indices 
  for i in range(len(final_l_ass[0])):  # to use the assigned cost from previous stages in calculating the current stage cost so adding it to the list of indices
    if final_l_ass[0][i] != -1:
      j2m.append(final_l_ass[0][i])
  m2w[0,final_l_ass[0][i]] = final_m_ass[0][final_l_ass[0][i]]
  j2m.append(fixed_l)             # adding the fixed index of l
  m2w[0,fixed_l] = bidder2item[fixed_l] # adding the assigned worker of fixed l machine
  if len(j2m) == 3:
    s_cost = arr[(np.arange(arr.shape[0]),j2m,m2w)].sum()
    return s_cost, m2w,j2m
  print('m2w after updating the final_m_Ass copy with fixed_ls worker: ',m2w,'j2m: ',j2m)
  for i in range(3-(fixed_j)):        # appending the current stage assigned indices of l and worker
    j2m.append(index_arr[i,jobstomachine[i]])
    m2w[0,index_arr[i,jobstomachine[i]]] = bidder2item[l_idx[index_arr[i,jobstomachine[i]]]]
  j2m = np.asarray(j2m)
  m2w = np.asarray(m2w)
  # j2m = final_l_ass.copy()
  # m2w = final_m_ass.copy()
  # j2m.insert(fixed_j,fixed_l)
  # j2m.insert(fixed_j,)
  print('j2w : ',j2m,'m2w: ',m2w[0])
  s_cost = arr[(np.arange(arr.shape[0]),j2m,m2w)].sum()       # computing the cost of the assignments aka total cost of the groupings (S1,S2.. Sn)
  print('s_cost,(np.arange(arr.shape[0]),j2m,m2w) : ',s_cost,j2m,m2w[0] )
  return s_cost, m2w[0],j2m#bidder2item, jobstomachine#auc_score #+ arr[]


c = np.array([[5,1,2],[5,5,6],[5,5,7],[2,6,8],[9,2,9],[0,3,3],[4,1,9],[4,9,8],[1,3,1]]).reshape(3,3,3)

# To store the final assignments of each stage after comparing with the base heuristic cost 
final_l_ass = np.full((1,3),fill_value=-1)
final_m_ass = np.full((1,3),fill_value=-1)
final_3d_ass = np.full((3,3,3),fill_value=0)

final_l = -1
# Base heuristic on 3D 
base_mtow ,base_jtom ,base_cost = three_dim_ass(c)
# to loop over the fixed set of assignments of job to machine 
for j in range(3):  
  st = []
  for l in range(3):
    if l not in final_l_ass: # Make sure the machine is not already assigned
      print('#################################################################### job: ',j,' machine: ',l)
      auc_score,mtow,jtom = three_dim_rollout(c,l,j)        # get the total cost of the groupings for each fixed assignment 
      st.append(auc_score)
      print('BASE AUC SCORE : ',base_cost)
      if auc_score <= base_cost:                # compare it to base and update the base heuristic trajectory
        final_l = l
        base_cost = auc_score
        base_mtow = mtow
        base_jtom = jtom
        # final_l_ass[0][j] = l
        # final_m_ass[0][l] = base_mtow[base_jtom[j]]
        

      print('############### end of ',j,'|',l,'st: ',st,'base_cost: ',base_cost,'final_l: ',final_l)
      print('final l ass: ',final_l_ass,'base_mtow: ',base_mtow,'base_jtom: ',base_jtom)
  final_3d_ass[j,base_jtom[j],base_mtow[base_jtom[j]]] = 1
  final_l_ass[0][j] = base_jtom[j]
  final_m_ass[0][base_jtom[j]] = base_mtow[base_jtom[j]]
  print('+++++++++++++++++++++++++++++++++')
  print('+++++++++++++++++++++++++++++++++')
  print('+++++++++++++++++++++++++++++++++')
  print(final_3d_ass)
  print('final_l_ass: ',final_l_ass,'final_m_ass: ',final_m_ass)
  print('+++++++++++++++++++++++++++++++++')
  print('+++++++++++++++++++++++++++++++++')
  print('+++++++++++++++++++++++++++++++++')
