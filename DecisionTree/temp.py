import numpy as np
# from src.decision_tree import DecisionTree, Node

#     s = len(targets)#total no of samples
#     c= np.unique(targets)#the labels, could binary or continuos
#     m = np.median(c)
#     #print(m)
#     Hs = 0
#     Hsa = 0
#     freq = [0,0]#Stores the total yes or no
#     for i in range(len(targets)):
#         if targets[i] <= m:
#             freq[0] +=1
#         else:
#             freq[1] +=1
#     #print(freq)
#     for i in range(2):
#         p = freq[i] / s
#         #print(p)
#         H = 0
#         if p == 0:
#             H = 0
#         else:
#             H = -p * np.log2(p)
#         Hs += H
#     v, f= np.unique(features[:,attribute_index], return_counts=True) #the classes of this attribute
#     #print(f)
#     for i in range(len(v)):
#         w = f[i] / s #weight of this attribute class
#         t = np.where( features[:,attribute_index]== v[i])#the no. of targets that fall under this class of attribute
#         #print(t[0][2])
#         f1 = [0,0]#the yes and nos of targets of each class of attribute
#         for j in range(len(t[0])):
#             #print(i, t[0][j], targets[t[0][j]])
#             if np.all(targets[t[0][j]]) <= m:
#                 f1[0] +=1
#             else:
#                 f1[1] +=1
#         #print(f1)
#         H = 0
#         Hsk = 0
#         for k in range(2):
#             p = f1[k] / f[i]
#             #print(p)
#             if p == 0:
#                 H = 0
#             else:
#                 H = -p * np.log2(p)
#             Hsk += H
#         #print(Hsk)
#         Hsa += (w * Hsk)    
#     return (Hs - Hsa)

arr = [0.06221976 ,0.06221976 ,0.14653542 ,-1. ,0.21659478 , 0.00100612, 0.12712196]
x = [3]
x.extend(arr)
print(x)
