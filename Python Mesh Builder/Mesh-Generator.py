"""
2024/7/5
Yuhao, Y.Q.Gao's group, Colledge of Chemistry and Molecular Engineering
Using for build a Finite Element Mesh for a cubiod with equal potential surface
"""
import time
from scipy import sparse
from math import ceil
import pandas as pd
import numba as nb
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.linalg import lu,solve
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure


eps = 0.08854187817 # F/A

def waiting(i,N):
    print("\r", end="")
    print("进度: {}%: ".format(ceil(i/N*100)), "▓" * (ceil(i/N*100) // 2), end="")
    sys.stdout.flush()
    time.sleep(0.01)
#-------------------------------------------------------------------------------------------------------------------
# Define box size and cubioc mesh size of the FE mesh.
box_size = np.array([100.0,100.0,100.0]).astype("float64")
(NX,NY,NZ) = (50,50,50)
(dx,dy,dz) = (box_size[0]/NX, box_size[1]/NY, box_size[2]/NZ)

#-------------------------------------------------------------------------------------------------------------------
# 等势面生成算法，需要人为修改为合适形状，这里以球形等势面 1/r^-2 势能来模拟原子表面的等势能形状
# 在格子中内画个球, 球心(15.0, 15.0, 15.0), 半径 5.5
iso_level = 1.0
eq_center = np.array([50.0, 50.0, 50.0])
eq_radius = 16.7
field = np.zeros((NX+1,NY+1,NZ+1))

def sphere_eqfield_generator(field, eq_center, eq_radius):
    for i in range(NX+1):
        for j in range(NY+1):
            for k in range(NZ+1):
                mesh_crd = np.array([i*dx,j*dy,k*dz])
                if mesh_crd == eq_center:
                    field[i,j,k] = 1e10
                else:
                    r2 = (mesh_crd - eq_center) @ (mesh_crd - eq_center)
                    field[i,j,k] = eq_radius**2 / r2 

sphere_eqfield_generator(field, eq_center, eq_radius)
#------------------------------------------------------------------------------------------------------------------
# Marching-Cubes 算法给出切点的坐标，以及几何序号
cut_crd = measure.marching_cubes(field, iso_level,spacing=(dx, dy, dz))[0]
# 切割点的数目
cut_cnt = cut_crd.shape[0]
cut_sgn = [int(i+(NX)*(NY)*(NZ)) for i in range(cut_cnt)]

# 给出需要计算的节点的全局序号，规整格点在前，切割格点在后
sgn_convert_to_calc = {}
# 全局序号的总数
sgn_calc_cnt = 0

now_sgn = 0
for k in range(NZ):
     for j in range(NY):
          for i in range(NX):
               sgn = i + j*NX + k*NX*NY
               if field[i,j,k] < iso_level:
                    sgn_convert_to_calc[sgn] = now_sgn
                    now_sgn += 1
for i in range(cut_cnt):
     sgn_convert_to_calc[cut_sgn[i]] = now_sgn
     now_sgn += 1

calc_convert_to_sgn = {v: k for k, v in sgn_convert_to_calc.items()}
sgn_calc_cnt = len(sgn_convert_to_calc)
# 第一个切点的编号（或理解为需要计算的正交格点数目）
first_cut_sgn = sgn_calc_cnt-cut_cnt
#------------------------------------------------------------------------------------------------------------------

# 存储正交网格格点计算编号到坐标编号的映射
calc_need_convert_list = []
for k in range(NZ):
     for j in range(NY):
          for i in range(NX):
               sgn = i + j*NX + k*NX*NY
               if field[i,j,k] < iso_level:
                    calc_need_convert_list.append(int(sgn))
f_trans = open(r"./CSR-Matrix/calc_need_convert_list.txt", 'w')
for i in range(first_cut_sgn):
     f_trans.write(str(calc_need_convert_list[i]) + '\n')
f_trans.close()

# 存储正交网格从坐标编号到计算编号的映射，若无对应编号返回-1
now_sgn = 0
ortho_need_convert_list = []
for k in range(NZ):
     for j in range(NY):
          for i in range(NX):
               if field[i,j,k] < iso_level:
                    ortho_need_convert_list.append(now_sgn)
                    now_sgn += 1
               else:
                    ortho_need_convert_list.append(-1)
f_trans = open(r"./CSR-Matrix/ortho_need_convert_list.txt", 'w')
for i in range(NX*NY*NZ):
     f_trans.write(str(ortho_need_convert_list[i]) + '\n')
f_trans.close()
#------------------------------------------------------------------------------------------------------------------
