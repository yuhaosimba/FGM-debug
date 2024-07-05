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

print("Finite Mesh Generate Finished")
#------------------------------------------------------------------------------------------------------------------
# 网格信息生成，定义cubes类，每个cube存储四面体信息

class Cube:
    def __init__(self,i,j,k): # 0 <= i,j,k <= 99 
        self.sgn = i + j*NX + k*NX*NY
        self.crd = np.array([[i*dx,j*dy,k*dz],
                             [(i+1)*dx,j*dy,k*dz],
                             [(i+1)*dx,(j+1)*dy,k*dz],
                             [i*dx,(j+1)*dy,k*dz],
                             [i*dx,j*dy,(k+1)*dz],
                             [(i+1)*dx,j*dy,(k+1)*dz],
                             [(i+1)*dx,(j+1)*dy,(k+1)*dz],
                             [i*dx,(j+1)*dy,(k+1)*dz]])
        self.points_sgn = np.array([i + j*NX + k*NX*NY,
                              (i+1)%NX + j*NX + k*NX*NY,
                              (i+1)%NX + (j+1)%NY*NX + k*NX*NY,
                              i + (j+1)%NY*NX + k*NX*NY,
                              i + j*NX + (k+1)%NZ*NX*NY,
                              (i+1)%NX + j*NX + (k+1)%NZ*NX*NY,
                              (i+1)%NX + (j+1)%NY*NX + (k+1)%NZ*NX*NY,
                              i + (j+1)%NY*NX + (k+1)%NZ*NX*NY])
        self.cut_crd = []
        self.cut_sgn = []
        self.tetra = None
        # 划定顶点位置，判断是否需要计算，判断是否按照四面体元计算
        self.if_out = np.zeros(8)
        self.if_calc = True
        self.if_tetra = False
        cnt = 0
        for i in range(8):
            if sphere(self.crd[i]) < iso_level:
                self.if_out[i] = 1
            else:
                self.if_out[i] = 0
                cnt += 1
        if cnt == 8:
            self.if_calc = False
        if cnt >0:
            self.if_tetra = True
            
        # 引入切点,输入的是几何序号！
    def add_point(self,crd,sgn):
        self.cut_crd.append(crd)
        self.cut_sgn.append(sgn)
        
    def delaunay(self):
        if self.if_calc != True or self.if_tetra != True:
            return
        else:
            self.tetra_crd = []
            self.tetra_sgn = []
            for i in range(8):
                if self.if_out[i] ==1:
                    self.tetra_crd.append(self.crd[i])
                    self.tetra_sgn.append(self.points_sgn[i])
            for i in range(len(self.cut_sgn)):
                self.tetra_crd.append(self.cut_crd[i])
                self.tetra_sgn.append(self.cut_sgn[i])
            #print(self.tetra_crd)
            self.tetra = Delaunay(self.tetra_crd).simplices
            self.tetra_crd = np.array(self.tetra_crd)
            self.tetra_sgn = np.array(self.tetra_sgn)
            return
            
def ijk_to_sgn(i,j,k):
    (l,m,n) = (i%NX,j%NY,k%NZ)
    return int(l + m*NX + n*NX*NY)

# 初始化网格信息
cubes = []
for k in range(NZ):
    for j in range(NY):
        for i in range(NX):
            cubes.append(Cube(i,j,k))

# 加入节点
def add_cut_in_cubes(crd,sgn):
    (i,j,k) = (crd[0]//dx, crd[1]//dy, crd[2]//dz)
    (if_x, if_y, if_z) = (crd[0]%dx, crd[1]%dy, crd[2]%dz)
    
    if if_x != 0:
        cubes[ijk_to_sgn(i,j,k)].add_point(crd,sgn)
        cubes[ijk_to_sgn(i,j-1,k)].add_point(crd,sgn)
        cubes[ijk_to_sgn(i,j-1,k-1)].add_point(crd,sgn)
        cubes[ijk_to_sgn(i,j,k-1)].add_point(crd,sgn)
    elif if_y != 0:
        cubes[ijk_to_sgn(i,j,k)].add_point(crd,sgn)
        cubes[ijk_to_sgn(i-1,j,k)].add_point(crd,sgn)
        cubes[ijk_to_sgn(i-1,j,k-1)].add_point(crd,sgn)
        cubes[ijk_to_sgn(i,j,k-1)].add_point(crd,sgn)
    elif if_z != 0:
        cubes[ijk_to_sgn(i,j,k)].add_point(crd,sgn)
        cubes[ijk_to_sgn(i,j-1,k)].add_point(crd,sgn)
        cubes[ijk_to_sgn(i-1,j-1,k)].add_point(crd,sgn)
        cubes[ijk_to_sgn(i-1,j,k)].add_point(crd,sgn)
        
        
for i in range(cut_cnt):
    add_cut_in_cubes(cut_crd[i],cut_sgn[i])
    
# 生成四面体
# 初始化网格
for cube in cubes:
    cube.delaunay()

print("Cube Info Generate Finished")
#------------------------------------------------------------------------------------------------------------------
# 生成电势项求解矩阵
Phi = np.zeros(first_cut_sgn)
iso_surf_phi = 1
Solvation_Const = np.zeros(first_cut_sgn)
Solvation_Matrix = sparse.lil_matrix((first_cut_sgn,first_cut_sgn),dtype="float64")

# 三线性插值&体积插值填充求解矩阵
# generate Matrix from cubes which contains tetrahedron
def from_tetra_generate_deviation(i,j,k):
    cube = cubes[ijk_to_sgn(i,j,k)]
    tetras =  cube.tetra
    for tet in tetras:
        # 初始化四面体格点坐标crd[i]、电势序号sgn[i]
        crd = np.zeros((4,3))
        sgn = np.zeros(4).astype(int)
        for q in range(4):
            crd[q] = cube.tetra_crd[tet[q]]
            sgn[q] = int(sgn_convert_to_calc[cube.tetra_sgn[tet[q]]])
        
        # 初始化系数
        Ve = np.abs((1/6)*np.linalg.det([[1,crd[0,0],crd[0,1],crd[0,2]],
                                  [1,crd[1,0],crd[1,1],crd[1,2]],
                                  [1,crd[2,0],crd[2,1],crd[2,2]],
                                  [1,crd[3,0],crd[3,1],crd[3,2]]]))
        # DEBUG 4-24
        if Ve == 0:
            print("error! Ve = 0")
            print(i,j,k)
            print(crd)
            continue
        cst = eps/(72*Ve)
        
        a1 = -np.linalg.det([[1,crd[1,1],crd[1,2]], [1,crd[2,1],crd[2,2]], [1,crd[3,1],crd[3,2]]])
        a2 = np.linalg.det([[1,crd[0,1],crd[0,2]], [1,crd[2,1],crd[2,2]], [1,crd[3,1],crd[3,2]]])
        a3 = -np.linalg.det([[1,crd[0,1],crd[0,2]], [1,crd[1,1],crd[1,2]], [1,crd[3,1],crd[3,2]]])
        
        b1 = np.linalg.det([[1,crd[1,0],crd[1,2]], [1,crd[2,0],crd[2,2]], [1,crd[3,0],crd[3,2]]])
        b2 = -np.linalg.det([[1,crd[0,0],crd[0,2]], [1,crd[2,0],crd[2,2]], [1,crd[3,0],crd[3,2]]])
        b3 = np.linalg.det([[1,crd[0,0],crd[0,2]], [1,crd[1,0],crd[1,2]], [1,crd[3,0],crd[3,2]]])

        c1 = -np.linalg.det([[1,crd[1,0],crd[1,1]], [1,crd[2,0],crd[2,1]], [1,crd[3,0],crd[3,1]]])
        c2 = np.linalg.det([[1,crd[0,0],crd[0,1]], [1,crd[2,0],crd[2,1]], [1,crd[3,0],crd[3,1]]])
        c3 = -np.linalg.det([[1,crd[0,0],crd[0,1]], [1,crd[1,0],crd[1,1]], [1,crd[3,0],crd[3,1]]])
        
        if sgn[0] < first_cut_sgn: # 如果sgn[0] 不是切点，正常计算
            Solvation_Matrix[sgn[0],sgn[0]] += cst*(2*a1*a1+2*b1*b1+2*c1*c1)
            if sgn[1] <first_cut_sgn:
                Solvation_Matrix[sgn[0],sgn[1]] += cst*(2*a1*a2+2*b1*b2+2*c1*c2)
            else:
                Solvation_Const[sgn[0]] -= cst*(2*a1*a2+2*b1*b2+2*c1*c2)*iso_surf_phi
                
            if sgn[2] <first_cut_sgn:
                Solvation_Matrix[sgn[0],sgn[2]] += cst*(2*a1*a3+2*b1*b3+2*c1*c3)
            else:
                Solvation_Const[sgn[0]] -= cst*(2*a1*a3+2*b1*b3+2*c1*c3)*iso_surf_phi
                
            if sgn[3] <first_cut_sgn:
                Solvation_Matrix[sgn[0],sgn[3]] += cst*(-2*a1*(a1+a2+a3)-2*b1*(b1+b2+b3)-2*c1*(c1+c2+c3))
            else:
                Solvation_Const[sgn[0]] -= cst*(-2*a1*(a1+a2+a3)-2*b1*(b1+b2+b3)-2*c1*(c1+c2+c3))*iso_surf_phi
                
        if sgn[1] < first_cut_sgn:
            Solvation_Matrix[sgn[1],sgn[1]] += cst*(2*a2*a2+2*b2*b2+2*c2*c2)
            if sgn[0] <first_cut_sgn:
                Solvation_Matrix[sgn[1],sgn[0]] += cst*(2*a2*a1+2*b2*b1+2*c2*c1)
            else:
                Solvation_Const[sgn[1]] -= cst*(2*a2*a1+2*b2*b1+2*c2*c1)*iso_surf_phi
            
            if sgn[2] <first_cut_sgn:
                Solvation_Matrix[sgn[1],sgn[2]] += cst*(2*a2*a3+2*b2*b3+2*c2*c3)
            else:
                Solvation_Const[sgn[1]] -= cst*(2*a2*a3+2*b2*b3+2*c2*c3)*iso_surf_phi
            
            if sgn[3] <first_cut_sgn:
                Solvation_Matrix[sgn[1],sgn[3]] += cst*(-2*a2*(a1+a2+a3)-2*b2*(b1+b2+b3)-2*c2*(c1+c2+c3))
            else:
                Solvation_Const[sgn[1]] -= cst*(-2*a2*(a1+a2+a3)-2*b2*(b1+b2+b3)-2*c2*(c1+c2+c3))*iso_surf_phi
        
        if sgn[2] < first_cut_sgn:  
            Solvation_Matrix[sgn[2],sgn[2]] += cst*(2*a3*a3+2*b3*b3+2*c3*c3)
            if sgn[0] <first_cut_sgn:
                Solvation_Matrix[sgn[2],sgn[0]] += cst*(2*a3*a1+2*b3*b1+2*c3*c1)
            else:
                Solvation_Const[sgn[2]] -= cst*(2*a3*a1+2*b3*b1+2*c3*c1)*iso_surf_phi
            
            if sgn[1] <first_cut_sgn:
                Solvation_Matrix[sgn[2],sgn[1]] += cst*(2*a3*a2+2*b3*b2+2*c3*c2)
            else:
                Solvation_Const[sgn[2]] -= cst*(2*a3*a2+2*b3*b2+2*c3*c2)*iso_surf_phi
            
            if sgn[3] < first_cut_sgn:  
                Solvation_Matrix[sgn[2],sgn[3]] += cst*(-2*a3*(a1+a2+a3)-2*b3*(b1+b2+b3)-2*c3*(c1+c2+c3))
            else:
                Solvation_Const[sgn[2]] -= cst*(-2*a3*(a1+a2+a3)-2*b3*(b1+b2+b3)-2*c3*(c1+c2+c3))*iso_surf_phi
        
        
        if sgn[3] < first_cut_sgn:    
            Solvation_Matrix[sgn[3],sgn[3]] += cst*(2*(a1+a2+a3)**2+2*(b1+b2+b3)**2+2*(c1+c2+c3)**2)
            if sgn[0] <first_cut_sgn:
                Solvation_Matrix[sgn[3],sgn[0]] += cst*(-2*a1*(a1+a2+a3)-2*b1*(b1+b2+b3)-2*c1*(c1+c2+c3))
            else:
                Solvation_Const[sgn[3]] -= cst*(-2*a1*(a1+a2+a3)-2*b1*(b1+b2+b3)-2*c1*(c1+c2+c3))*iso_surf_phi
            
            if sgn[1] <first_cut_sgn:
                Solvation_Matrix[sgn[3],sgn[1]] += cst*(-2*a2*(a1+a2+a3)-2*b2*(b1+b2+b3)-2*c2*(c1+c2+c3))
            else:
                Solvation_Const[sgn[3]] -= cst*(-2*a2*(a1+a2+a3)-2*b2*(b1+b2+b3)-2*c2*(c1+c2+c3))*iso_surf_phi
                
            if sgn[2] <first_cut_sgn:    
                Solvation_Matrix[sgn[3],sgn[2]] += cst*(-2*a3*(a1+a2+a3)-2*b3*(b1+b2+b3)-2*c3*(c1+c2+c3))
            else:
                Solvation_Const[sgn[3]] -= cst*(-2*a3*(a1+a2+a3)-2*b3*(b1+b2+b3)-2*c3*(c1+c2+c3))*iso_surf_phi
           
# generate Matrix from pure cubes
# 从一个Cube中generate 偏微分贡献，加入Matrix中
# Debug 4-16 没有+= 
# DEBUG 检查对称性，从而判断序号是否正确

def from_cube_generate_deviation(i,j,k):
    phi_sgn = [sgn_convert_to_calc[i] for i in cubes[ijk_to_sgn(i,j,k)].points_sgn]
    (a,b,c) = (eps*dx*dy/(dz*9), eps*dx*dz/(dy*9), eps*dy*dz/(dx*9))
    Solvation_Matrix[phi_sgn[0],phi_sgn[0]] += a + b + c
    Solvation_Matrix[phi_sgn[0],phi_sgn[1]] += 0.5*a + 0.5*b - c
    Solvation_Matrix[phi_sgn[0],phi_sgn[2]] += 0.25*a - 0.5*b - 0.5*c
    Solvation_Matrix[phi_sgn[0],phi_sgn[3]] += 0.5*a - b + 0.5*c
    Solvation_Matrix[phi_sgn[0],phi_sgn[4]] += -a + 0.5*b + 0.5*c
    Solvation_Matrix[phi_sgn[0],phi_sgn[5]] += -0.5*a + 0.25*b - 0.5*c
    Solvation_Matrix[phi_sgn[0],phi_sgn[6]] += -0.25*a - 0.25*b - 0.25*c
    Solvation_Matrix[phi_sgn[0],phi_sgn[7]] += -0.5*a - 0.5*b + 0.25*c
    
    Solvation_Matrix[phi_sgn[1],phi_sgn[0]] += 0.5*a + 0.5*b - c
    Solvation_Matrix[phi_sgn[1],phi_sgn[1]] += a + b + c
    Solvation_Matrix[phi_sgn[1],phi_sgn[2]] += 0.5*a - b + 0.5*c
    Solvation_Matrix[phi_sgn[1],phi_sgn[3]] += 0.25*a - 0.5*b - 0.5*c
    Solvation_Matrix[phi_sgn[1],phi_sgn[4]] += -0.5*a + 0.25*b - 0.5*c
    Solvation_Matrix[phi_sgn[1],phi_sgn[5]] += -a + 0.5*b + 0.5*c
    Solvation_Matrix[phi_sgn[1],phi_sgn[6]] += -0.5*a - 0.5*b + 0.25*c
    Solvation_Matrix[phi_sgn[1],phi_sgn[7]] += -0.25*a - 0.25*b - 0.25*c
    
    Solvation_Matrix[phi_sgn[2],phi_sgn[0]] += 0.25*a - 0.5*b - 0.5*c
    Solvation_Matrix[phi_sgn[2],phi_sgn[1]] += 0.5*a - b + 0.5*c
    Solvation_Matrix[phi_sgn[2],phi_sgn[2]] += a + b + c
    Solvation_Matrix[phi_sgn[2],phi_sgn[3]] += 0.5*a + 0.5*b - c
    Solvation_Matrix[phi_sgn[2],phi_sgn[4]] += -0.25*a - 0.25*b - 0.25*c
    Solvation_Matrix[phi_sgn[2],phi_sgn[5]] += -0.5*a - 0.5*b + 0.25*c
    Solvation_Matrix[phi_sgn[2],phi_sgn[6]] += -a + 0.5*b + 0.5*c
    Solvation_Matrix[phi_sgn[2],phi_sgn[7]] += -0.5*a + 0.25*b - 0.5*c
    
    Solvation_Matrix[phi_sgn[3],phi_sgn[0]] += 0.5*a - b + 0.5*c
    Solvation_Matrix[phi_sgn[3],phi_sgn[1]] += 0.25*a - 0.5*b - 0.5*c
    Solvation_Matrix[phi_sgn[3],phi_sgn[2]] += 0.5*a + 0.5*b - c
    Solvation_Matrix[phi_sgn[3],phi_sgn[3]] += a + b + c
    Solvation_Matrix[phi_sgn[3],phi_sgn[4]] += -0.5*a - 0.5*b + 0.25*c
    Solvation_Matrix[phi_sgn[3],phi_sgn[5]] += -0.25*a - 0.25*b - 0.25*c
    Solvation_Matrix[phi_sgn[3],phi_sgn[6]] += -0.5*a + 0.25*b - 0.5*c
    Solvation_Matrix[phi_sgn[3],phi_sgn[7]] += -a + 0.5*b + 0.5*c
    
    Solvation_Matrix[phi_sgn[4],phi_sgn[0]] += -a + 0.5*b + 0.5*c
    Solvation_Matrix[phi_sgn[4],phi_sgn[1]] += -0.5*a + 0.25*b - 0.5*c
    Solvation_Matrix[phi_sgn[4],phi_sgn[2]] += -0.25*a - 0.25*b - 0.25*c
    Solvation_Matrix[phi_sgn[4],phi_sgn[3]] += -0.5*a - 0.5*b + 0.25*c
    Solvation_Matrix[phi_sgn[4],phi_sgn[4]] += a + b + c
    Solvation_Matrix[phi_sgn[4],phi_sgn[5]] += 0.5*a + 0.5*b - c
    Solvation_Matrix[phi_sgn[4],phi_sgn[6]] += 0.25*a - 0.5*b - 0.5*c
    Solvation_Matrix[phi_sgn[4],phi_sgn[7]] += 0.5*a - b + 0.5*c
    
    Solvation_Matrix[phi_sgn[5],phi_sgn[0]] += -0.5*a + 0.25*b - 0.5*c
    Solvation_Matrix[phi_sgn[5],phi_sgn[1]] += -a + 0.5*b + 0.5*c
    Solvation_Matrix[phi_sgn[5],phi_sgn[2]] += -0.5*a - 0.5*b + 0.25*c
    Solvation_Matrix[phi_sgn[5],phi_sgn[3]] += -0.25*a - 0.25*b - 0.25*c
    Solvation_Matrix[phi_sgn[5],phi_sgn[4]] += 0.5*a + 0.5*b - c
    Solvation_Matrix[phi_sgn[5],phi_sgn[5]] += a + b + c
    Solvation_Matrix[phi_sgn[5],phi_sgn[6]] += 0.5*a - b + 0.5*c
    Solvation_Matrix[phi_sgn[5],phi_sgn[7]] += 0.25*a - 0.5*b - 0.5*c
    
    Solvation_Matrix[phi_sgn[6],phi_sgn[0]] += -0.25*a - 0.25*b - 0.25*c
    Solvation_Matrix[phi_sgn[6],phi_sgn[1]] += -0.5*a - 0.5*b + 0.25*c
    Solvation_Matrix[phi_sgn[6],phi_sgn[2]] += -a + 0.5*b + 0.5*c
    Solvation_Matrix[phi_sgn[6],phi_sgn[3]] += -0.5*a + 0.25*b - 0.5*c
    Solvation_Matrix[phi_sgn[6],phi_sgn[4]] += 0.25*a - 0.5*b - 0.5*c
    Solvation_Matrix[phi_sgn[6],phi_sgn[5]] += 0.5*a - b + 0.5*c
    Solvation_Matrix[phi_sgn[6],phi_sgn[6]] += a + b + c
    Solvation_Matrix[phi_sgn[6],phi_sgn[7]] += 0.5*a + 0.5*b - c
    
    Solvation_Matrix[phi_sgn[7],phi_sgn[0]] += -0.5*a - 0.5*b + 0.25*c
    Solvation_Matrix[phi_sgn[7],phi_sgn[1]] += -0.25*a - 0.25*b - 0.25*c
    Solvation_Matrix[phi_sgn[7],phi_sgn[2]] += -0.5*a + 0.25*b - 0.5*c
    Solvation_Matrix[phi_sgn[7],phi_sgn[3]] += -a + 0.5*b + 0.5*c
    Solvation_Matrix[phi_sgn[7],phi_sgn[4]] += 0.5*a - b + 0.5*c
    Solvation_Matrix[phi_sgn[7],phi_sgn[5]] += 0.25*a - 0.5*b - 0.5*c
    Solvation_Matrix[phi_sgn[7],phi_sgn[6]] += 0.5*a + 0.5*b - c
    Solvation_Matrix[phi_sgn[7],phi_sgn[7]] += a + b + c

# 由电势信息构造 求解矩阵与净参数向量
for i in range(NX):
    for j in range(NY):
        for k in range(NZ):
            if cubes[ijk_to_sgn(i,j,k)].if_calc == True:
                if cubes[ijk_to_sgn(i,j,k)].if_tetra == True:
                    from_tetra_generate_deviation(i,j,k)
                else:
                    from_cube_generate_deviation(i,j,k)
print("Potential Calculator Matrices Generate Finished")
#------------------------------------------------------------------------------------------------------------------
# 输出文件
CSR_Matrix = Solvation_Matrix.tocsr()
# 存储稀疏矩阵信息
np.savetxt(r"./CSR-Matrix/data.txt",CSR_Matrix.data)

f_indices = open(r"./CSR-Matrix/indices.txt", 'w')
for i in range(len(CSR_Matrix.indices)):
    f_indices.write(str(int(CSR_Matrix.indices[i])))
    f_indices.write('\n')
f_indices.close()

f_indptr = open(r"./CSR-Matrix/indptr.txt", 'w')
for i in range(len(CSR_Matrix.indptr)):
    f_indptr.write(str(int(CSR_Matrix.indptr[i])))
    f_indptr.write('\n')
f_indptr.close()

np.save("Const.npy",Solvation_Const)
np.savetxt(r"./CSR-Matrix/Const.txt",Solvation_Const)

Phi = np.random.random(first_cut_sgn)
np.save("Phi.npy",Phi)
np.savetxt(r"./CSR-Matrix/phi.txt",Phi)

# 存储信息 dim, indptr, indices, data
f_info = open(r"./CSR-Matrix/info.txt", 'w')
f_info.write(str(CSR_Matrix.shape[0]) + '\n')
f_info.write(str(len(CSR_Matrix.data)) + '\n')
f_info.write(str(NX) + '\n')
f_info.write(str(NY) + '\n')
f_info.write(str(NZ) + '\n')
f_info.write(str(box_size[0]) + '\n')
f_info.write(str(box_size[1]) + '\n')
f_info.write(str(box_size[2]) + '\n')
f_info.write(str(first_cut_sgn) + '\n')
f_info.close()
