#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h> 
#include <fstream>
#include <cublas_v2.h>
#include<time.h>
#include <math.h>

// 500个格点的单位球面，给定坐标，边标号，面标号，初始边长
static const int N_points = 500;
static int N_edges, N_faces;
//位置、初始边长、加速度、速度
static double* h_points, * h_edges_length_unit, * d_points, * d_edges_length_unit, * d_Acc, * d_Vel;
static double* result;
static int* h_edges, * h_faces, * d_edges, * d_faces;


// Debug: Set parameters
// VDW_boundary 是等势面半径，R 是采样球面平衡半径
static double k1 = 5000., VDW_boundary = 23., Attach_force = 1000.,
k2 = 5000., k3 = 50000., k4 = 10000., k5 = 15000., R = 9., dt = 0.0005;
// 摩擦系数
static double gamma = 0.1;
static const int max_iter = 20;
static double Err = 0.;


// Debug 
static double h_boundary_atom_center[3] = { 50., 50., 50. };
static double* d_boundary_atom_center;
static double* d_charge_center;
static double* h_charge_center;

// 以下是函数声明

// 四种球面应力的函数
__global__ static void boundary_force_shpere(double* points, double* boundary_atom_center,
	const double k1, const double VDW_boundary, const double Attach_force, double* Acc);

__global__ static void triangle_force(double* points, int* faces, const double k2, double* Acc, int N_faces);


__global__ static void edge_force(double* points, int* edges, double* edge_length_init,
	const double k3, const double k4, double* Acc, int N_edges);

__global__ static void center_force(double* points, double* charge_center,
	const double k5, const double R, double* Acc);

// 更新位置，传统派更新不引入速度，摩擦系数无限大
__global__ static void update_position(double* points, double* Acc, const double dt);

// 更新位置，方式采用Leap-Frog算法
__global__ static void update_position_Leapfrog(double* points, double* Vel, const double dt);

// 更新速度，方式采用Leap-Frog算法
__global__ static void update_velocity_Leapfrog(double* Vel, double* Acc, const double dt);

// 以下是向量化函数
__global__ static void refresh_to_zero(double* d_array, int n);

__global__ static void multiply_by_constant(double* d_array, int n, double cst);

__global__ static void add_by_constant(double* d_array, int n, double cst);

__global__ static void add_by_vector(double* d_array, int N_points, double dx, double dy, double dz);

// 初始化球面信息
static void Read_and_Initialize_sphere_info();
// 存储平衡球面位置
static void Write_sphere_info();
// 球面更新函数
static void Sphere_Autoconfort_Iterator();

