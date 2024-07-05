#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h> 
#include <fstream>
#include <cublas_v2.h>
#include<time.h>
#include <math.h>

// 500�����ĵ�λ���棬�������꣬�߱�ţ����ţ���ʼ�߳�
static const int N_points = 500;
static int N_edges, N_faces;
//λ�á���ʼ�߳������ٶȡ��ٶ�
static double* h_points, * h_edges_length_unit, * d_points, * d_edges_length_unit, * d_Acc, * d_Vel;
static double* result;
static int* h_edges, * h_faces, * d_edges, * d_faces;


// Debug: Set parameters
// VDW_boundary �ǵ�����뾶��R �ǲ�������ƽ��뾶
static double k1 = 5000., VDW_boundary = 23., Attach_force = 1000.,
k2 = 5000., k3 = 50000., k4 = 10000., k5 = 15000., R = 9., dt = 0.0005;
// Ħ��ϵ��
static double gamma = 0.1;
static const int max_iter = 20;
static double Err = 0.;


// Debug 
static double h_boundary_atom_center[3] = { 50., 50., 50. };
static double* d_boundary_atom_center;
static double* d_charge_center;
static double* h_charge_center;

// �����Ǻ�������

// ��������Ӧ���ĺ���
__global__ static void boundary_force_shpere(double* points, double* boundary_atom_center,
	const double k1, const double VDW_boundary, const double Attach_force, double* Acc);

__global__ static void triangle_force(double* points, int* faces, const double k2, double* Acc, int N_faces);


__global__ static void edge_force(double* points, int* edges, double* edge_length_init,
	const double k3, const double k4, double* Acc, int N_edges);

__global__ static void center_force(double* points, double* charge_center,
	const double k5, const double R, double* Acc);

// ����λ�ã���ͳ�ɸ��²������ٶȣ�Ħ��ϵ�����޴�
__global__ static void update_position(double* points, double* Acc, const double dt);

// ����λ�ã���ʽ����Leap-Frog�㷨
__global__ static void update_position_Leapfrog(double* points, double* Vel, const double dt);

// �����ٶȣ���ʽ����Leap-Frog�㷨
__global__ static void update_velocity_Leapfrog(double* Vel, double* Acc, const double dt);

// ����������������
__global__ static void refresh_to_zero(double* d_array, int n);

__global__ static void multiply_by_constant(double* d_array, int n, double cst);

__global__ static void add_by_constant(double* d_array, int n, double cst);

__global__ static void add_by_vector(double* d_array, int N_points, double dx, double dy, double dz);

// ��ʼ��������Ϣ
static void Read_and_Initialize_sphere_info();
// �洢ƽ������λ��
static void Write_sphere_info();
// ������º���
static void Sphere_Autoconfort_Iterator();

