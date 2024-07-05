#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h> 
#include <fstream>
#include <cublas_v2.h>
#include<time.h>
#include <math.h>

static clock_t start_t, finish_t;
static double total_t;


// Host problem definition
static int A_num_rows;  // dim of dense matrix
static int A_nnz;       // non-zero 
static int* hA_csrOffsets;
static int* hA_columns;
static double* hA_values;
static double* h_phi;  // Phi
static double* h_const;  // Charge
static double     alpha = 0.0;
static double     beta = 0.0;
static double     r_r = 0.0;
static double     r_r_next = 0.0;
static double     p_A_p = 0.0;


static double     err = 0.0;

static const double     one = 1.0;
static const double     zero = 0.0;
static const double     minus = -1.0;


// Device memory management 
static int* dA_csrOffsets, * dA_columns;
static double* dA_values, * d_const;
// Transition matrix from calc to NX NY NZ
static int* dA_trans_csrOffsets, * dA_trans_columns;
static int* d_from_calc_to_ortho;
static int* d_from_ortho_to_calc;
static double* dA_trans_values;
static double* d_phi;
static double* d_r;
static double* d_p, * d_Ap;

static cusparseSpMatDescr_t matA;
static cusparseDnVecDescr_t vec_phi, vec_r, vec_p, vec_Ap;


static cusparseSpMatDescr_t mat_Trans;
static cusparseDnVecDescr_t vec_phi_ortho;
static cusparseDnVecDescr_t vec_charge, vec_charge_ortho;

static void* dBuffer = NULL;
static void* dBuffer2 = NULL;
static size_t bufferSize = 0;
static size_t bufferSize2 = 0;

// Phi-mesh transition
static int nx; static  int ny; static  int nz;
static int first_cut_sgn;
static double box_x; static double box_y; static double box_z;
static double dx, dy, dz;
static double* h_phi_ortho;    // data-shape: x*(NY*NZ) + y*NZ + z
static double* d_phi_ortho;

static double* d_charge, * d_charge_ortho;
static double* d_charge_crd;
static double* d_charge_discrete;

static int* h_from_calc_to_ortho;
static int* h_from_ortho_to_calc;

// Debug
static const int n_charge = 1;
static double* h_charge_discrete;
static double* h_charge_crd;

static double* result_phi;

static cublasHandle_t bandle = NULL;
static cusparseHandle_t handle = NULL;

static void CG_Solver(cusparseHandle_t handle, double minus, cusparseSpMatDescr_t matA, cusparseDnVecDescr_t vec_phi,
    double zero, cusparseDnVecDescr_t vec_r, cublasHandle_t bandle, int A_num_rows, double one, double* d_const, double* d_r,
    double* d_p, double err, double r_r, cusparseDnVecDescr_t vec_p, cusparseDnVecDescr_t vec_Ap, double* d_Ap, double p_A_p,
    double alpha, double* d_phi, double r_r_next, double beta, void* dBuffer);

__global__ static  void charge_interpolation(double* crd, double* charge_q, int nx, int ny, int nz, double box_x, double box_y, double box_z,
    double dx, double dy, double dz, int n, double* d_charge_ortho);

__global__ static void refresh_to_zero_potential(double* d_phi_ortho, int n);

__global__ static void trans_from_ortho_to_calc(double* d_phi_ortho, double* d_phi, int* d_from_ortho_to_calc, int nx, int ny, int nz);

__global__ static void trans_from_calc_to_ortho(double* d_phi, double* d_phi_ortho, int* d_from_calc_to_ortho, int first_cut_sgn);

static void Mesh_initialize();
static void Charge_Interpolation();
static void Potential_file_saver();
static void Safe_cuda_free();