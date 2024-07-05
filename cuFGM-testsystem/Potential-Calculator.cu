#include "Potential-Calculator.cuh"

void Charge_Interpolation(){
    refresh_to_zero_potential << <(nx * ny * nz + 255) / 256, 256 >> > (d_charge_ortho, nx * ny * nz);
    charge_interpolation << <(n_charge + 255) / 256, 256 >> > (d_charge_crd, d_charge_discrete, nx, ny, nz, box_x, box_y, box_z, dx, dy, dz, n_charge, d_charge_ortho);
    trans_from_ortho_to_calc << <(nx * ny * nz + 255) / 256, 256 >> > (d_charge_ortho, d_charge, d_from_ortho_to_calc, nx, ny, nz);
    cublasDaxpy(bandle, A_num_rows, &one, d_const, 1, d_charge, 1);
    printf("charge-interpolation finished\n");
}

void Potential_file_saver() {
    cudaMemcpy(result_phi, d_phi, A_num_rows * sizeof(double), cudaMemcpyDeviceToHost);
    // clean result.txt
    std::ofstream f_clean("C:\\Users\\15653\\Desktop\\FGM-Python-Debug\\CSR-Matrix\\result.txt");
    // write result to txt
    std::ofstream f_result("C:\\Users\\15653\\Desktop\\FGM-Python-Debug\\CSR-Matrix\\result.txt");
    for (int i = 0; i < A_num_rows; i++) { f_result << result_phi[i] << std::endl; }
}

void Safe_cuda_free() {
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vec_phi);
    cusparseDestroyDnVec(vec_r);
    cusparseDestroyDnVec(vec_p);
    cusparseDestroyDnVec(vec_Ap);
    cublasDestroy(bandle);
    cusparseDestroy(handle);

    cudaFree(dBuffer);
    cudaFree(dA_csrOffsets); cudaFree(dA_columns); cudaFree(dA_values);
    cudaFree(d_phi); cudaFree(d_const);
    cudaFree(d_r); cudaFree(d_p); cudaFree(d_Ap);
    cudaFree(d_charge); cudaFree(d_charge_ortho);
    cudaFree(d_charge_crd); cudaFree(d_charge_discrete);
}



void CG_Solver(cusparseHandle_t handle, double minus, cusparseSpMatDescr_t matA, cusparseDnVecDescr_t vec_phi,
    double zero, cusparseDnVecDescr_t vec_r, cublasHandle_t bandle, int A_num_rows, double one, double* d_const, double* d_r,
    double* d_p, double err, double r_r, cusparseDnVecDescr_t vec_p, cusparseDnVecDescr_t vec_Ap, double* d_Ap, double p_A_p,
    double alpha, double* d_phi, double r_r_next, double beta, void* dBuffer) {

    // Initialize r = b - Ax
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minus, matA, vec_phi, &zero, vec_r, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
    cublasDaxpy(bandle, A_num_rows, &one, d_const, 1, d_r, 1);
    cudaMemcpy(d_p, d_r, A_num_rows * sizeof(double), cudaMemcpyDeviceToDevice);

    // Initialize err
    cublasDnrm2(bandle, A_num_rows, d_r, 1, &err);

    // CG-iteration
    while (err >= 1e-5) {
        // alpha = (r, r) / (Ap, p)
        cublasDdot(bandle, A_num_rows, d_r, 1, d_r, 1, &r_r);
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vec_p, &zero, vec_Ap, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
        cublasDdot(bandle, A_num_rows, d_Ap, 1, d_p, 1, &p_A_p);
        alpha = r_r / p_A_p;

        // phi = phi + alpha * p;  r = r - alpha * Ap
        cublasDaxpy(bandle, A_num_rows, &alpha, d_p, 1, d_phi, 1);
        alpha = -alpha;
        cublasDaxpy(bandle, A_num_rows, &alpha, d_Ap, 1, d_r, 1);

        // calc new (r,r) ;  beta = (r,r)_new / (r,r)
        cublasDdot(bandle, A_num_rows, d_r, 1, d_r, 1, &r_r_next);
        beta = r_r_next / r_r;

        // calc new p = r + beta * p
        cublasDscal(bandle, A_num_rows, &beta, d_p, 1);
        cublasDaxpy(bandle, A_num_rows, &one, d_r, 1, d_p, 1);

        // calc err
        cublasDnrm2(bandle, A_num_rows, d_r, 1, &err);
        printf("err = %0.9f\n", err);
    }
}



// Debug 假设 Crd 以 线性存储，第 i 个 电荷的 x 坐标为 crd[3*i], y 坐标为 crd[3*i+1], z 坐标为 crd[3*i+2]
__global__ void charge_interpolation(double* crd, double* charge_q, int nx, int ny, int nz, double box_x, double box_y, double box_z,
    double dx, double dy, double dz, int n, double* d_charge_ortho) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_x, idx_y, idx_z; // 电荷所在的正交网格坐标
    int idx_x_plus, idx_y_plus, idx_z_plus; // 电荷所在的正交网格坐标的下一个,符合3d周期性边界条件
    double xx, yy, zz; // 电荷在该正交网格中的归一化位置
    if (i < n) {
        idx_x = (crd[3 * i] / dx);
        idx_y = (crd[3 * i + 1] / dy);
        idx_z = (crd[3 * i + 2] / dz);
        xx = (crd[3 * i] - idx_x * dx) / dx;
        yy = (crd[3 * i + 1] - idx_y * dy) / dy;
        zz = (crd[3 * i + 2] - idx_z * dz) / dz;
        idx_x_plus = (idx_x + 1) % nx;
        idx_y_plus = (idx_y + 1) % ny;
        idx_z_plus = (idx_z + 1) % nz;
        double q = charge_q[i];
        // 伪逆向三线性插值
        d_charge_ortho[idx_x + idx_y * nx + idx_z * nx * ny] += (1 - xx) * (1 - yy) * (1 - zz) * q;
        d_charge_ortho[idx_x_plus + idx_y * nx + idx_z * nx * ny] += xx * (1 - yy) * (1 - zz) * q;
        d_charge_ortho[idx_x + idx_y_plus * nx + idx_z * nx * ny] += (1 - xx) * yy * (1 - zz) * q;
        d_charge_ortho[idx_x_plus + idx_y_plus * nx + idx_z * nx * ny] += xx * yy * (1 - zz) * q;
        d_charge_ortho[idx_x + idx_y * nx + idx_z_plus * nx * ny] += (1 - xx) * (1 - yy) * zz * q;
        d_charge_ortho[idx_x_plus + idx_y * nx + idx_z_plus * nx * ny] += xx * (1 - yy) * zz * q;
        d_charge_ortho[idx_x + idx_y_plus * nx + idx_z_plus * nx * ny] += (1 - xx) * yy * zz * q;
        d_charge_ortho[idx_x_plus + idx_y_plus * nx + idx_z_plus * nx * ny] += xx * yy * zz * q;
    }
    return;
}

__global__ void refresh_to_zero_potential(double* d_phi_ortho, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        d_phi_ortho[i] = 0;
    }
}

__global__ void trans_from_ortho_to_calc(double* d_phi_ortho, double* d_phi, int* d_from_ortho_to_calc, int nx, int ny, int nz) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nx * ny * nz) {
        if (d_from_ortho_to_calc[i] != -1) {
            d_phi[d_from_ortho_to_calc[i]] = d_phi_ortho[i];
        }
    }
}

__global__ void trans_from_calc_to_ortho(double* d_phi, double* d_phi_ortho, int* d_from_calc_to_ortho, int first_cut_sgn) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < first_cut_sgn) {
        d_phi_ortho[d_from_calc_to_ortho[i]] = d_phi[i];
    }
}

void Mesh_initialize() {
    std::ifstream f_info("C:\\Users\\15653\\Desktop\\FGM-Python-Debug\\CSR-Matrix\\info.txt");
    f_info >> A_num_rows >> A_nnz >> nx >> ny >> nz >> box_x >> box_y >> box_z >> first_cut_sgn;
    f_info.close();
    dx = box_x / nx;
    dy = box_y / ny;
    dz = box_z / nz;

    // 有限元网格、初始电势、基础电荷常数向量
    hA_csrOffsets = (int*)malloc((A_num_rows + 1) * sizeof(int));
    hA_columns = (int*)malloc(A_nnz * sizeof(int));
    hA_values = (double*)malloc(A_nnz * sizeof(double));
    h_phi = (double*)malloc(A_num_rows * sizeof(double));
    h_const = (double*)malloc(A_num_rows * sizeof(double));

    // 从计算网格到正交网格的映射
    h_from_calc_to_ortho = (int*)malloc(first_cut_sgn * sizeof(int));
    h_from_ortho_to_calc = (int*)malloc(nx * ny * nz * sizeof(int));

    // 正交网格变换矩阵与正交网格电势
    h_phi_ortho = (double*)malloc(nx * ny * nz * sizeof(double));   // 位置在(i,j,k)上的网格，序号为 i + j*NX + k*NX*NY




    // 计算结果输出
    result_phi = (double*)malloc(nx * ny * nz * sizeof(double));

    // Read parameters to host
    std::ifstream f_indptr("C:\\Users\\15653\\Desktop\\FGM-Python-Debug\\CSR-Matrix\\indptr.txt");
    for (int i = 0; i < A_num_rows + 1; i++) { f_indptr >> hA_csrOffsets[i]; }
    f_indptr.close();
    std::ifstream f_indices("C:\\Users\\15653\\Desktop\\FGM-Python-Debug\\CSR-Matrix\\indices.txt");
    for (int i = 0; i < A_nnz; i++) { f_indices >> hA_columns[i]; }
    f_indices.close();
    std::ifstream f_data("C:\\Users\\15653\\Desktop\\FGM-Python-Debug\\CSR-Matrix\\data.txt");
    for (int i = 0; i < A_nnz; i++) { f_data >> hA_values[i]; }
    f_data.close();

    std::ifstream f_phi("C:\\Users\\15653\\Desktop\\FGM-Python-Debug\\CSR-Matrix\\phi.txt");
    for (int i = 0; i < A_num_rows; i++) { f_phi >> h_phi[i]; }
    f_phi.close();
    std::ifstream f_charge("C:\\Users\\15653\\Desktop\\FGM-Python-Debug\\CSR-Matrix\\Const.txt");
    for (int i = 0; i < A_num_rows; i++) { f_charge >> h_const[i]; }
    f_charge.close();

    std::ifstream f_trans("C:\\Users\\15653\\Desktop\\FGM-Python-Debug\\CSR-Matrix\\calc_need_convert_list.txt");
    for (int i = 0; i < first_cut_sgn; i++) { f_trans >> h_from_calc_to_ortho[i]; }
    f_trans.close();

    std::ifstream f_trans2("C:\\Users\\15653\\Desktop\\FGM-Python-Debug\\CSR-Matrix\\ortho_need_convert_list.txt");
    for (int i = 0; i < nx * ny * nz; i++) { f_trans2 >> h_from_ortho_to_calc[i]; }
    f_trans2.close();


    cudaMalloc((void**)&dA_csrOffsets, (A_num_rows + 1) * sizeof(int));
    cudaMalloc((void**)&dA_columns, A_nnz * sizeof(int));
    cudaMalloc((void**)&dA_values, A_nnz * sizeof(double));
    cudaMalloc((void**)&d_phi, A_num_rows * sizeof(double));
    cudaMalloc((void**)&d_const, A_num_rows * sizeof(double));
    cudaMalloc((void**)&d_phi_ortho, nx * ny * nz * sizeof(double));  // 正交网格电势


    cudaMalloc((void**)&d_charge, A_num_rows * sizeof(double)); // 网格电荷
    cudaMalloc((void**)&d_charge_ortho, (nx * ny * nz) * sizeof(double)); // 正交网格电荷

    cudaMalloc((void**)&d_r, A_num_rows * sizeof(double));
    cudaMalloc((void**)&d_p, A_num_rows * sizeof(double));
    cudaMalloc((void**)&d_Ap, A_num_rows * sizeof(double));

    cudaMalloc((void**)&d_charge_crd, 3 * n_charge * sizeof(double));
    cudaMalloc((void**)&d_charge_discrete, n_charge * sizeof(double));

    cudaMalloc((void**)&d_from_calc_to_ortho, first_cut_sgn * sizeof(int));
    cudaMalloc((void**)&d_from_ortho_to_calc, nx * ny * nz * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(dA_csrOffsets, hA_csrOffsets, (A_num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi, h_phi, A_num_rows * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_const, h_const, A_num_rows * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_charge_crd, h_charge_crd, n_charge * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_charge_discrete, h_charge_discrete, n_charge * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_from_calc_to_ortho, h_from_calc_to_ortho, first_cut_sgn * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_from_ortho_to_calc, h_from_ortho_to_calc, nx * ny * nz * sizeof(int), cudaMemcpyHostToDevice);

    finish_t = clock();
    total_t = (double)(finish_t - start_t) / CLOCKS_PER_SEC;


    // CUSPARSE & CUBLAS APIs 
    cublasCreate_v2(&bandle);
    cusparseCreate(&handle);

    // Create matrix and vector descriptors
    cusparseCreateCsr(&matA, A_num_rows, A_num_rows, A_nnz, dA_csrOffsets, dA_columns, dA_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateDnVec(&vec_phi, A_num_rows, d_phi, CUDA_R_64F);
    cusparseCreateDnVec(&vec_phi_ortho, nx * ny * nz, d_phi_ortho, CUDA_R_64F);
    cusparseCreateDnVec(&vec_r, A_num_rows, d_r, CUDA_R_64F);
    cusparseCreateDnVec(&vec_p, A_num_rows, d_p, CUDA_R_64F);
    cusparseCreateDnVec(&vec_Ap, A_num_rows, d_Ap, CUDA_R_64F);
    cusparseCreateDnVec(&vec_charge, A_num_rows, d_charge, CUDA_R_64F);
    cusparseCreateDnVec(&vec_charge_ortho, nx * ny * nz, d_charge_ortho, CUDA_R_64F);
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minus, matA, vec_phi, &zero, vec_r, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    printf("Initialization & memory copy host -> device \n");
    printf("Time used = %f\n", total_t);
}