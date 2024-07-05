#include "Green-Func.cu"
#include "Potential-Calculator.cu"
#include "Electric_force_calculator.cu"



int main()
{
	// 电荷坐标存储  DEBUG 4-9
	h_charge_discrete = (double*)malloc(n_charge * sizeof(double));
	h_charge_crd = (double*)malloc(n_charge * 3 * sizeof(double));
	// Debug 4-9 手动输入电荷初始坐标
	h_charge_discrete[0] = 1.0;
	h_charge_crd[0] = 75.;
	h_charge_crd[1] = 50.; 
	h_charge_crd[2] = 50.;

	Mesh_initialize();
	Charge_Interpolation();
	CG_Solver(handle, minus, matA, vec_phi, zero, vec_r, bandle, A_num_rows, one, d_charge, d_r,
		d_p, err, r_r, vec_p, vec_Ap, d_Ap, p_A_p, alpha, d_phi, r_r_next, beta, dBuffer);
	printf("CG-iteration finished\n");
	Potential_file_saver();

	// Form auto-conformed shpere
	Read_and_Initialize_sphere_info();

	// Debug: Set a charge at (75,50,50)
	h_charge_center = (double*)malloc(3 * sizeof(double));
	h_charge_center[0] = 75.;
	h_charge_center[1] = 50.;
	h_charge_center[2] = 50.;
	cudaMalloc((void**)&d_charge_center, 3 * sizeof(double));

	// Debug: Set boundary centre at (50., 50., 50.)
	cudaMalloc((void**)&d_boundary_atom_center, 3 * sizeof(double));
	cudaMemcpy(d_charge_center, h_charge_center, 3 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_boundary_atom_center, h_boundary_atom_center, 3 * sizeof(double), cudaMemcpyHostToDevice);

	Sphere_Autoconfort_Iterator();
	Write_sphere_info();

	cudaMalloc((void**)&d_sphere_phi_each_point, N_points * sizeof(double));
	cudaMalloc((void**)&d_sphere_Electric_Field_each_point, N_points * 3 * sizeof(double));
	cudaMalloc((void**)&d_E_result, 3 * sizeof(double));
	h_E_result = (double*)malloc(3 * sizeof(double));

	// Calculate Electric Force Using Green Function
	Potential_and_Elecfield_calculator << <(N_points * 3 + 1023) / 1024, 1024 >> > (d_phi, d_from_ortho_to_calc, N_points, d_points, nx, ny, nz, box_x, box_y, box_z, d_sphere_phi_each_point, d_sphere_Electric_Field_each_point);
	Calc_Electric_Force_Using_Green_Function_Serial << <(N_faces * 3 + 1023) / 1024, 1024 >> > (N_faces, d_faces, d_points, d_charge_center, d_sphere_phi_each_point, d_sphere_Electric_Field_each_point, d_E_result);
	cudaMemcpy(h_E_result, d_E_result, 3 * sizeof(double), cudaMemcpyDeviceToHost);
	printf("Electric Force: %f %f %f\n", h_E_result[0], h_E_result[1], h_E_result[2]);



	Safe_cuda_free();
	return 0;
}
