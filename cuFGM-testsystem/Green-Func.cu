#include "Green-Func.cuh"




__global__ void refresh_to_zero(double* d_array, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		d_array[i] = 0;
	}
}

__global__ void multiply_by_constant(double* d_array, int n, double cst) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		d_array[i] = d_array[i] * cst;
	}
}

__global__ void add_by_constant(double* d_array, int n, double cst) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		d_array[i] += cst;
	}
}

__global__ void add_by_vector(double* d_array, int N_points, double dx, double dy, double dz) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N_points) {
		d_array[i * 3] += dx;
		d_array[i * 3 + 1] += dy;
		d_array[i * 3 + 2] += dz;
	}
}

__global__ void boundary_force_shpere(double* points, double* boundary_atom_center,
	const double k1, const double VDW_boundary, const double Attach_force, double* Acc) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N_points) {
		double dx = points[i * 3] - boundary_atom_center[0];
		double dy = points[i * 3 + 1] - boundary_atom_center[1];
		double dz = points[i * 3 + 2] - boundary_atom_center[2];
		double r = sqrt(dx * dx + dy * dy + dz * dz);
		if (r > VDW_boundary) {
			Acc[i * 3] += k1 * dx / pow(r, 2);
			Acc[i * 3 + 1] += k1 * dy / pow(r, 2);
			Acc[i * 3 + 2] += k1 * dz / pow(r, 2);
		}
		else {
			Acc[i * 3] += k1 * dx * (1 / pow(r, 2) + Attach_force * pow(VDW_boundary - r, 1) / r);
			Acc[i * 3 + 1] += k1 * dy * (1 / pow(r, 2) + Attach_force * pow(VDW_boundary - r, 1) / r);
			Acc[i * 3 + 2] += k1 * dz * (1 / pow(r, 2) + Attach_force * pow(VDW_boundary - r, 1) / r);
		}
	}
}

__global__ void triangle_force(double* points, int* faces, const double k2, double* Acc, int N_faces) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N_faces) {
		double x1 = points[faces[i * 3] * 3];
		double y1 = points[faces[i * 3] * 3 + 1];
		double z1 = points[faces[i * 3] * 3 + 2];
		double x2 = points[faces[i * 3 + 1] * 3];
		double y2 = points[faces[i * 3 + 1] * 3 + 1];
		double z2 = points[faces[i * 3 + 1] * 3 + 2];
		double x3 = points[faces[i * 3 + 2] * 3];
		double y3 = points[faces[i * 3 + 2] * 3 + 1];
		double z3 = points[faces[i * 3 + 2] * 3 + 2];
		double nx = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1);
		double ny = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1);
		double nz = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
		double r = sqrt(nx * nx + ny * ny + nz * nz);
		nx /= r; ny /= r; nz /= r;
		Acc[faces[i * 3] * 3] += k2 * nx;
		Acc[faces[i * 3] * 3 + 1] += k2 * ny;
		Acc[faces[i * 3] * 3 + 2] += k2 * nz;
		Acc[faces[i * 3 + 1] * 3] += k2 * nx;
		Acc[faces[i * 3 + 1] * 3 + 1] += k2 * ny;
		Acc[faces[i * 3 + 1] * 3 + 2] += k2 * nz;
		Acc[faces[i * 3 + 2] * 3] += k2 * nx;
		Acc[faces[i * 3 + 2] * 3 + 1] += k2 * ny;
		Acc[faces[i * 3 + 2] * 3 + 2] += k2 * nz;
	}
}


__global__ void edge_force(double* points, int* edges, double* edge_length_init,
	const double k3, const double k4, double* Acc, int N_edges) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N_edges) {
		double x1 = points[edges[i * 2] * 3];
		double y1 = points[edges[i * 2] * 3 + 1];
		double z1 = points[edges[i * 2] * 3 + 2];
		double x2 = points[edges[i * 2 + 1] * 3];
		double y2 = points[edges[i * 2 + 1] * 3 + 1];
		double z2 = points[edges[i * 2 + 1] * 3 + 2];
		double dx = x1 - x2;
		double dy = y1 - y2;
		double dz = z1 - z2;
		double r = sqrt(dx * dx + dy * dy + dz * dz);
		double F = k4 * (edge_length_init[i] - r);
		if (edge_length_init[i] > r) { F += k3 * (edge_length_init[i] - r); }
		Acc[edges[i * 2] * 3] += F * dx;
		Acc[edges[i * 2] * 3 + 1] += F * dy;
		Acc[edges[i * 2] * 3 + 2] += F * dz;
		Acc[edges[i * 2 + 1] * 3] -= F * dx;
		Acc[edges[i * 2 + 1] * 3 + 1] -= F * dy;
		Acc[edges[i * 2 + 1] * 3 + 2] -= F * dz;
	}
}

__global__ void center_force(double* points, double* charge_center,
	const double k5, const double R, double* Acc) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N_points) {
		double dx = points[i * 3] - charge_center[0];
		double dy = points[i * 3 + 1] - charge_center[1];
		double dz = points[i * 3 + 2] - charge_center[2];
		double r = sqrt(dx * dx + dy * dy + dz * dz);
		double F = -k5 / r;
		if (R > r) { F = k5 / r; }
		Acc[i * 3] += F * dx;
		Acc[i * 3 + 1] += F * dy;
		Acc[i * 3 + 2] += F * dz;
	}
}

__global__ void update_position_Leapfrog(double* points, double* Vel, const double dt) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N_points) {
		points[i * 3] += Vel[i * 3] * dt;
		points[i * 3 + 1] += Vel[i * 3 + 1] * dt;
		points[i * 3 + 2] += Vel[i * 3 + 2] * dt;
	}
}

__global__ void update_position(double* points, double* Acc, const double dt) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N_points) {
		points[i * 3] += Acc[i * 3] * pow(dt, 2);
		points[i * 3 + 1] += Acc[i * 3 + 1] * pow(dt, 2);
		points[i * 3 + 2] += Acc[i * 3 + 2] * pow(dt, 2);
	}
}


__global__ void update_velocity_Leapfrog(double* Vel, double* Acc, const double dt) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N_points) {
		Vel[i * 3] += Acc[i * 3] * dt;
		Vel[i * 3 + 1] += Acc[i * 3 + 1] * dt;
		Vel[i * 3 + 2] += Acc[i * 3 + 2] * dt;
	}
}


// 现已弃用
__global__ static void langevin_friction(double* Vel, double* Acc, const double gamma) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N_points) {
		Acc[i * 3] -= gamma * Vel[i * 3];
		Acc[i * 3 + 1] -= gamma * Vel[i * 3 + 1];
		Acc[i * 3 + 2] -= gamma * Vel[i * 3 + 2];
	}
}

// 初始化球面
void Read_and_Initialize_sphere_info() {
	std::ifstream f_info("C:\\Users\\15653\\Desktop\\FGM-Python-Debug\\unit_500Sphere\\info.txt");
	f_info >> N_edges >> N_faces;
	f_info.close();
	h_points = (double*)malloc(N_points * 3 * sizeof(double));
	h_edges = (int*)malloc(N_edges * 2 * sizeof(int));
	h_faces = (int*)malloc(N_faces * 3 * sizeof(int));
	h_edges_length_unit = (double*)malloc(N_edges * sizeof(double));
	result = (double*)malloc(N_points * 3 * sizeof(double));

	cudaMalloc((void**)&d_points, N_points * 3 * sizeof(double));
	cudaMalloc((void**)&d_edges, N_edges * 2 * sizeof(int));
	cudaMalloc((void**)&d_faces, N_faces * 3 * sizeof(int));
	cudaMalloc((void**)&d_edges_length_unit, N_edges * sizeof(double));
	cudaMalloc((void**)&d_Acc, N_points * 3 * sizeof(double));
	cudaMalloc((void**)&d_Vel, N_points * 3 * sizeof(double));

	std::ifstream f_points("C:\\Users\\15653\\Desktop\\FGM-Python-Debug\\unit_500Sphere\\points.txt");
	for (int i = 0; i < N_points * 3; i++) { f_points >> h_points[i]; }
	f_points.close();

	std::ifstream f_edges("C:\\Users\\15653\\Desktop\\FGM-Python-Debug\\unit_500Sphere\\edges.txt");
	for (int i = 0; i < N_edges * 2; i++) { f_edges >> h_edges[i]; }
	f_edges.close();

	std::ifstream f_faces("C:\\Users\\15653\\Desktop\\FGM-Python-Debug\\unit_500Sphere\\faces.txt");
	for (int i = 0; i < N_faces * 3; i++) { f_faces >> h_faces[i]; }
	f_faces.close();

	std::ifstream f_edges_length_unit("C:\\Users\\15653\\Desktop\\FGM-Python-Debug\\unit_500Sphere\\edge_length_init.txt");
	for (int i = 0; i < N_edges; i++) { f_edges_length_unit >> h_edges_length_unit[i]; }
	f_edges_length_unit.close();

	cudaMemcpy(d_points, h_points, N_points * 3 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_edges, h_edges, N_edges * 2 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_faces, h_faces, N_faces * 3 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_edges_length_unit, h_edges_length_unit, N_edges * sizeof(double), cudaMemcpyHostToDevice);

}

// 保存球面位置信息
void Write_sphere_info() {
	cudaMemcpy(result, d_points, N_points * 3 * sizeof(double), cudaMemcpyDeviceToHost);
	// clean result.txt
	std::ofstream f_clean("C:\\Users\\15653\\Desktop\\FGM-Python-Debug\\unit_500Sphere\\result.txt");
	// write result to txt
	std::ofstream f_result("C:\\Users\\15653\\Desktop\\FGM-Python-Debug\\unit_500Sphere\\result.txt");
	for (int i = 0; i < N_points; i++) {
		f_result << result[3 * i] << "\t"; f_result << result[3 * i + 1] << "\t"; f_result << result[3 * i + 2] << std::endl;
	}
}

void Sphere_Autoconfort_Iterator() {
	// 初始化采样球面位置
	multiply_by_constant << <(N_points * 3 + 1023) / 1024, 1024 >> > (d_points, N_points * 3, R);
	add_by_vector << <(N_points + 1023) / 1024, 1024 >> > (d_points, N_points, 75., 50., 50.);
	// 初始化速度
	refresh_to_zero << <(N_points * 3 + 1023) / 1024, 1024 >> > (d_Vel, N_points * 3);


	// 动力学演化自适应曲面
	for (int i = 0; i < max_iter; i++) {
		// 重置加速度
		refresh_to_zero << <(N_points * 3 + 1023) / 1024, 1024 >> > (d_Acc, N_points * 3);
		// 计算力
		boundary_force_shpere << <(N_points + 1023) / 1024, 1024 >> > (d_points, d_boundary_atom_center, k1, VDW_boundary, Attach_force, d_Acc);
		triangle_force << <(N_faces + 1023) / 1024, 1024 >> > (d_points, d_faces, k2, d_Acc, N_faces);
		edge_force << <(N_edges + 1023) / 1024, 1024 >> > (d_points, d_edges, d_edges_length_unit, k3, k4, d_Acc, N_edges);
		center_force << <(N_points + 1023) / 1024, 1024 >> > (d_points, d_charge_center, k5, R, d_Acc);
		//langevin_friction << <(N_points + 1023) / 1024, 1024 >> > (d_Vel, d_Acc, 0.1); 现已弃用该方法

		// 更新速度
		update_velocity_Leapfrog << <(N_points + 1023) / 1024, 1024 >> > (d_Vel, d_Acc, dt);
		// 引入摩擦
		multiply_by_constant << <(N_points * 3 + 1023) / 1024, 1024 >> > (d_Vel, N_points * 3, gamma);
		// 更新位置
		update_position_Leapfrog << <(N_points + 1023) / 1024, 1024 >> > (d_points, d_Vel, dt);
	}
}
