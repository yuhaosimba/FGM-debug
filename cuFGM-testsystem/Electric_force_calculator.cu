#include "Electric_force_calculator.cuh"


static __global__ void Potential_and_Elecfield_calculator(double* phi, int* from_ortho_to_calc, 
	int N_points, double* sphere_crd,
	int nx, int ny, int nz,
	double box_x, double box_y, double box_z, 
	double* sphere_phi_each_point, double* sphere_Electric_Field_each_point)
{
	int ker = blockIdx.x * blockDim.x + threadIdx.x;
	if (ker < N_points) {
		double box_dx = box_x / nx; double box_dy = box_y / ny; double box_dz = box_z / nz;
		double x = sphere_crd[3 * ker]; double y = sphere_crd[3 * ker + 1]; double z = sphere_crd[3 * ker + 2];
		int i = (int)(x / box_dx); int j = (int)(y / box_dy); int k = (int)(z / box_dz);
		double dx = x / box_dx - i;
		double dy = y / box_dy - j;
		double dz = z / box_dz - k;
		double Phi1 = phi[from_ortho_to_calc[i + nx * j + nx * ny * k]];
		double Phi2 = phi[from_ortho_to_calc[(i + 1) + nx * j + nx * ny * k]];
		double Phi3 = phi[from_ortho_to_calc[(i + 1) + nx * (j + 1) + nx * ny * k]];
		double Phi4 = phi[from_ortho_to_calc[i + nx * (j + 1) + nx * ny * k]];
		double Phi5 = phi[from_ortho_to_calc[i + nx * j + nx * ny * (k + 1)]];
		double Phi6 = phi[from_ortho_to_calc[(i + 1) + nx * j + nx * ny * (k + 1)]];
		double Phi7 = phi[from_ortho_to_calc[(i + 1) + nx * (j + 1) + nx * ny * (k + 1)]];
		double Phi8 = phi[from_ortho_to_calc[i + nx * (j + 1) + nx * ny * (k + 1)]];

		sphere_phi_each_point[ker] = (1 - dx) * (1 - dy) * (1 - dz) * Phi1 + dx * (1 - dy) * (1 - dz) * Phi2 + dx * dy * (1 - dz) * Phi3 + (1 - dx) * dy * (1 - dz) * Phi4 +
			(1 - dx) * (1 - dy) * dz * Phi5 + dx * (1 - dy) * dz * Phi6 + dx * dy * dz * Phi7 + (1 - dx) * dy * dz * Phi8;

		sphere_Electric_Field_each_point[3 * ker] = -Phi1 * (1 - dy) * (1 - dz) + Phi2 * (1 - dy) * (1 - dz) + Phi3 * dy * (1 - dz) - Phi4 * dy * (1 - dz) - Phi5 * (1 - dy) * dz + Phi6 * (1 - dy) * dz + Phi7 * dy * dz - Phi8 * dy * dz;
		sphere_Electric_Field_each_point[3 * ker + 1] = -Phi1 * (1 - dx) * (1 - dz) - Phi2 * dx * (1 - dz) + Phi3 * dx * (1 - dz) + Phi4 * (1 - dx) * (1 - dz) - Phi5 * (1 - dx) * dz - Phi6 * dx * dz + Phi7 * dx * dz + Phi8 * (1 - dx) * dz;
		sphere_Electric_Field_each_point[3 * ker + 2] = -Phi1 * (1 - dx) * (1 - dy) - Phi2 * dx * (1 - dy) - Phi3 * dx * dy - Phi4 * (1 - dx) * dy + Phi5 * (1 - dx) * (1 - dy) + Phi6 * dx * (1 - dy) + Phi7 * dx * dy + Phi8 * (1 - dx) * dy;
		sphere_Electric_Field_each_point[3 * ker] *= -1;
		sphere_Electric_Field_each_point[3 * ker + 1] *= -1;
		sphere_Electric_Field_each_point[3 * ker + 2] *= -1;
	}
}

static __global__ void Calc_Electric_Force_Using_Green_Function(int N_faces, int* faces, double* sphere_crd, double* center_crd, 
	double* sphere_phi_each_point, double* sphere_Electric_Field_each_point, double* E_result)
{
	int ker = blockIdx.x * blockDim.x + threadIdx.x;
	if (ker < N_faces) {
		int i = faces[3 * ker]; int j = faces[3 * ker + 1]; int k = faces[3 * ker + 2];
		double x1 = sphere_crd[3 * i]; double y1 = sphere_crd[3 * i + 1]; double z1 = sphere_crd[3 * i + 2];
		double x2 = sphere_crd[3 * j]; double y2 = sphere_crd[3 * j + 1]; double z2 = sphere_crd[3 * j + 2];
		double x3 = sphere_crd[3 * k]; double y3 = sphere_crd[3 * k + 1]; double z3 = sphere_crd[3 * k + 2];

		double Ex = (sphere_Electric_Field_each_point[3 * i] + sphere_Electric_Field_each_point[3 * j] + sphere_Electric_Field_each_point[3 * k]) / 3;
		double Ey = (sphere_Electric_Field_each_point[3 * i + 1] + sphere_Electric_Field_each_point[3 * j + 1] + sphere_Electric_Field_each_point[3 * k + 1]) / 3;
		double Ez = (sphere_Electric_Field_each_point[3 * i + 2] + sphere_Electric_Field_each_point[3 * j + 2] + sphere_Electric_Field_each_point[3 * k + 2]) / 3;
		double phi = (sphere_phi_each_point[i] + sphere_phi_each_point[j] + sphere_phi_each_point[k]) / 3;

		double nx = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1);
		double ny = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1);
		double nz = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
		double S = 0.5 * sqrt(nx * nx + ny * ny + nz * nz);
		nx /= 2 * S; ny /= 2 * S; nz /= 2 * S;

		double vec_x = (x1 + x2 + x3) / 3 - center_crd[0]; double vec_y = (y1 + y2 + y3) / 3 - center_crd[1]; double vec_z = (z1 + z2 + z3) / 3 - center_crd[2];
		double r = sqrt(vec_x * vec_x + vec_y * vec_y + vec_z * vec_z);

		E_result[0] += 1 / pow(r, 3) * phi * S * nx;
		E_result[1] += 1 / pow(r, 3) * phi * S * ny;
		E_result[2] += 1 / pow(r, 3) * phi * S * nz;

		double Add_const = 1 / pow(r, 3) * S * (Ex * nx + Ey * ny + Ez * nz - 3 * phi / pow(r, 2) * (vec_x * nx + vec_y * ny + vec_z * nz));
		E_result[0] += Add_const * vec_x;
		E_result[1] += Add_const * vec_y;
		E_result[2] += Add_const * vec_z;
	}
}

static __global__ void Calc_Electric_Force_Using_Green_Function_Serial(int N_faces, int* faces, double* sphere_crd, double* center_crd,
	double* sphere_phi_each_point, double* sphere_Electric_Field_each_point, double* E_result)
{
	int gpu = blockIdx.x * blockDim.x + threadIdx.x;
	if (gpu ==0) {
		for (int ker = 0; ker < N_faces; ker++) {
			int i = faces[3 * ker]; int j = faces[3 * ker + 1]; int k = faces[3 * ker + 2];
			double x1 = sphere_crd[3 * i]; double y1 = sphere_crd[3 * i + 1]; double z1 = sphere_crd[3 * i + 2];
			double x2 = sphere_crd[3 * j]; double y2 = sphere_crd[3 * j + 1]; double z2 = sphere_crd[3 * j + 2];
			double x3 = sphere_crd[3 * k]; double y3 = sphere_crd[3 * k + 1]; double z3 = sphere_crd[3 * k + 2];

			double Ex = (sphere_Electric_Field_each_point[3 * i] + sphere_Electric_Field_each_point[3 * j] + sphere_Electric_Field_each_point[3 * k]) / 3;
			double Ey = (sphere_Electric_Field_each_point[3 * i + 1] + sphere_Electric_Field_each_point[3 * j + 1] + sphere_Electric_Field_each_point[3 * k + 1]) / 3;
			double Ez = (sphere_Electric_Field_each_point[3 * i + 2] + sphere_Electric_Field_each_point[3 * j + 2] + sphere_Electric_Field_each_point[3 * k + 2]) / 3;
			double phi = (sphere_phi_each_point[i] + sphere_phi_each_point[j] + sphere_phi_each_point[k]) / 3;

			double nx = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1);
			double ny = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1);
			double nz = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
			double S = 0.5 * sqrt(nx * nx + ny * ny + nz * nz);
			nx /= 2 * S; ny /= 2 * S; nz /= 2 * S;

			double vec_x = (x1 + x2 + x3) / 3 - center_crd[0]; double vec_y = (y1 + y2 + y3) / 3 - center_crd[1]; double vec_z = (z1 + z2 + z3) / 3 - center_crd[2];
			double r = sqrt(vec_x * vec_x + vec_y * vec_y + vec_z * vec_z);

			E_result[0] += 1 / pow(r, 3) * phi * S * nx;
			E_result[1] += 1 / pow(r, 3) * phi * S * ny;
			E_result[2] += 1 / pow(r, 3) * phi * S * nz;

			double Add_const = 1 / pow(r, 3) * S * (Ex * nx + Ey * ny + Ez * nz - 3 * phi / pow(r, 2) * (vec_x * nx + vec_y * ny + vec_z * nz));
			E_result[0] += Add_const * vec_x;
			E_result[1] += Add_const * vec_y;
			E_result[2] += Add_const * vec_z;
		}
	}
}
