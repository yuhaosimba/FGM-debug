static double* d_sphere_phi_each_point;
static double* d_sphere_Electric_Field_each_point;
static double* d_E_result;  // 最终电场力结果
static double* h_E_result;  // 最终电场力结果

static __global__ void Potential_and_Elecfield_calculator(double* phi, int* from_ortho_to_calc,
	int N_points, double* sphere_crd,
	int nx, int ny, int nz,
	double box_x, double box_y, double box_z,
	double* sphere_phi_each_point, double* sphere_Electric_Field_each_point);


static __global__ void Calc_Electric_Force_Using_Green_Function(int N_faces, int* faces, double* sphere_crd, 
	double* center_crd, double* sphere_phi_each_point, double* sphere_Electric_Field_each_point, double* E_result);

static __global__ void Calc_Electric_Force_Using_Green_Function_Serial(int N_faces, int* faces, double* sphere_crd, double* center_crd,
	double* sphere_phi_each_point, double* sphere_Electric_Field_each_point, double* E_result);