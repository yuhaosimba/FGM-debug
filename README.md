# FGM-debug
this is FGM module with mixture mesh for cudaSPONGE

## Python Mesh Builder
python files for constructing Finite Mesh, Potential solving matrice and Shpere for green integral.

## CSR-Matrix 
data for Finite Mesh and Green Sphere

## cuFGM-testsystem
one-step test for FGM process：
- Calculating Potential 
- Generating Green Shpere
- Calculating Force using Green integral
the system of cuFGM-testsystem is a 3DPBC, 100*100*100 Angstrom box and a equalpotential shpere with R = 5.5 A, centered at (50.0, 50.0, 50.0). a point-charge is placed at somewhere outside the shpere.
