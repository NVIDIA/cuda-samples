The auto-generated pair of files named matrixMul_ptxdump.c and
matrixMul_ptxdump.h can be acquired by treating matrixMul_kernel.ptx
as binary file and representing its contents as an array of chars.

An example of the script written in Python language that performs 
such translation can be found in "extras" directory of the sample. 

The matrixMul_kernel.ptx contains the same PTX code as the file in
"data" directory of matrixMulDrv sample after compilation by nvcc.

The command line for generation using Python script is as follows:
> python ptx2c.py matrixMul_kernel.ptx matrixMul
