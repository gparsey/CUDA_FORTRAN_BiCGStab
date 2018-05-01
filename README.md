# CUDA_FORTRAN_BiCGStab
Test configuration for a BiCGStab implementation in Fortran using CUDA cuSparse routines

This repos contains a set of files for testing implementations of a BiCGStab solver
written in fortran and using CUDA cuSparse routines; the repo was created in order
to ask for help on StackOverflow while providing source code. These solvers were
written for a legacy Fortran 77 code with the added restriction of only being able 
to use a fortran compiler (as opposed to having a c/c++ interface). 

NOTE
====
The two implementations of BiCGStab are not working, any help on the matter would be 
greatly appreciated. The implementation of the QR decomposition is working correctly.

Testing has been performed using the CUDA 9.1 Toolkit and iFort 17.0.4.196

GPU being used is a Tesla P4 card

FILES
=====
maklefile_testcuda : Makefile (with debugging flags) used for the test case

test_fortcuda.for : Main test program with a simple test system (5x5 matrix with 13 non-zeros,
                    a right hand side, and a known solution) and three logical flags
                    for using the three implemented solvers. This is written in f77 
                    standard to emulate the legacy code.
                    
cuda_fortran_solvers.f90 : Fortran header-equivalent module for referencing CUDA
                           functions and three solvers implemented as subroutines.
                           
---> module cuda_cusolve_map : Interface module to cuda functions for reference within
                               fortran code. This was largely modeled on the corrected
                               answer from 
   https://devtalk.nvidia.com/default/topic/882492/gpu-accelerated-libraries/using-cusolverdn-in-fortran-code/
                           
---> subroutine cuda_sparse_solve_qr : Direct solver using QR decomposition. This solver
                                       is working and is used to confirm that the
                                       interface module is performing correctly. This was
                                       modeled from the CUDA sample cuSolverSp_LinearSolver.cpp
                                       -- method1 in test_fortcuda.for
                                       
---> subroutine cuda_BiCGStab : Iterative BiCGStab solver based on the CUDA sample 
                                pbicgstab.cpp using the standard ILU preconditioner. This
                                solver is not working with a CUSPARSE_INTERNAL_ERROR being 
                                returned from cusparseDcsrsv_analysis (line 1467)
                                -- method2 in test_fortcuda.for
                                
---> subroutine cuda_BiCGStab2 : Iterative BiCGStab solver based off of the previous subroutine
                                 and the example usage domino scheme cusparseDcsrilu02 (section 10.9-11
                                 of the CUDA 9.1 Toolkit documentation). This was
                                 mainly implemented as a sanity check of the previous cuda_BiCGStab,
                                 however seems to be broken in the same manner. This 
                                 solver is not working with a CUSPARSE_INTERNAL_ERROR being 
                                 returned from cusparseDcsrilu02_analysis (line 953)
                                 -- method3 in test_fortcuda.for
                    


