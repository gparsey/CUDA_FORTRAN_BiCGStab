!
! CUDA
!
module cuda_cusolve_map

 interface

 ! cudaMemset
 integer (c_int) function cudaMemset( devPtr,value, count ) &
                              bind (C, name="cudaMemset" ) 
   use iso_c_binding
   implicit none
   type (c_ptr),value  :: devPtr
   integer(c_int), value :: value
   integer(c_size_t), value :: count
 end function cudaMemset
 ! cudaMalloc
 integer (c_int) function cudaMalloc ( buffer, size ) &
                              bind (C, name="cudaMalloc" ) 
   use iso_c_binding
   implicit none
   type (c_ptr)  :: buffer
   integer (c_size_t), value :: size
 end function cudaMalloc

 integer (c_int) function cudaMemcpy ( dst, src, count, kind ) &
                              bind (C, name="cudaMemcpy" )
   ! note: cudaMemcpyHostToDevice = 1
   ! note: cudaMemcpyDeviceToHost = 2
   ! note: cudaMemcpyDeviceToDevice = 3
   use iso_c_binding
   type (c_ptr), value :: dst, src
   integer (c_size_t), value :: count, kind
 end function cudaMemcpy

 ! cudaFree
 integer (c_int) function cudaFree(buffer)  bind(C, name="cudaFree")
   use iso_c_binding
   implicit none
   type (c_ptr), value :: buffer
 end function cudaFree

 integer (c_int) function cudaMemGetInfo(fre, tot) &
                              bind(C, name="cudaMemGetInfo")
   use iso_c_binding
   implicit none
   type(c_ptr),value :: fre
   type(c_ptr),value :: tot
 end function cudaMemGetInfo

 integer(c_int) function cusparseCreate(cusparseHandle) &
                             bind(C,name="cusparseCreate")

   use iso_c_binding
   implicit none
   type(c_ptr)::cusparseHandle
   end function cusparseCreate

 integer(c_int) function cudaStreamCreate(stream) &
                             bind(C,name="cudaStreamCreate")

 use iso_c_binding
 implicit none
 type(c_ptr)::stream
 end function cudaStreamCreate

 integer(c_int) function cusolverSpSetStream(handle,stream) &
                             bind(C,name="cusolverSpSetStream")

 use iso_c_binding
 implicit none
 type(c_ptr),value :: handle
 type(c_ptr),value :: stream
 end function cusolverSpSetStream

 integer(c_int) function cusparseSetStream(cusparseHandle,stream) &
                             bind(C,name="cusparseSetStream")

 use iso_c_binding
 implicit none
 type(c_ptr),value :: cusparseHandle
 type(c_ptr),value :: stream
 end function cusparseSetStream

 integer(c_int) function cusparseCreateMatDescr(descrA) &
                             bind(C,name="cusparseCreateMatDescr")

 use iso_c_binding
 implicit none
 type(c_ptr):: descrA
 end function cusparseCreateMatDescr

 integer(c_int) function cusparseGetMatType(descrA) &
                             bind(C,name="cusparseGetMatType")

 use iso_c_binding
 implicit none
 type(c_ptr), value:: descrA
 end function cusparseGetMatType

 integer(c_int) function cusparseGetMatIndexBase(descrA) &
                             bind(C,name="cusparseGetMatIndexBase")

 use iso_c_binding
 implicit none
 type(c_ptr), value:: descrA
 end function cusparseGetMatIndexBase

 integer(c_int) function cusparseGetMatFillMode(descrA) &
                             bind(C,name="cusparseGetMatFillMode")

 use iso_c_binding
 implicit none
 type(c_ptr), value:: descrA
 end function cusparseGetMatFillMode

 integer(c_int) function cusparseGetMatDiagType(descrA) &
                             bind(C,name="cusparseGetMatDiagType")

 use iso_c_binding
 implicit none
 type(c_ptr), value:: descrA
 end function cusparseGetMatDiagType


 integer(c_int) function cusparseSetMatType2(descrA,CUSPARSE_MATRIX_TYPE) &
                             bind(C,name="cusparseSetMatType")

 use iso_c_binding
 implicit none
 type(c_ptr), value:: descrA
 integer(c_int),value :: CUSPARSE_MATRIX_TYPE
 end function cusparseSetMatType2

 integer(c_int) function cusparseSetMatIndexBase2(descrA,CUSPARSE_INDEX_BASE) &
                             bind(C,name="cusparseSetMatIndexBase")

 use iso_c_binding
 implicit none
 type(c_ptr), value:: descrA
 integer(c_int),value :: CUSPARSE_INDEX_BASE
 end function cusparseSetMatIndexBase2

 integer(c_int) function cusparseSetMatFillMode(descrA,CUSPARSE_FILL_TYPE) &
                 bind(C,name="cusparseSetMatFillMode")

 use iso_c_binding
 implicit none
 type(c_ptr), value:: descrA
 integer(c_int),value :: CUSPARSE_FILL_TYPE
 end function cusparseSetMatFillMode

 integer(c_int) function cusparseSetMatDiagType(descrA,CUSPARSE_DIAG_TYPE) &
                 bind(C,name="cusparseSetMatDiagType")

 use iso_c_binding
 implicit none
 type(c_ptr), value:: descrA
 integer(c_int),value :: CUSPARSE_DIAG_TYPE
 end function cusparseSetMatDiagType

 integer(c_int) function cusolverSpXcsrsymrcmHost(handle,rowsA,nnzA,descrA,&
                             h_csrRowPtrA,h_csrColIndA,h_Q) &
                             bind(C,name="cusolverSpXcsrsymrcmHost")

 use iso_c_binding
 implicit none
 type(c_ptr),value :: handle
 integer(c_int),value :: rowsA
 integer(c_int),value :: nnzA
 type(c_ptr), value:: descrA
 type(c_ptr),value :: h_csrRowPtrA
 type(c_ptr),value :: h_csrColIndA
 type(c_ptr),value :: h_Q
 end function cusolverSpXcsrsymrcmHost

 integer(c_int) function cusolverSpXcsrsymamdHost(handle,rowsA,nnzA,descrA,&
                             h_csrRowPtrA,h_csrColIndA,h_Q) &
                             bind(C,name="cusolverSpXcsrsymamdHost")

 use iso_c_binding
 implicit none
 type(c_ptr),value :: handle
 integer(c_int),value :: rowsA
 integer(c_int),value :: nnzA
 type(c_ptr), value:: descrA
 type(c_ptr),value :: h_csrRowPtrA
 type(c_ptr),value :: h_csrColIndA
 type(c_ptr),value :: h_Q
 end function cusolverSpXcsrsymamdHost

 integer(c_int) function cusolverSpXcsrperm_bufferSizeHost(handle,rowsA,colsA,&
                             nnzA,descrA,h_csrRowPtrB,h_csrColIndB,h_Q1,h_Q2,reorderWork) &
                             bind(C,name="cusolverSpXcsrperm_bufferSizeHost")

 use iso_c_binding
 implicit none
 type(c_ptr),value :: handle
 integer(c_int),value :: rowsA
 integer(c_int),value :: colsA
 integer(c_int),value :: nnzA
 type(c_ptr), value:: descrA
 type(c_ptr),value :: h_csrRowPtrB
 type(c_ptr),value :: h_csrColIndB
 type(c_ptr),value :: h_Q1
 type(c_ptr),value :: h_Q2
 !type(c_ptr) :: reorderWork
 integer(c_size_t) :: reorderWork
 end function cusolverSpXcsrperm_bufferSizeHost

 integer(c_int) function cusolverSpXcsrpermHost(handle,rowsA,colsA,nnzA,descrA,&
                             h_csrRowPtrB,h_csrColIndB,h_Q1,h_Q2,h_mapBfromA,buffer_cpu) &
                             bind(C,name="cusolverSpXcsrpermHost")

 use iso_c_binding
 implicit none
 type(c_ptr),value :: handle
 integer(c_int),value :: rowsA
 integer(c_int),value :: colsA
 integer(c_int),value :: nnzA
 type(c_ptr), value:: descrA
 type(c_ptr),value :: h_csrRowPtrB
 type(c_ptr),value :: h_csrColIndB
 type(c_ptr),value :: h_Q1
 type(c_ptr),value :: h_Q2
 type(c_ptr),value :: h_mapBfromA
 type(c_ptr),value :: buffer_cpu
 end function cusolverSpXcsrpermHost

 integer(c_int) function cusolverSpDcsrlsvqr(handle,rowsA,nnzA,descrA,d_csrValA,&
                             d_csrRowPtrA,d_csrColIndA,d_b,tol,reorder,d_x,singularity) &
                             bind(C,name="cusolverSpDcsrlsvqr")

 use iso_c_binding
 implicit none
 type(c_ptr),value :: handle
 integer(c_int),value :: rowsA
 integer(c_int),value :: nnzA
 type(c_ptr), value:: descrA
 type(c_ptr),value :: d_csrValA
 type(c_ptr),value :: d_csrRowPtrA
 type(c_ptr),value :: d_csrColIndA
 type(c_ptr),value :: d_b
 real(c_double),value :: tol
 integer(c_int), value :: reorder
 type(c_ptr),value :: d_x
 integer(c_int) :: singularity
 end function cusolverSpDcsrlsvqr

 integer(c_int) function cusolverSpDcsrlsvqrHost(handle,rowsA,nnzA,descrA,h_csrValA,&
                             h_csrRowPtrA,h_csrColIndA,h_b,tol,reorder,h_x,singularity) &
                             bind(C,name="cusolverSpDcsrlsvqrHost")

 use iso_c_binding
 implicit none
 type(c_ptr),value :: handle
 integer(c_int),value :: rowsA
 integer(c_int),value :: nnzA
 type(c_ptr), value:: descrA
 type(c_ptr),value :: h_csrValA
 type(c_ptr),value :: h_csrRowPtrA
 type(c_ptr),value :: h_csrColIndA
 type(c_ptr),value :: h_b
 real(c_double),value :: tol
 integer(c_int), value :: reorder
 type(c_ptr),value :: h_x
 integer(c_int) :: singularity
 end function cusolverSpDcsrlsvqrHost

 integer(c_int) function cusolverSpDcsrlsvluHost(handle,rowsA,nnzA,descrA,h_csrValA,&
                             h_csrRowPtrA,h_csrColIndA,h_b,tol,reorder,h_x,singularity) &
                             bind(C,name="cusolverSpDcsrlsvluHost")

 use iso_c_binding
 implicit none
 type(c_ptr),value :: handle
 integer(c_int),value :: rowsA
 integer(c_int),value :: nnzA
 type(c_ptr), value:: descrA
 type(c_ptr),value :: h_csrValA
 type(c_ptr),value :: h_csrRowPtrA
 type(c_ptr),value :: h_csrColIndA
 type(c_ptr),value :: h_b
 real(c_double),value :: tol
 integer(c_int), value :: reorder
 type(c_ptr),value :: h_x
 integer(c_int) :: singularity
 end function cusolverSpDcsrlsvluHost

 integer(c_int) function cudaDeviceSynchronize() bind(C,name="cudaDeviceSynchronize")

 use iso_c_binding
 implicit none
 end function cudaDeviceSynchronize

 integer(c_int) function cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION,rowsA,colsA,&
                             nnzA,minus_one,descrA,d_csrValA,d_csrRowPtrA,d_csrColIndA,d_x,one,d_r) &
                             bind(C,name="cusparseDcsrmv")

 use iso_c_binding
 implicit none
 type(c_ptr),value::cusparseHandle
 integer(c_int),value::CUSPARSE_OPERATION
 integer(c_int),value::rowsA
 integer(c_int),value::colsA
 integer(c_int),value::nnzA
 real(c_double)::minus_one
 type(c_ptr), value:: descrA
 type(c_ptr),value::d_csrValA
 type(c_ptr),value::d_csrRowPtrA
 type(c_ptr),value::d_csrColIndA
 type(c_ptr),value::d_x
 real(c_double)::one
 type(c_ptr),value::d_r
 end function cusparseDcsrmv

 integer(c_int) function cusolverSpCreate(handle) bind(C,name="cusolverSpCreate")

 use iso_c_binding
 implicit none
 type(c_ptr)::handle
 end function cusolverSpCreate

 integer(c_int) function cusolverSpDestroy(handle) bind(C,name="cusolverSpDestroy")

 use iso_c_binding
 implicit none
 type(c_ptr),value::handle
 end function cusolverSpDestroy

 integer(c_int) function cublasDestroy(handle) bind(C,name="cublasDestroy_v2")

 use iso_c_binding
 implicit none
 type(c_ptr),value::handle
 end function cublasDestroy

 integer(c_int) function cusparseDestroy(cusparseHandle) bind(C,name="cusparseDestroy")

 use iso_c_binding
 implicit none
 type(c_ptr),value::cusparseHandle
 end function cusparseDestroy

 integer(c_int) function cudaStreamDestroy(stream) bind(C,name="cudaStreamDestroy")

 use iso_c_binding
 implicit none
 type(c_ptr),value :: stream
 end function cudaStreamDestroy

 integer(c_int) function cusparseDestroyMatDescr(descrA) bind(C,name="cusparseDestroyMatDescr")

 use iso_c_binding
 implicit none
 type(c_ptr), value:: descrA
 end function cusparseDestroyMatDescr

!
 integer(c_int) function cublasCreate(handle) &
               bind(C,name="cublasCreate_v2")

 use iso_c_binding
 implicit none
 type(c_ptr):: handle
 end function cublasCreate

 integer(c_int) function cusparseCreateSolveAnalysisInfo(info) &
               bind(C,name="cusparseCreateSolveAnalysisInfo")

 use iso_c_binding
 implicit none
 type(c_ptr) :: info
 end function cusparseCreateSolveAnalysisInfo

 integer(c_int) function cusparseDcsrsv_analysis(handle,transA, &
                 m,nnz,descrA,csrValA,csrRowPtrA,csrColIndA,info) &
                 bind(C,name="cusparseDcsrsv_analysis")

 use iso_c_binding
 implicit none
 type(c_ptr), value :: handle
 integer(c_int), value :: transA
 integer(c_int), value :: m
 integer(c_int),value :: nnz
 type(c_ptr), value :: descrA
 type(c_ptr), value :: csrValA
 type(c_ptr), value :: csrRowPtrA
 type(c_ptr), value :: csrColIndA
 type(c_ptr), value :: info
 end function cusparseDcsrsv_analysis

 integer(c_int) function cublasDscal(handle,n,alpha,x,incx) &
               bind(C,name="cublasDscal_v2")

 use iso_c_binding
 implicit none
 type(c_ptr), value :: handle
 integer(c_int), value :: n
 real(c_double) ::alpha
 type(c_ptr),value :: x
 integer(c_int),value :: incx
 end function cublasDscal

 integer(c_int) function cublasDaxpy(handle,n,alpha,x,incx,y,incy) &
               bind(C,name="cublasDaxpy_v2")

 use iso_c_binding
 implicit none
 type(c_ptr), value :: handle
 integer(c_int), value :: n
 real(c_double)::alpha
 type(c_ptr),value :: x
 integer(c_int),value :: incx
 type(c_ptr),value :: y
 integer(c_int),value :: incy
 end function cublasDaxpy

 integer(c_int) function cublasDcopy(handle,n,x,incx,y,incy) &
               bind(C,name="cublasDcopy_v2")

 use iso_c_binding
 implicit none
 type(c_ptr), value :: handle
 integer(c_int), value :: n
 type(c_ptr),value :: x
 integer(c_int),value :: incx
 type(c_ptr),value :: y
 integer(c_int),value :: incy
 end function cublasDcopy

 integer(c_int) function cublasDnrm2(handle,n,x,incx,result) &
               bind(C,name="cublasDnrm2_v2")

 use iso_c_binding
 implicit none
 type(c_ptr), value :: handle
 integer(c_int), value :: n
 type(c_ptr),value :: x
 integer(c_int),value :: incx
 type(c_ptr),value :: result
 end function cublasDnrm2

 integer(c_int) function cublasDdot(handle,n,x,incx,y,incy,result) &
               bind(C,name="cublasDdot_v2")

 use iso_c_binding
 implicit none
 type(c_ptr), value :: handle
 integer(c_int), value :: n
 type(c_ptr),value :: x
 integer(c_int),value :: incx
 type(c_ptr),value :: y
 integer(c_int),value :: incy
 type(c_ptr),value :: result
 end function cublasDdot

 integer(c_int) function cusparseDcsrsv_solve(handle,transA,m, &
                 alpha, descrA,csrSortedValA,csrSortedRowPtrA, &
                 csrSortedColIndA,info,f,x) &
                 bind(C,name="cusparseDcsrsv_solve")

 use iso_c_binding
 implicit none
 type(c_ptr), value :: handle
 integer(c_int), value :: transA
 integer(c_int), value :: m
 real(c_double)::alpha
 type(c_ptr), value :: descrA
 type(c_ptr), value :: csrSortedValA
 type(c_ptr), value :: csrSortedRowPtrA
 type(c_ptr), value :: csrSortedColIndA
 type(c_ptr), value :: info
 type(c_ptr), value :: f
 type(c_ptr), value :: x
 end function cusparseDcsrsv_solve

 integer(c_int) function cusparseDestroySolveAnalysisInfo(info) &
                 bind(C,name="cusparseDestroySolveAnalysisInfo")

 use iso_c_binding
 implicit none
 type(c_ptr),value::info
 end function cusparseDestroySolveAnalysisInfo

! Older API: cublasSetKernelStream = cublasSetStream
 integer(c_int) function cublasSetStream(handle,stream) &
                 bind(C,name="cublasSetStream_v2")

 use iso_c_binding
 implicit none
 type(c_ptr),value::handle
 type(c_ptr),value::stream
 end function cublasSetStream

 integer(c_int) function cusparseDcsrilu0(handle,CUSPARSE_OPERATION,&
                 m,descrA,csrValM,csrRowPtrA,csrColIndA,info) &
                 bind(C,name="cusparseDcsrilu0")

 use iso_c_binding
 implicit none
 type(c_ptr),value::handle
 integer(c_int),value::CUSPARSE_OPERATION
 integer(c_int),value::m
 type(c_ptr), value:: descrA
 type(c_ptr), value::csrValM
 type(c_ptr), value::csrRowPtrA
 type(c_ptr), value::csrColIndA
 type(c_ptr), value::info
 end function cusparseDcsrilu0

 integer(c_int) function cusparseCreateCsrsv2Info(info) &
                 bind(C,name="cusparseCreateCsrsv2Info")

 use iso_c_binding
 implicit none
 type(c_ptr)::info
 end function cusparseCreateCsrsv2Info

 integer(c_int) function cusparseCreateCsrilu02Info(info) &
                 bind(C,name="cusparseCreateCsrilu02Info")

 use iso_c_binding
 implicit none
 type(c_ptr)::info
 end function cusparseCreateCsrilu02Info

 integer(c_int) function cusparseDcsrilu02_bufferSize(handle,m,nnz,descrA,&
                 csrValM,csrRowPtrA,csrColIndA,info,pBufferSize) &
                 bind(C,name="cusparseDcsrilu02_bufferSize")

 use iso_c_binding
 implicit none
 type(c_ptr),value::handle
 integer(c_int),value::m
 integer(c_int),value::nnz
 type(c_ptr), value:: descrA
 type(c_ptr), value::csrValM
 type(c_ptr), value::csrRowPtrA
 type(c_ptr), value::csrColIndA
 type(c_ptr),value::info
 integer(c_int) :: pBufferSize
 end function cusparseDcsrilu02_bufferSize

 integer(c_int) function cusparseDcsrsv2_bufferSize(handle,CUSPARSE_OPERATION,&
                 m,nnz,descrA,csrValM,csrRowPtrA,csrColIndA,info,pBufferSize) &
                 bind(C,name="cusparseDcsrsv2_bufferSize")

 use iso_c_binding
 implicit none
 type(c_ptr),value::handle
 integer(c_int),value::CUSPARSE_OPERATION
 integer(c_int),value::m
 integer(c_int),value::nnz
 type(c_ptr), value:: descrA
 type(c_ptr), value::csrValM
 type(c_ptr), value::csrRowPtrA
 type(c_ptr), value::csrColIndA
 type(c_ptr),value::info
 integer(c_int) :: pBufferSize
 end function cusparseDcsrsv2_bufferSize

 integer(c_int) function cusparseDcsrilu02_analysis(handle,m,nnz,descrA,&
                 csrValM,csrRowPtrA,csrColIndA,info,policy,pBuffer) &
                 bind(C,name="cusparseDcsrilu02_analysis")

 use iso_c_binding
 implicit none
 type(c_ptr),value::handle
 integer(c_int),value::m
 integer(c_int),value::nnz
 type(c_ptr), value:: descrA
 type(c_ptr), value::csrValM
 type(c_ptr), value::csrRowPtrA
 type(c_ptr), value::csrColIndA
 type(c_ptr), value::info
 integer(c_int),value::policy
 type(c_ptr)::pBuffer
 end function cusparseDcsrilu02_analysis

 integer(c_int) function cusparseDcsrsv2_analysis(handle,CUSPARSE_OPERATION,&
                 m,nnz,descrA,csrValM,csrRowPtrA,csrColIndA,info,policy,pBuffer) &
                 bind(C,name="cusparseDcsrsv2_analysis")

 use iso_c_binding
 implicit none
 type(c_ptr),value::handle
 integer(c_int),value::CUSPARSE_OPERATION
 integer(c_int),value::m
 integer(c_int),value::nnz
 type(c_ptr), value:: descrA
 type(c_ptr), value::csrValM
 type(c_ptr), value::csrRowPtrA
 type(c_ptr), value::csrColIndA
 type(c_ptr),value::info
 integer(c_int),value::policy
 type(c_ptr)::pBuffer
 end function cusparseDcsrsv2_analysis

 integer(c_int) function cusparseDcsrsv2_solve(handle,CUSPARSE_OPERATION,&
                 m,nnz,alpha,descrA,csrValM,csrRowPtrA,csrColIndA,info,x,y,&
                 policy,pBuffer) &
                 bind(C,name="cusparseDcsrsv2_solve")

 use iso_c_binding
 implicit none
 type(c_ptr),value::handle
 integer(c_int),value::CUSPARSE_OPERATION
 integer(c_int),value::m
 integer(c_int),value::nnz
 real(c_double)::alpha
 type(c_ptr), value:: descrA
 type(c_ptr),value::csrValM
 type(c_ptr),value::csrRowPtrA
 type(c_ptr),value::csrColIndA
 type(c_ptr),value::info
 type(c_ptr)::x
 type(c_ptr)::y
 integer(c_int),value::policy
 type(c_ptr)::pBuffer
 end function cusparseDcsrsv2_solve

 integer(c_int) function cusparseDcsrilu02(handle,n,inz,descrA,csrValM,&
                 csrRowPtrA,csrColIndA,info,policy,pBuffer) &
                 bind(C,name="cusparseDcsrilu02")

 use iso_c_binding
 implicit none
 type(c_ptr),value::handle
 integer(c_int),value::n
 integer(c_int),value::inz
 type(c_ptr), value:: descrA
 type(c_ptr),value::csrValM
 type(c_ptr),value::csrRowPtrA
 type(c_ptr),value::csrColIndA
 type(c_ptr),value::info
 integer(c_int),value::policy
 type(c_ptr)::pBuffer
 end function cusparseDcsrilu02

 integer(c_int) function cusparseXcsrilu02_zeroPivot(handle,info,jcord) &
                 bind(C,name="cusparseXcsrilu02_zeroPivot")

 use iso_c_binding
 implicit none
 type(c_ptr),value::handle
 type(c_ptr),value::info
 type(c_ptr) :: jcord
 end function cusparseXcsrilu02_zeroPivot

 end interface  

end module cuda_cusolve_map
!
!
!=========================================================
subroutine cuda_BiCGStab2(n,rhs,x,inz,ib,jb,b, tol)
!=========================================================
use iso_c_binding
use cuda_cusolve_map
implicit none
integer n, inz
double precision x(n), rhs(n), b(inz)
target rhs,b,x
integer ib(n+1),jb(inz)
target ib,jb
integer ii,ierr,ierr2,inum_it,imax_num_it,isol_found
double precision tol,norm_val,norm_val0
target norm_val,norm_val0
double precision rho,rhop,beta,alpha,negalpha
target rho
double precision omega,negomega,temp,temp2,bad_pos
target temp,temp2,bad_pos

integer, parameter :: dp = kind(1.d0)

type(c_ptr) :: cublashandle !cusolver_Hndl
type(c_ptr) :: cusparseHandle
type(c_ptr) :: stream
type(c_ptr) :: descrA
type(c_ptr) :: descrM
type(c_ptr) :: descrL
type(c_ptr) :: descrU
type(c_ptr) :: info_m
type(c_ptr) :: info_l
type(c_ptr) :: info_u
type(c_ptr) :: ArowsIndex
type(c_ptr) :: AcolsIndex
type(c_ptr) :: Aval
type(c_ptr) :: h_x  
type(c_ptr) :: h_rhs
type(c_ptr) :: normPtr
type(c_ptr) :: normPtr0
type(c_ptr) :: rhoPtr
type(c_ptr) :: tempPtr
type(c_ptr) :: tempPtr2
type(c_ptr) :: bad_posPtr

! -------------------- pointers to device memory    
type(c_ptr) :: devPtrArowsIndex
type(c_ptr) :: devPtrAcolsIndex
type(c_ptr) :: devPtrAval
type(c_ptr) :: devPtrMrowsIndex
type(c_ptr) :: devPtrMcolsIndex
type(c_ptr) :: devPtrMval
type(c_ptr) :: devPtrX
type(c_ptr) :: devPtrF
type(c_ptr) :: devPtrR
type(c_ptr) :: devPtrRW
type(c_ptr) :: devPtrP
type(c_ptr) :: devPtrPW
type(c_ptr) :: devPtrS
type(c_ptr) :: devPtrT
type(c_ptr) :: devPtrV

integer*8 Arow1_i_size,Arow_d_size,Acol_d_size,Annz_i_size,Annz_d_size
integer(c_int) matrixM,matrixN, nnz, mNNZ
integer(c_int) matrixSizeAval, matrixSizeAcolsIndex, matrixSizeArowsIndex
integer(c_int) mSizeAval, mSizeAcolsIndex, mSizeArowsIndex
integer(c_int) arraySizeX, arraySizeF, arraySizeR, arraySizeRW, arraySizeP
integer(c_int) arraySizePW, arraySizeS, arraySizeT, arraySizeV
 
integer*8 cudaMemcpyDeviceToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToDevice
integer*4 CUBLAS_OP_N, CUBLAS_OP_T
parameter (cudaMemcpyHostToDevice=1)
parameter (cudaMemcpyDeviceToHost=2)
parameter (cudaMemcpyDeviceToDevice=3)
parameter (CUBLAS_OP_N=0)
parameter (CUBLAS_OP_T=1)

! the constants are used in residual evaluation, r = b - A*x
real(kind=dp) minus_one
parameter (minus_one=-1.0d0)
real(kind=dp) one
parameter (one=1.0d0)
real(kind=dp) zero
parameter (zero=1.0d0)
integer(c_int) :: buffer_m,buffer_l,buffer_u,buffer_size
target buffer_m,buffer_l,buffer_u
integer(c_size_t) :: buffer_malloc
type(c_ptr) :: buffer_m_ptr
type(c_ptr) :: buffer_l_ptr
type(c_ptr) :: buffer_u_ptr
type(c_ptr) :: buffer_gpu 


ierr2 = 0
isol_found=0
imax_num_it =  10
rho = 0.0d0

! write(*,*) 'Overwriting solution space'
! n=5
! inz = 13

! b(1) = 1.0d0; b(2)=1.0d0; b(3)=5.0d0; b(4)=2.0d0
! b(5) = 1.0d0; b(6)=3.0d0; b(7)=2.0d0; b(8)=1.0d0
! b(9) = 6.0d0; b(10)=3.0d0; b(11)=1.0d0; b(12)=2.0d0
! b(13) = 1.0d0

! rhs(1)=1.0d0;rhs(2)=2.0d0;rhs(3)=1.0d0;rhs(4)=3.0d0;rhs(5)=0.0d0

! ib(1) = 1;ib(2) = 5;ib(3) = 7
! ib(4) = 9;ib(5) = 12;ib(6) = 14

! jb(1) = 1; jb(2)  = 2; jb(3) = 4; jb(4) = 5
! jb(5) = 2; jb(6)  = 3; jb(7) = 2; jb(8) = 3
! jb(9) = 1; jb(10) = 3;jb(11) = 4;jb(12) = 4
! jb(13) = 5

!solution
! x(1)=0.08
! x(2)=0.2
! x(3)=0.6
! x(4)=0.72
! x(5)=-1.44


! define pointers to host memory
ArowsIndex = c_loc(ib)
AcolsIndex = c_loc(jb)
Aval = c_loc(b)
h_x  = c_loc(x)  ! x = A \ b
h_rhs  = c_loc(rhs)  ! b = ones(m,1)
normPtr = c_loc(norm_val)
normPtr0 = c_loc(norm_val0)
rhoPtr = c_loc(rho)
tempPtr = c_loc(temp)
tempPtr2 = c_loc(temp2)
bad_posPtr = c_loc(bad_pos)
buffer_m_ptr = c_loc(buffer_m)
buffer_l_ptr = c_loc(buffer_l)
buffer_u_ptr = c_loc(buffer_u)

Arow1_i_size=sizeof(ib(1:n+1))
Arow_d_size=sizeof(rhs(1:n))
Acol_d_size=sizeof(x(1:n))
Annz_i_size=sizeof(jb(1:inz))
Annz_d_size=sizeof(b(1:inz))

! Define the CUDA stream and matrix parameters
ierr = cublasCreate(cublashandle)
ierr2 = ierr2 + ierr
ierr = cusparseCreate(cusparseHandle)
ierr2 = ierr2 + ierr
ierr = cusparseCreateMatDescr(descrA)
ierr2 = ierr2 + ierr
ierr = cusparseCreateMatDescr(descrM)
ierr2 = ierr2 + ierr
ierr = cusparseCreateMatDescr(descrL)
ierr2 = ierr2 + ierr
ierr = cusparseCreateMatDescr(descrU)
ierr2 = ierr2 + ierr
ierr = cudaStreamCreate(stream) 
ierr2 = ierr2 + ierr
ierr = cublasSetStream(cublashandle,stream)
ierr2 = ierr2 + ierr
ierr = cusparseSetStream(cusparseHandle,stream) 
ierr2 = ierr2 + ierr
ierr = cusparseSetMatType2(descrA,CUBLAS_OP_N) 
ierr2 = ierr2 + ierr
ierr = cusparseSetMatIndexBase2(descrA,CUBLAS_OP_T) 
ierr2 = ierr2 + ierr
ierr = cusparseSetMatType2(descrM,CUBLAS_OP_N) 
ierr2 = ierr2 + ierr
ierr = cusparseSetMatIndexBase2(descrM,CUBLAS_OP_T) 
ierr2 = ierr2 + ierr
ierr = cusparseSetMatType2(descrL,CUBLAS_OP_N) 
ierr2 = ierr2 + ierr
ierr = cusparseSetMatIndexBase2(descrL,CUBLAS_OP_T) 
ierr2 = ierr2 + ierr
ierr = cusparseSetMatFillMode(descrL,CUBLAS_OP_N)
ierr2 = ierr2 + ierr
ierr = cusparseSetMatDiagType(descrL,CUBLAS_OP_T)
ierr2 = ierr2 + ierr
ierr = cusparseSetMatType2(descrU,CUBLAS_OP_N) 
ierr2 = ierr2 + ierr
ierr = cusparseSetMatIndexBase2(descrU,CUBLAS_OP_T) 
ierr2 = ierr2 + ierr
ierr = cusparseSetMatFillMode(descrU,CUBLAS_OP_T)
ierr2 = ierr2 + ierr
ierr = cusparseSetMatDiagType(descrU,CUBLAS_OP_N)
ierr2 = ierr2 + ierr
if (ierr2.ne.0) then
  write(*,'(A, I2)') 'Error during matrix setup ',ierr2
  stop
end if 
write(*,*) 'Allocating GPU memory'
ierr = cudaMalloc(devPtrX,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMalloc(devPtrF,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMalloc(devPtrR,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMalloc(devPtrRW,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMalloc(devPtrP,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMalloc(devPtrPW,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMalloc(devPtrS,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMalloc(devPtrT,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMalloc(devPtrV,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMalloc(devPtrAval,Annz_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMalloc(devPtrAcolsIndex,Annz_i_size)
ierr2 = ierr2 + ierr
ierr = cudaMalloc(devPtrArowsIndex,Arow1_i_size)
ierr2 = ierr2 + ierr
ierr = cudaMalloc(devPtrMval,Annz_d_size)
ierr2 = ierr2 + ierr
ierr = cudaDeviceSynchronize()
ierr2 = ierr2 + ierr
if (ierr2.ne.0) then
  write(*,'(A, I2)') 'Error during CUDA allocation: ',ierr2
  stop
end if 
write(*,*) 'Cleaning GPU memory'
ierr = cudaMemset(devPtrX,0,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMemset(devPtrF,0,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMemset(devPtrR,0,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMemset(devPtrRW,0,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMemset(devPtrP,0,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMemset(devPtrPW,0,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMemset(devPtrS,0,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMemset(devPtrT,0,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMemset(devPtrV,0,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMemset(devPtrAval,0,Annz_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMemset(devPtrAcolsIndex,0,Annz_i_size)
ierr2 = ierr2 + ierr
ierr = cudaMemset(devPtrArowsIndex,0,Arow1_i_size)
ierr2 = ierr2 + ierr
ierr = cudaMemset(devPtrMval,0,Annz_d_size)
ierr2 = ierr2 + ierr
ierr = cudaDeviceSynchronize()
ierr2 = ierr2 + ierr
if (ierr2.ne.0) then
  write(*,'(A, I3)') 'Error during CUDA memory cleaning : ',ierr2
  stop
end if 

! transfer memory over to GPU
write(*,*) 'Transferring memory to GPU'
ierr = cudaMemcpy(devPtrArowsIndex,ArowsIndex,Arow1_i_size,cudaMemcpyHostToDevice)
ierr2 = ierr2 + ierr
ierr = cudaMemcpy(devPtrAcolsIndex,AcolsIndex,Annz_i_size,cudaMemcpyHostToDevice)
ierr2 = ierr2 + ierr
ierr = cudaMemcpy(devPtrAval,Aval,Annz_d_size,cudaMemcpyHostToDevice)
ierr2 = ierr2 + ierr
ierr = cudaMemcpy(devPtrMval,devPtrAval,Annz_d_size,cudaMemcpyDeviceToDevice)
ierr2 = ierr2 + ierr
ierr = cudaMemcpy(devPtrX,h_x,Arow_d_size,cudaMemcpyHostToDevice)
ierr2 = ierr2 + ierr
ierr = cudaMemcpy(devPtrF,h_rhs,Arow_d_size,cudaMemcpyHostToDevice)
ierr2 = ierr2 + ierr
ierr = cudaDeviceSynchronize()
ierr2 = ierr2 + ierr
devPtrMrowsIndex=devPtrArowsIndex
devPtrMcolsIndex=devPtrAcolsIndex
if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during cuda memcpy ", ierr2
    stop
end if

write(*,*) 'Creating analysis for LU'
ierr = cusparseCreateCsrilu02Info(info_m)
ierr2 = ierr2 + ierr
ierr = cusparseCreateCsrsv2Info(info_l) 
ierr2 = ierr2 + ierr
ierr = cusparseCreateCsrsv2Info(info_u)
ierr2 = ierr2 + ierr
if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during LU info creation ", ierr2
    stop
end if

write(*,*) 'Querying memory space for buffer'
ierr = cusparseDcsrilu02_bufferSize(cusparseHandle,n,inz,descrM,devPtrMval,&
        devPtrMrowsIndex,devPtrMcolsIndex,info_m,buffer_m)
ierr2 = ierr2 + ierr
ierr = cusparseDcsrsv2_bufferSize(cusparseHandle,CUBLAS_OP_N,n,inz,descrL,&
        devPtrMval,devPtrMrowsIndex,devPtrMcolsIndex,info_l,buffer_l)
ierr2 = ierr2 + ierr
ierr = cusparseDcsrsv2_bufferSize(cusparseHandle,CUBLAS_OP_N,n,inz,descrU,&
        devPtrMval,devPtrMrowsIndex,devPtrMcolsIndex,info_u,buffer_u)
ierr2 = ierr2 + ierr
buffer_size = max(buffer_m,buffer_l,buffer_u)
write(*,*) 'Buffer sizes : ',buffer_m, buffer_l, buffer_u
write(*,*) 'Max buffer size : ',buffer_size
if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during buffer space calculation ", ierr2
    stop
end if

write(*,*) 'Allocating buffer'
buffer_malloc=buffer_size*1
write(*,*) 'Buffer malloc size : ',buffer_malloc
ierr = cudaMalloc(buffer_gpu,buffer_malloc)
ierr2 = ierr2 + ierr
if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during buffer space allocation ", ierr2
    stop
end if

write(*,*) 'Analysis of L,U,M'
ierr = cusparseDcsrilu02_analysis(cusparseHandle,n,inz,descrM,devPtrMval,&
        devPtrMrowsIndex,devPtrMcolsIndex,info_m,CUBLAS_OP_T,buffer_gpu)
ierr2 = ierr2 + ierr
if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during csrilu02_analysis ", ierr2
    stop
end if
ierr = cusparseXcsrilu02_zeroPivot(cusparseHandle,info_m,bad_posPtr)
ierr2 = ierr2 + ierr
if (bad_pos .ne. -1 ) then
    write (*, '(A, I2)') " Diagonal element is missing: ", bad_pos
end if
if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during zero point ", ierr2
    stop
end if
ierr = cusparseDcsrsv2_analysis(cusparseHandle,CUBLAS_OP_N,n,inz,descrL,&
        devPtrMval,devPtrMrowsIndex,devPtrMcolsIndex,info_l,CUBLAS_OP_T,&
        buffer_gpu)
ierr2 = ierr2 + ierr
ierr = cusparseDcsrsv2_analysis(cusparseHandle,CUBLAS_OP_N,n,inz,descrU,&
        devPtrMval,devPtrMrowsIndex,devPtrMcolsIndex,info_u,CUBLAS_OP_T,&
        buffer_gpu)
ierr2 = ierr2 + ierr
if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during csrsv2_analysis ", ierr2
    stop
end if

write(*,*) 'Begin calculation for M=L*U'
ierr = cusparseDcsrilu02(cusparseHandle,n,inz,descrM,devPtrMval,&
        devPtrMrowsIndex,devPtrMcolsIndex,info_m,CUBLAS_OP_T,buffer_gpu)
ierr2 = ierr2 + ierr
if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during csrilu02 ", ierr2
    stop
end if
ierr = cusparseXcsrilu02_zeroPivot(cusparseHandle,info_m,bad_posPtr)
ierr2 = ierr2 + ierr
if (bad_pos .ne. -1 ) then
    write (*, '(A, I2)') " Diagonal element U is zero: ", bad_pos
end if
if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during zero point ", ierr2
    stop
end if


! Compute the initial residual
write(*,*) 'Computing initial residual'
ierr = cusparseDcsrmv(cusparseHandle,CUBLAS_OP_N,n,n,inz,one,descrA,devPtrAval,devPtrArowsIndex,devPtrAcolsIndex,devPtrX,zero,devPtrR)
ierr2 = ierr2 + ierr
ierr = cudaDeviceSynchronize()
ierr2 = ierr2 + ierr
ierr = cublasDscal(cublashandle,n,minus_one,devPtrR,1)
ierr2 = ierr2 + ierr
ierr = cublasDaxpy(cublashandle,n,one,devPtrF,1,devPtrR,1)
ierr2 = ierr2 + ierr
ierr = cublasDcopy(cublashandle,n,devPtrR,1,devPtrRW,1)
ierr2 = ierr2 + ierr
ierr = cublasDcopy(cublashandle,n,devPtrR,1,devPtrP,1)
ierr2 = ierr2 + ierr
ierr = cublasDnrm2(cublashandle,n,devPtrR,1,normPtr)
ierr2 = ierr2 + ierr
ierr = cudaDeviceSynchronize()
ierr2 = ierr2 + ierr
if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during iteration setup ", ierr2
    stop
end if
inum_it=1
do 12 ii = 1,imax_num_it
  rhop=rho
  ierr = cublasDdot(cublashandle,n,devPtrRW,1,devPtrR,1,rhoPtr)
  if (ii.gt.1) then
    beta=(rho/rhop)*(alpha/omega)
    negomega=-(omega)
    ierr = cublasDaxpy(cublashandle,n,negomega,devPtrV,1,devPtrP,1)
    ierr2 = ierr2 + ierr
    ierr = cublasDscal(cublashandle,n,beta,devPtrP,1)
    ierr2 = ierr2 + ierr
    ierr = cublasDaxpy(cublashandle,n,one,devPtrR,1,devPtrP,1)
    ierr2 = ierr2 + ierr
    if (ierr2 .ne. 0 ) then
      write (*, '(A, I2)') " Error on sub iteration ", ierr2
      stop
    end if
  end if
  write(*,*) 'Preconditioning solve step 1'
  ierr = cudaDeviceSynchronize()
  ierr2 = ierr2 + ierr
  ierr = cusparseSetMatFillMode(descrM,CUBLAS_OP_N)
  ierr2 = ierr2 + ierr
  ierr = cusparseSetMatDiagType(descrM,CUBLAS_OP_T)
  ierr2 = ierr2 + ierr
  ierr = cusparseDcsrsv2_solve(cusparseHandle,CUBLAS_OP_N,n,inz,one,descrL,&
          devPtrMval,devPtrMrowsIndex,devPtrMcolsIndex,info_l,devPtrP,devPtrT,&
          CUBLAS_OP_T,buffer_gpu)
  ierr2 = ierr2 + ierr
  ierr = cudaDeviceSynchronize()
  ierr2 = ierr2 + ierr
  ierr = cusparseSetMatFillMode(descrM,CUBLAS_OP_T)
  ierr2 = ierr2 + ierr
  ierr = cusparseSetMatDiagType(descrM,CUBLAS_OP_N)
  ierr2 = ierr2 + ierr
  ierr = cusparseDcsrsv2_solve(cusparseHandle,CUBLAS_OP_N,n,inz,one,descrU,&
          devPtrMval,devPtrMrowsIndex,devPtrMcolsIndex,info_u,devPtrT,devPtrPW,&
          CUBLAS_OP_T,buffer_gpu)
  ierr2 = ierr2 + ierr
  ierr = cudaDeviceSynchronize()
  ierr2 = ierr2 + ierr
  if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during preconditioning solve 1 ", ierr2
    stop
  end if
  write(*,*) ' Checking solution to stage 1'
  ierr = cusparseDcsrmv(cusparseHandle,CUBLAS_OP_N,n,n,inz,one,descrA,&
          devPtrAval,devPtrArowsIndex,devPtrAcolsIndex,devPtrPW,zero,devPtrV)
  ierr2 = ierr2 + ierr
  ierr = cudaDeviceSynchronize()
  ierr2 = ierr2 + ierr
  ierr = cublasDdot(cublashandle,n,devPtrRW,1,devPtrV,1,tempPtr)
  ierr2 = ierr2 + ierr
  alpha=rho/temp
  negalpha=-(alpha)
  ierr = cublasDaxpy(cublashandle,n,negalpha,devPtrV,1,devPtrR,1)
  ierr2 = ierr2 + ierr
  ierr = cublasDaxpy(cublashandle,n,alpha,devPtrPW,1,devPtrX,1)
  ierr2 = ierr2 + ierr
  ierr = cublasDnrm2(cublashandle,n,devPtrR,1,normPtr)
  ierr2 = ierr2 + ierr
  if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during stage 1 solution ", ierr2
    stop
  end if
  if (norm_val.lt.(tol*norm_val0)) then
    isol_found=1
    go to 13
  end if
  write(*,*) 'Preconditioning solve step 2'
  ierr = cudaDeviceSynchronize()
  ierr2 = ierr2 + ierr
  ierr = cusparseSetMatFillMode(descrM,CUBLAS_OP_N)
  ierr2 = ierr2 + ierr
  ierr = cusparseSetMatDiagType(descrM,CUBLAS_OP_T)
  ierr2 = ierr2 + ierr
  ierr = cusparseDcsrsv2_solve(cusparseHandle,CUBLAS_OP_N,n,inz,one,descrL,&
          devPtrMval,devPtrMrowsIndex,devPtrMcolsIndex,info_l,devPtrR,devPtrT,&
          CUBLAS_OP_T,buffer_gpu)
  ierr2 = ierr2 + ierr
  ierr = cudaDeviceSynchronize()
  ierr2 = ierr2 + ierr
  ierr = cusparseSetMatFillMode(descrM,CUBLAS_OP_T)
  ierr2 = ierr2 + ierr
  ierr = cusparseSetMatDiagType(descrM,CUBLAS_OP_N)
  ierr2 = ierr2 + ierr
  ierr = cusparseDcsrsv2_solve(cusparseHandle,CUBLAS_OP_N,n,inz,one,descrU,&
          devPtrMval,devPtrMrowsIndex,devPtrMcolsIndex,info_l,devPtrT,devPtrS,&
          CUBLAS_OP_T,buffer_gpu)
  ierr2 = ierr2 + ierr
  ierr = cudaDeviceSynchronize()
  ierr2 = ierr2 + ierr
  if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during preconditioning solve 2 ", ierr2
    stop
  end if
  write(*,*) ' Checking solution to stage 2'
  ierr = cusparseDcsrmv(cusparseHandle,CUBLAS_OP_N,n,n,inz,one,descrA,&
          devPtrAval,devPtrArowsIndex,devPtrAcolsIndex,devPtrS,zero,devPtrT)
  ierr2 = ierr2 + ierr
  ierr = cudaDeviceSynchronize()
  ierr2 = ierr2 + ierr
  ierr = cublasDdot(cublashandle,n,devPtrT,1,devPtrR,1,tempPtr)
  ierr2 = ierr2 + ierr
  ierr = cublasDdot(cublashandle,n,devPtrT,1,devPtrT,1,tempPtr2)
  ierr2 = ierr2 + ierr
  omega=temp/temp2
  negomega=-(omega)
  ierr = cublasDaxpy(cublashandle,n,omega,devPtrS,1,devPtrX,1)
  ierr2 = ierr2 + ierr
  ierr = cublasDaxpy(cublashandle,n,negomega,devPtrT,1,devPtrR,1)
  ierr2 = ierr2 + ierr
  ierr = cublasDnrm2(cublashandle,n,devPtrR,1,normPtr)
  ierr2 = ierr2 + ierr
  if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during stage 2 solution ", ierr2
    stop
  end if
  if (norm_val.lt.(tol*norm_val0)) then
    isol_found=1
    go to 13
  end if
12 continue

13 ierr = cudaDeviceSynchronize()
ierr2 = ierr2 + ierr
if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error synchronizing after iterations ", ierr2
    stop
end if

if (isol_found.eq.1) then
  write(*,'(A, I2)') 'Solution found within tolerance after iter: ',ii
else if (ii.eq.imax_num_it) then
  write(*,'(A, I2)') 'Did not achieve convergence within max iterations',imax_num_it
else
  write(*,*) 'Should not get here'
end if

write(*,*) 'Copy solution from GPU to CPU'
ierr = cudaMemcpy(h_x,devPtrX,Acol_d_size,cudaMemcpyDeviceToHost)
if (ierr .ne. 0 ) then
    write (*, '(A, I2)') " cudaMemcpy 6 error: ", ierr
    stop
end if

ierr = cudaFree(devPtrArowsIndex)
ierr2 = ierr2 + ierr
ierr = cudaFree(devPtrAcolsIndex)
ierr2 = ierr2 + ierr
ierr = cudaFree(devPtrAval)
ierr2 = ierr2 + ierr
ierr = cudaFree(devPtrMrowsIndex)
ierr2 = ierr2 + ierr
ierr = cudaFree(devPtrMcolsIndex)
ierr2 = ierr2 + ierr
ierr = cudaFree(devPtrMval)
ierr2 = ierr2 + ierr
ierr = cudaFree(devPtrX)
ierr2 = ierr2 + ierr
ierr = cudaFree(devPtrF)
ierr2 = ierr2 + ierr
ierr = cudaFree(devPtrR)
ierr2 = ierr2 + ierr
ierr = cudaFree(devPtrRW)
ierr2 = ierr2 + ierr
ierr = cudaFree(devPtrP)
ierr2 = ierr2 + ierr
ierr = cudaFree(devPtrPW)
ierr2 = ierr2 + ierr
ierr = cudaFree(devPtrS)
ierr2 = ierr2 + ierr
ierr = cudaFree(devPtrT)
ierr2 = ierr2 + ierr
ierr = cudaFree(devPtrV)
ierr2 = ierr2 + ierr
if (ierr2.ne.0) then
  write(*,'(A, I2)') 'Error during cudafree: ',ierr2
  stop
end if 

ierr = cublasDestroy(cublashandle)
ierr2 = ierr2 + ierr
ierr = cusparseDestroy(cusparseHandle)
ierr2 = ierr2 + ierr
ierr = cudaStreamDestroy(stream)
ierr2 = ierr2 + ierr
ierr = cusparseDestroyMatDescr(descrA)
ierr2 = ierr2 + ierr
ierr = cusparseDestroyMatDescr(descrM)
ierr2 = ierr2 + ierr
ierr = cusparseDestroySolveAnalysisInfo(info_l)
ierr2 = ierr2 + ierr
ierr = cusparseDestroySolveAnalysisInfo(info_u)
ierr2 = ierr2 + ierr
if (ierr2.ne.0) then
  write(*,'(A, I2)') 'Error during cuda handle destruction: ',ierr2
  stop
end if 

return
end subroutine cuda_BiCGStab2
!
!=========================================================
subroutine cuda_BiCGStab(n,rhs,x,inz,ib,jb,b, tol)
!=========================================================
use iso_c_binding
use cuda_cusolve_map
implicit none
integer n, inz
double precision x(n), rhs(n), b(inz)
target rhs,b,x
integer ib(n+1),jb(inz)
target ib,jb
integer ii,ierr,ierr2,inum_it,imax_num_it,isol_found
double precision tol,norm_val,norm_val0
target norm_val,norm_val0
double precision rho,rhop,beta,alpha,negalpha
target rho
double precision omega,negomega,temp,temp2
target temp,temp2

integer, parameter :: dp = kind(1.d0)

type(c_ptr) :: cublashandle !cusolver_Hndl
type(c_ptr) :: cusparseHandle
type(c_ptr) :: stream
type(c_ptr) :: descrA
type(c_ptr) :: descrM
type(c_ptr) :: info_l
type(c_ptr) :: info_u
type(c_ptr) :: ArowsIndex
type(c_ptr) :: AcolsIndex
type(c_ptr) :: Aval
type(c_ptr) :: h_x  
type(c_ptr) :: h_rhs
type(c_ptr) :: normPtr
type(c_ptr) :: normPtr0
type(c_ptr) :: rhoPtr
type(c_ptr) :: tempPtr
type(c_ptr) :: tempPtr2

! -------------------- pointers to device memory    
type(c_ptr) :: devPtrArowsIndex
type(c_ptr) :: devPtrAcolsIndex
type(c_ptr) :: devPtrAval
type(c_ptr) :: devPtrMrowsIndex
type(c_ptr) :: devPtrMcolsIndex
type(c_ptr) :: devPtrMval
type(c_ptr) :: devPtrX
type(c_ptr) :: devPtrF
type(c_ptr) :: devPtrR
type(c_ptr) :: devPtrRW
type(c_ptr) :: devPtrP
type(c_ptr) :: devPtrPW
type(c_ptr) :: devPtrS
type(c_ptr) :: devPtrT
type(c_ptr) :: devPtrV

integer*8 Arow1_i_size,Arow_d_size,Acol_d_size,Annz_i_size,Annz_d_size
integer(c_int) matrixM,matrixN, nnz, mNNZ
integer(c_int) matrixSizeAval, matrixSizeAcolsIndex, matrixSizeArowsIndex, mSizeAval, mSizeAcolsIndex, mSizeArowsIndex
integer(c_int) arraySizeX, arraySizeF, arraySizeR, arraySizeRW, arraySizeP,  arraySizePW, arraySizeS, arraySizeT, arraySizeV
 
integer*8 cudaMemcpyDeviceToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToDevice
integer*4 CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_TRI
parameter (cudaMemcpyHostToDevice=1)
parameter (cudaMemcpyDeviceToHost=2)
parameter (cudaMemcpyDeviceToDevice=3)
parameter (CUBLAS_OP_N=0)
parameter (CUBLAS_OP_T=1)
parameter (CUBLAS_OP_TRI=3)

! the constants are used in residual evaluation, r = b - A*x
real(kind=dp) minus_one
parameter (minus_one=-1.0d0)
real(kind=dp) one
parameter (one=1.0d0)
real(kind=dp) zero
parameter (zero=1.0d0)




ierr2 = 0
isol_found=0
imax_num_it =  10
rho = 0.0d0
norm_val = 1.d5
norm_val0 = 1.d5

! write(*,*) 'Overwriting solution space'
! n=5
! inz = 13

! b(1) = 1.0d0; b(2)=1.0d0; b(3)=5.0d0; b(4)=2.0d0
! b(5) = 1.0d0; b(6)=3.0d0; b(7)=2.0d0; b(8)=1.0d0
! b(9) = 6.0d0; b(10)=3.0d0; b(11)=1.0d0; b(12)=2.0d0
! b(13) = 1.0d0

! rhs(1)=1.0d0;rhs(2)=2.0d0;rhs(3)=1.0d0;rhs(4)=3.0d0;rhs(5)=0.0d0

! ib(1) = 1;ib(2) = 5;ib(3) = 7
! ib(4) = 9;ib(5) = 12;ib(6) = 14

! jb(1) = 1; jb(2)  = 2; jb(3) = 4; jb(4) = 5
! jb(5) = 2; jb(6)  = 3; jb(7) = 2; jb(8) = 3
! jb(9) = 1; jb(10) = 3;jb(11) = 4;jb(12) = 4
! jb(13) = 5

!solution
! x(1)=0.08
! x(2)=0.2
! x(3)=0.6
! x(4)=0.72
! x(5)=-1.44

! define pointers to host memory
ArowsIndex = c_loc(ib)
AcolsIndex = c_loc(jb)
Aval = c_loc(b)
h_x  = c_loc(x)  ! x = A \ b
h_rhs  = c_loc(rhs)  ! b = ones(m,1)
normPtr = c_loc(norm_val)
normPtr0 = c_loc(norm_val0)
rhoPtr = c_loc(rho)
tempPtr = c_loc(temp)
tempPtr2 = c_loc(temp2)

Arow1_i_size=sizeof(ib(1:n+1))
Arow_d_size=sizeof(rhs(1:n))
Acol_d_size=sizeof(x(1:n))
Annz_i_size=sizeof(jb(1:inz))
Annz_d_size=sizeof(b(1:inz))

! Define the CUDA stream and matrix parameters
ierr = cublasCreate(cublashandle)
ierr2 = ierr2 + ierr
ierr = cusparseCreate(cusparseHandle)
ierr2 = ierr2 + ierr
ierr = cusparseCreateMatDescr(descrA)
ierr2 = ierr2 + ierr
ierr = cusparseCreateMatDescr(descrM)
ierr2 = ierr2 + ierr
ierr = cudaStreamCreate(stream) 
ierr2 = ierr2 + ierr
ierr = cublasSetStream(cublashandle,stream)
ierr2 = ierr2 + ierr
ierr = cusparseSetStream(cusparseHandle,stream) 
ierr2 = ierr2 + ierr
ierr = cusparseSetMatType2(descrA,CUBLAS_OP_N) 
ierr2 = ierr2 + ierr
ierr = cusparseSetMatIndexBase2(descrA,CUBLAS_OP_T) 
ierr2 = ierr2 + ierr
ierr = cusparseSetMatType2(descrM,CUBLAS_OP_N) 
ierr2 = ierr2 + ierr
ierr = cusparseSetMatIndexBase2(descrM,CUBLAS_OP_T) 
ierr2 = ierr2 + ierr
if (ierr2.ne.0) then
  write(*,'(A, I2)') 'Error during matrix setup ',ierr2
  stop
end if 
write(*,*) 'Allocating GPU memory'
ierr = cudaMalloc(devPtrX,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMalloc(devPtrF,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMalloc(devPtrR,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMalloc(devPtrRW,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMalloc(devPtrP,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMalloc(devPtrPW,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMalloc(devPtrS,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMalloc(devPtrT,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMalloc(devPtrV,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMalloc(devPtrAval,Annz_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMalloc(devPtrAcolsIndex,Annz_i_size)
ierr2 = ierr2 + ierr
ierr = cudaMalloc(devPtrArowsIndex,Arow1_i_size)
ierr2 = ierr2 + ierr
ierr = cudaMalloc(devPtrMval,Annz_d_size)
ierr2 = ierr2 + ierr
ierr = cudaDeviceSynchronize()
ierr2 = ierr2 + ierr
if (ierr2.ne.0) then
  write(*,'(A, I2)') 'Error during CUDA allocation: ',ierr2
  stop
end if 
write(*,*) 'Cleaning GPU memory'
ierr = cudaMemset(devPtrX,0,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMemset(devPtrF,0,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMemset(devPtrR,0,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMemset(devPtrRW,0,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMemset(devPtrP,0,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMemset(devPtrPW,0,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMemset(devPtrS,0,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMemset(devPtrT,0,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMemset(devPtrV,0,Arow_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMemset(devPtrAval,0,Annz_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMemset(devPtrAcolsIndex,0,Annz_i_size)
ierr2 = ierr2 + ierr
ierr = cudaMemset(devPtrArowsIndex,0,Arow1_i_size)
ierr2 = ierr2 + ierr
ierr = cudaMemset(devPtrMval,0,Annz_d_size)
ierr2 = ierr2 + ierr
ierr = cudaDeviceSynchronize()
ierr2 = ierr2 + ierr
if (ierr2.ne.0) then
  write(*,'(A, I3)') 'Error during CUDA memory cleaning : ',ierr2
  stop
end if 

! transfer memory over to GPU
write(*,*) 'Transferring memory to GPU'
ierr = cudaMemcpy(devPtrArowsIndex,ArowsIndex,Arow1_i_size,cudaMemcpyHostToDevice)
ierr2 = ierr2 + ierr
ierr = cudaMemcpy(devPtrAcolsIndex,AcolsIndex,Annz_i_size,cudaMemcpyHostToDevice)
ierr2 = ierr2 + ierr
ierr = cudaMemcpy(devPtrAval,Aval,Annz_d_size,cudaMemcpyHostToDevice)
ierr2 = ierr2 + ierr
ierr = cudaMemcpy(devPtrMval,devPtrAval,Annz_d_size,cudaMemcpyDeviceToDevice)
ierr2 = ierr2 + ierr
ierr = cudaMemcpy(devPtrX,h_x,Arow_d_size,cudaMemcpyHostToDevice)
ierr2 = ierr2 + ierr
ierr = cudaMemcpy(devPtrF,h_rhs,Arow_d_size,cudaMemcpyHostToDevice)
ierr2 = ierr2 + ierr
ierr = cudaDeviceSynchronize()
ierr2 = ierr2 + ierr
if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during cuda memcpy ", ierr2
    stop
end if

write(*,*) 'Creating analysis for LU'
ierr = cusparseCreateSolveAnalysisInfo(info_l)
ierr2 = ierr2 + ierr
ierr = cusparseCreateSolveAnalysisInfo(info_u)
ierr2 = ierr2 + ierr
if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during LU analysis creation ", ierr2
    stop
end if

write(*,*) 'Analyzing L of LU'
!write(*,*) 'mat type : ',cusparseGetMatType(descrM)
!write(*,*) 'mat index base : ',cusparseGetMatIndexBase(descrM)
ierr = cusparseSetMatFillMode(descrM,CUBLAS_OP_N)
ierr2 = ierr2 + ierr
ierr = cusparseSetMatDiagType(descrM,CUBLAS_OP_T)
ierr2 = ierr2 + ierr
!if (ierr2 .ne. 0 ) then
!    write (*, '(A, I2)') " Error during L of LU analyzing sub1 ", ierr2
!    stop
!end if
!write(*,*) 'mat diag type : ',cusparseGetMatDiagType(descrM)
!write(*,*) 'mat fill type : ',cusparseGetMatFillMode(descrM)
ierr = cudaDeviceSynchronize()
ierr2 = ierr2 + ierr
ierr = cusparseDcsrsv_analysis(cusparseHandle,CUBLAS_OP_N,n,inz,descrM,devPtrAval,&
                               devPtrArowsIndex,devPtrAcolsIndex,info_l)
ierr2 = ierr2 + ierr
if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during L of LU analyzing sub2 ", ierr2
    stop
end if
ierr = cudaDeviceSynchronize()
ierr2 = ierr2 + ierr
if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during L of LU analyzing ", ierr2
    stop
end if

write(*,*) 'Analyzing U of LU'
ierr = cusparseSetMatFillMode(descrM,CUBLAS_OP_T)
ierr2 = ierr2 + ierr
ierr = cusparseSetMatDiagType(descrM,CUBLAS_OP_N)
ierr2 = ierr2 + ierr
ierr = cusparseDcsrsv_analysis(cusparseHandle,CUBLAS_OP_N,n,inz,descrM,devPtrAval,&
                               devPtrArowsIndex,devPtrAcolsIndex,info_u)
ierr2 = ierr2 + ierr
if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during U of LU analyzing sub2 ", ierr2
    stop
end if
ierr = cudaDeviceSynchronize()
ierr2 = ierr2 + ierr
if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during U of LU analyzing ", ierr2
    stop
end if

write(*,*) 'Calculating LU'
devPtrMrowsIndex=devPtrArowsIndex
devPtrMcolsIndex=devPtrAcolsIndex
ierr = cusparseDcsrilu0(cusparseHandle,CUBLAS_OP_N,n,descrA,devPtrMval,devPtrArowsIndex,devPtrAcolsIndex,info_l)
ierr2 = ierr2 + ierr
ierr = cudaDeviceSynchronize()
ierr2 = ierr2 + ierr
if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during LU calculation ", ierr2
    stop
end if

! Compute the initial residual
write(*,*) 'Computing initial residual'
ierr = cusparseDcsrmv(cusparseHandle,CUBLAS_OP_N,n,n,inz,one,descrA,devPtrAval,devPtrArowsIndex,devPtrAcolsIndex,devPtrX,zero,devPtrR)
ierr2 = ierr2 + ierr
ierr = cudaDeviceSynchronize()
ierr2 = ierr2 + ierr
ierr = cublasDscal(cublashandle,n,minus_one,devPtrR,1)
ierr2 = ierr2 + ierr
if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during Dscal 2 ", ierr2
    stop
end if
ierr = cublasDaxpy(cublashandle,n,one,devPtrF,1,devPtrR,1)
ierr2 = ierr2 + ierr
if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during Daxpy ", ierr2
    stop
end if
ierr = cublasDcopy(cublashandle,n,devPtrR,1,devPtrRW,1)
ierr2 = ierr2 + ierr
if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during Dcopy 1 ", ierr2
    stop
end if
ierr = cublasDcopy(cublashandle,n,devPtrR,1,devPtrP,1)
ierr2 = ierr2 + ierr
if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during Dcopy 2 ", ierr2
    stop
end if
ierr = cublasDnrm2(cublashandle,n,devPtrR,1,normPtr)
ierr2 = ierr2 + ierr
if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during dnrm2 ", ierr2
    stop
end if
ierr = cudaDeviceSynchronize()
ierr2 = ierr2 + ierr
if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during iteration setup ", ierr2
    stop
end if
inum_it=1
do 12 ii = 1,imax_num_it
  rhop=rho
  ierr = cublasDdot(cublashandle,n,devPtrRW,1,devPtrR,1,rhoPtr)
  if (ii.gt.1) then
    beta=(rho/rhop)*(alpha/omega)
    negomega=-(omega)
    ierr = cublasDaxpy(cublashandle,n,negomega,devPtrV,1,devPtrP,1)
    ierr2 = ierr2 + ierr
    ierr = cublasDscal(cublashandle,n,beta,devPtrP,1)
    ierr2 = ierr2 + ierr
    ierr = cublasDaxpy(cublashandle,n,one,devPtrR,1,devPtrP,1)
    ierr2 = ierr2 + ierr
    if (ierr2 .ne. 0 ) then
      write (*, '(A, I2)') " Error on sub iteration ", ierr2
      stop
    end if
  end if
  write(*,*) 'Preconditioning solve step 1'
  ierr = cudaDeviceSynchronize()
  ierr2 = ierr2 + ierr
  ierr = cusparseSetMatFillMode(descrM,CUBLAS_OP_N)
  ierr2 = ierr2 + ierr
  ierr = cusparseSetMatDiagType(descrM,CUBLAS_OP_T)
  ierr2 = ierr2 + ierr
  ierr = cusparseDcsrsv_solve(cusparseHandle,CUBLAS_OP_N,n,one,descrM,&
          devPtrMval,devPtrMrowsIndex,devPtrMcolsIndex,info_l,devPtrP,devPtrT)
  ierr2 = ierr2 + ierr
  ierr = cudaDeviceSynchronize()
  ierr2 = ierr2 + ierr
  ierr = cusparseSetMatFillMode(descrM,CUBLAS_OP_T)
  ierr2 = ierr2 + ierr
  ierr = cusparseSetMatDiagType(descrM,CUBLAS_OP_N)
  ierr2 = ierr2 + ierr
  ierr = cusparseDcsrsv_solve(cusparseHandle,CUBLAS_OP_N,n,one,descrM,&
          devPtrMval,devPtrMrowsIndex,devPtrMcolsIndex,info_l,devPtrT,devPtrPW)
  ierr2 = ierr2 + ierr
  ierr = cudaDeviceSynchronize()
  ierr2 = ierr2 + ierr
  if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during preconditioning solve 1 ", ierr2
    stop
  end if
  write(*,*) ' Checking solution to stage 1'
  ierr = cusparseDcsrmv(cusparseHandle,CUBLAS_OP_N,n,n,inz,one,descrA,&
          devPtrAval,devPtrArowsIndex,devPtrAcolsIndex,devPtrPW,zero,devPtrV)
  ierr2 = ierr2 + ierr
  ierr = cudaDeviceSynchronize()
  ierr2 = ierr2 + ierr
  ierr = cublasDdot(cublashandle,n,devPtrRW,1,devPtrV,1,tempPtr)
  ierr2 = ierr2 + ierr
  alpha=rho/temp
  negalpha=-(alpha)
  ierr = cublasDaxpy(cublashandle,n,negalpha,devPtrV,1,devPtrR,1)
  ierr2 = ierr2 + ierr
  ierr = cublasDaxpy(cublashandle,n,alpha,devPtrPW,1,devPtrX,1)
  ierr2 = ierr2 + ierr
  ierr = cublasDnrm2(cublashandle,n,devPtrR,1,normPtr)
  ierr2 = ierr2 + ierr
  if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during stage 1 solution ", ierr2
    stop
  end if
  if (norm_val.lt.(tol*norm_val0)) then
    isol_found=1
    go to 13
  end if
  write(*,*) 'Preconditioning solve step 2'
  ierr = cudaDeviceSynchronize()
  ierr2 = ierr2 + ierr
  ierr = cusparseSetMatFillMode(descrM,CUBLAS_OP_N)
  ierr2 = ierr2 + ierr
  ierr = cusparseSetMatDiagType(descrM,CUBLAS_OP_T)
  ierr2 = ierr2 + ierr
  ierr = cusparseDcsrsv_solve(cusparseHandle,CUBLAS_OP_N,n,one,descrM,&
          devPtrMval,devPtrMrowsIndex,devPtrMcolsIndex,info_l,devPtrR,devPtrT)
  ierr2 = ierr2 + ierr
  ierr = cudaDeviceSynchronize()
  ierr2 = ierr2 + ierr
  ierr = cusparseSetMatFillMode(descrM,CUBLAS_OP_T)
  ierr2 = ierr2 + ierr
  ierr = cusparseSetMatDiagType(descrM,CUBLAS_OP_N)
  ierr2 = ierr2 + ierr
  ierr = cusparseDcsrsv_solve(cusparseHandle,CUBLAS_OP_N,n,one,descrM,&
          devPtrMval,devPtrMrowsIndex,devPtrMcolsIndex,info_l,devPtrT,devPtrS)
  ierr2 = ierr2 + ierr
  ierr = cudaDeviceSynchronize()
  ierr2 = ierr2 + ierr
  if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during preconditioning solve 2 ", ierr2
    stop
  end if
  write(*,*) ' Checking solution to stage 2'
  ierr = cusparseDcsrmv(cusparseHandle,CUBLAS_OP_N,n,n,inz,one,descrA,&
          devPtrAval,devPtrArowsIndex,devPtrAcolsIndex,devPtrS,zero,devPtrT)
  ierr2 = ierr2 + ierr
  ierr = cudaDeviceSynchronize()
  ierr2 = ierr2 + ierr
  ierr = cublasDdot(cublashandle,n,devPtrT,1,devPtrR,1,tempPtr)
  ierr2 = ierr2 + ierr
  ierr = cublasDdot(cublashandle,n,devPtrT,1,devPtrT,1,tempPtr2)
  ierr2 = ierr2 + ierr
  omega=temp/temp2
  negomega=-(omega)
  ierr = cublasDaxpy(cublashandle,n,omega,devPtrS,1,devPtrX,1)
  ierr2 = ierr2 + ierr
  ierr = cublasDaxpy(cublashandle,n,negomega,devPtrT,1,devPtrR,1)
  ierr2 = ierr2 + ierr
  ierr = cublasDnrm2(cublashandle,n,devPtrR,1,normPtr)
  ierr2 = ierr2 + ierr
  if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during stage 2 solution ", ierr2
    stop
  end if
  if (norm_val.lt.(tol*norm_val0)) then
    isol_found=1
    go to 13
  end if
12 continue

13 ierr = cudaDeviceSynchronize()
ierr2 = ierr2 + ierr
if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error synchronizing after iterations ", ierr2
    stop
end if

if (isol_found.eq.1) then
  write(*,'(A, I2)') 'Solution found within tolerance after iter: ',ii
else if (ii.eq.imax_num_it) then
  write(*,'(A, I2)') 'Did not achieve convergence within max iterations',imax_num_it
else
  write(*,*) 'Should not get here'
end if

write(*,*) 'Copy solution from GPU to CPU'
ierr = cudaMemcpy(h_x,devPtrX,Acol_d_size,cudaMemcpyDeviceToHost)
if (ierr .ne. 0 ) then
    write (*, '(A, I2)') " cudaMemcpy 6 error: ", ierr
    stop
end if

ierr = cudaFree(devPtrArowsIndex)
ierr2 = ierr2 + ierr
ierr = cudaFree(devPtrAcolsIndex)
ierr2 = ierr2 + ierr
ierr = cudaFree(devPtrAval)
ierr2 = ierr2 + ierr
ierr = cudaFree(devPtrMrowsIndex)
ierr2 = ierr2 + ierr
ierr = cudaFree(devPtrMcolsIndex)
ierr2 = ierr2 + ierr
ierr = cudaFree(devPtrMval)
ierr2 = ierr2 + ierr
ierr = cudaFree(devPtrX)
ierr2 = ierr2 + ierr
ierr = cudaFree(devPtrF)
ierr2 = ierr2 + ierr
ierr = cudaFree(devPtrR)
ierr2 = ierr2 + ierr
ierr = cudaFree(devPtrRW)
ierr2 = ierr2 + ierr
ierr = cudaFree(devPtrP)
ierr2 = ierr2 + ierr
ierr = cudaFree(devPtrPW)
ierr2 = ierr2 + ierr
ierr = cudaFree(devPtrS)
ierr2 = ierr2 + ierr
ierr = cudaFree(devPtrT)
ierr2 = ierr2 + ierr
ierr = cudaFree(devPtrV)
ierr2 = ierr2 + ierr
if (ierr2.ne.0) then
  write(*,'(A, I2)') 'Error during cudafree: ',ierr2
  stop
end if 

ierr = cublasDestroy(cublashandle)
ierr2 = ierr2 + ierr
ierr = cusparseDestroy(cusparseHandle)
ierr2 = ierr2 + ierr
ierr = cudaStreamDestroy(stream)
ierr2 = ierr2 + ierr
ierr = cusparseDestroyMatDescr(descrA)
ierr2 = ierr2 + ierr
ierr = cusparseDestroyMatDescr(descrM)
ierr2 = ierr2 + ierr
ierr = cusparseDestroySolveAnalysisInfo(info_l)
ierr2 = ierr2 + ierr
ierr = cusparseDestroySolveAnalysisInfo(info_u)
ierr2 = ierr2 + ierr
if (ierr2.ne.0) then
  write(*,'(A, I2)') 'Error during cuda handle destruction: ',ierr2
  stop
end if 

return
end subroutine cuda_BiCGStab
!
!=========================================================
subroutine cuda_sparse_solve_qr(n,rhs,x,inz,ib,jb,b, tol)
!=========================================================
use iso_c_binding
use cuda_cusolve_map
implicit none
integer n, inz,   ierr,ierr2
double precision x(n), rhs(n), b(inz)
integer ib(n+1),jb(inz)
double precision residual(n)
integer ii,icpu_check
integer reorder_ib(n+1),reorder_jb(inz),reorder_map(inz)
integer permute_vec(n)
double precision reorder_rhs(n), reorder_b(inz)
double precision tol

integer(c_int) ireorder
integer(c_int) iresid
integer(c_int) singularity 

integer, parameter :: dp = kind(1.d0)

type(c_ptr) :: handle !cusolver_Hndl
type(c_ptr) :: cusparseHandle
type(c_ptr) :: stream
type(c_ptr) :: descrA
type(c_ptr) :: h_csrRowPtrA
type(c_ptr) :: h_csrColIndA
type(c_ptr) :: h_csrValA 
type(c_ptr) :: h_x  
type(c_ptr) :: h_b 
type(c_ptr) :: h_r 
! For reordering
integer(c_size_t) :: reorderWork
type(c_ptr) :: h_Q 
type(c_ptr) :: h_csrRowPtrB 
type(c_ptr) :: h_csrColIndB 
type(c_ptr) :: h_csrValB   
type(c_ptr) :: h_mapBfromA  
integer*1,allocatable :: buffer_space(:)
target buffer_space
integer*1 :: dummy_value
type(c_ptr) :: buffer_cpu 

! -------------------- pointers to device memory    
type(c_ptr) :: d_csrRowPtrA
type(c_ptr) :: d_csrColIndA
type(c_ptr) :: d_csrValA 
type(c_ptr) :: d_x    ! x = A \ b
type(c_ptr) :: d_b    ! a copy of h_b
type(c_ptr) :: d_r    ! r = b - A*x

integer*8 Arow1_i_size,Arow_d_size,Acol_d_size,Annz_i_size,Annz_d_size
integer*8 cudaMemcpyDeviceToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToDevice
integer*4 CUBLAS_OP_N, CUBLAS_OP_T
parameter (cudaMemcpyHostToDevice=1)
parameter (cudaMemcpyDeviceToHost=2)
parameter (cudaMemcpyDeviceToDevice=3)
parameter (CUBLAS_OP_N=0)
parameter (CUBLAS_OP_T=1)

! the constants are used in residual evaluation, r = b - A*x
real(kind=dp) minus_one
parameter (minus_one=-1.0d0)
real(kind=dp) one
parameter (one=1.0d0)


! write(*,*) 'Overwriting solution space'
! n=5
! inz = 13

! b(1) = 1.0d0; b(2)=1.0d0; b(3)=5.0d0; b(4)=2.0d0
! b(5) = 1.0d0; b(6)=3.0d0; b(7)=2.0d0; b(8)=1.0d0
! b(9) = 6.0d0; b(10)=3.0d0; b(11)=1.0d0; b(12)=2.0d0
! b(13) = 1.0d0

! rhs(1)=1.0d0;rhs(2)=2.0d0;rhs(3)=1.0d0;rhs(4)=3.0d0;rhs(5)=0.0d0

! ib(1) = 1;ib(2) = 5;ib(3) = 7
! ib(4) = 9;ib(5) = 12;ib(6) = 14

! jb(1) = 1; jb(2)  = 2; jb(3) = 4; jb(4) = 5
! jb(5) = 2; jb(6)  = 3; jb(7) = 2; jb(8) = 3
! jb(9) = 1; jb(10) = 3;jb(11) = 4;jb(12) = 4
! jb(13) = 5

!solution
! x(1)=0.08
! x(2)=0.2
! x(3)=0.6
! x(4)=0.72
! x(5)=-1.44

! write(*,*) n,inz

ireorder = 1
iresid = 0
ierr2 = 0
icpu_check = 0
tol=1.0d-12


! define pointers to hose memory
h_csrRowPtrA = c_loc(ib)
h_csrColIndA = c_loc(jb)
h_csrValA = c_loc(b)
h_x  = c_loc(x)  ! x = A \ b
h_b  = c_loc(rhs)  ! b = ones(m,1)
h_r  = c_loc(residual)  ! r = b - A*x

Arow1_i_size=sizeof(ib(1:n+1))
Arow_d_size=sizeof(rhs(1:n))
Acol_d_size=sizeof(x(1:n))
Annz_i_size=sizeof(jb(1:inz))
Annz_d_size=sizeof(b(1:inz))

! Define the CUDA stream and matrix parameters
ierr = cusolverSpCreate(handle)
ierr2 = ierr2 + ierr
ierr = cusparseCreate(cusparseHandle)
ierr2 = ierr2 + ierr
ierr = cudaStreamCreate(stream) 
ierr2 = ierr2 + ierr
ierr = cusolverSpSetStream(handle,stream) 
ierr2 = ierr2 + ierr
ierr = cusparseSetStream(cusparseHandle,stream) 
ierr2 = ierr2 + ierr
ierr = cusparseCreateMatDescr(descrA)
ierr2 = ierr2 + ierr
ierr = cusparseSetMatType2(descrA,CUBLAS_OP_N) 
ierr2 = ierr2 + ierr
ierr = cusparseSetMatIndexBase2(descrA,CUBLAS_OP_T) 
ierr2 = ierr2 + ierr
if (ierr2.ne.0) then
  write(*,'(A, I2)') 'Error during matrix setup ',ierr2
  stop
end if 

if (icpu_check.eq.1) go to 55
! Allocate memory on the GPU
write(*,*) 'Allocating GPU memory'
ierr = cudaMalloc(d_csrRowPtrA,Arow1_i_size)
ierr2 = ierr2 + ierr
ierr = cudaMalloc(d_csrColIndA,Annz_i_size) 
ierr2 = ierr2 + ierr
ierr = cudaMalloc(d_csrValA,Annz_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMalloc(d_x,Acol_d_size)
ierr2 = ierr2 + ierr
ierr = cudaMalloc(d_b,Arow_d_size)  
ierr2 = ierr2 + ierr
if (ierr2.ne.0) then
  write(*,'(A, I2)') 'Error during CUDA allocation: ',ierr2
  stop
end if 

55 if (ireorder.gt.0) then
  write(*,*) 'Beginning reordering'
  h_Q  = c_loc(permute_vec)  
  h_csrRowPtrB  = c_loc(reorder_ib) 
  h_csrColIndB  = c_loc(reorder_jb)
  h_csrValB  = c_loc(reorder_b)    
  h_mapBfromA  = c_loc(reorder_map)  
  if (ireorder.eq.1) then
    write(*,*) '---> Reordering with RCM'
    ierr = cusolverSpXcsrsymrcmHost(handle,n,inz, &
                     descrA,h_csrRowPtrA,h_csrColIndA,h_Q)
    if (ierr .ne. 0) then
      write (*, '(A, I2)') " cusolverSpXcsrsymrcmHost error: ", ierr
      stop
    end if

  else if (ireorder.eq.2) then
    write(*,*) '---> Reordering with AMD'
    ierr = cusolverSpXcsrsymamdHost(handle,n,inz, &
                     descrA,h_csrRowPtrA,h_csrColIndA,h_Q)
    if (ierr .ne. 0) then
      write (*, '(A, I2)') " cusolverSpXcsrsymamdHost error: ", ierr
      stop
    end if
  else
    write(*,*) 'Unknown reordering method'
  end if
  do ii = 1,n+1
    reorder_ib(ii)=ib(ii)
  end do
  do ii = 1,n
    reorder_rhs(ii)=rhs(ii)
  end do
  do ii = 1,inz
    reorder_jb(ii)=jb(ii)
  end do
  write(*,*) 'Finding permutation buffer'
  ierr = cusolverSpXcsrperm_bufferSizeHost(handle,n,n, &
                   inz,descrA,h_csrRowPtrB,h_csrColIndB,h_Q,h_Q,reorderWork)
  if (ierr .ne. 0) then
    write (*, '(A, I2)') " cusolverSpXcsrperm_bufferSizeHost error: ", ierr
    stop
  end if

  allocate(buffer_space(sizeof(dummy_value)*reorderWork))
  buffer_cpu = c_loc(buffer_space)

  do ii = 1,inz
    reorder_map(ii)=ii
  end do

  ierr = cusolverSpXcsrpermHost(handle,n,n, inz, &
                   descrA,h_csrRowPtrB,h_csrColIndB,h_Q,h_Q,&
                   h_mapBfromA, buffer_cpu)
  if (ierr .ne. 0) then
    write (*, '(A, I2)') " cusolverSpXcsrpermHost error: ", ierr
    stop
  end if

  do ii = 1,inz
    reorder_b(ii)=b(reorder_map(ii))
  end do
  do ii = 1,n+1
    ib(ii)=reorder_ib(ii)
  end do
  do ii = 1,n
    rhs(ii)=reorder_rhs(permute_vec(ii)+1)
  end do
  do ii = 1,inz
    jb(ii)=reorder_jb(ii)
    b(ii)=reorder_b(ii)
  end do
end if ! end of reordering loop\

if (icpu_check.eq.1) go to 65

! transfer memory over to GPU
write(*,*) 'Transferring memory to GPU'
ierr = cudaMemcpy(d_csrRowPtrA,h_csrRowPtrA,Arow1_i_size,cudaMemcpyHostToDevice)
ierr2 = ierr2 + ierr
ierr = cudaMemcpy(d_csrColIndA,h_csrColIndA,Annz_i_size,cudaMemcpyHostToDevice)
ierr2 = ierr2 + ierr
ierr = cudaMemcpy(d_csrValA,h_csrValA,Annz_d_size,cudaMemcpyHostToDevice)
ierr2 = ierr2 + ierr
ierr = cudaMemcpy(d_b,h_b,Arow_d_size,cudaMemcpyHostToDevice)
ierr2 = ierr2 + ierr
if (ierr2 .ne. 0 ) then
    write (*, '(A, I2)') " Error during cuda memcpy ", ierr2
    stop
end if

65 if (icpu_check.eq.1) then
!Linear solve on CPU
write(*,*) 'Beginning CPU solve'
write(*,*) 'Tolerance ',tol
ierr = cusolverSpDcsrlsvqrHost(handle,n,inz,descrA,h_csrValA,h_csrRowPtrA, &
                 h_csrColIndA,h_b,tol,ireorder,h_x,singularity )
   if (ierr .ne. 0 ) then
    write (*, '(A, I2)') " cusolverSpDcsrlsvqrHost: ", ierr
    stop
   end if
go to 75
else 
!Linear solve on GPU
write(*,*) 'Beginning GPU solve'
write(*,*) 'Tolerance ',tol
ierr = cusolverSpDcsrlsvqr(handle,n,inz,descrA,d_csrValA,d_csrRowPtrA, &
                 d_csrColIndA,d_b,tol,ireorder,d_x,singularity )
   if (ierr .ne. 0 ) then
    write (*, '(A, I2)') " cusolverSpDcsrlsvqr: ", ierr
    stop
   end if
end if

ierr = cudaDeviceSynchronize()
if (ierr .ne. 0 ) then
    write (*, '(A, I2)') " cudaDeviceSynchronize: ", ierr
    stop
   end if

if (singularity.gt.0) then
  write(*,*) 'WARNING: the matrix is singular at row ',singularity
  write(*,*) '         under tol ', tol
end if

if (iresid.eq.1) then
  !--------- Calculate the residual
  ierr = cudaMalloc(d_r,Arow_d_size)
  if (ierr .ne. 0 ) then
      write (*, '(A, I2)') " Could not allocate residual array ", ierr
      stop
  end if
  write(*,*) 'Computing residual'
  ierr = cudaMemcpy(d_r,d_b,Arow_d_size,cudaMemcpyDeviceToDevice)
  if (ierr .ne. 0 ) then
      write (*, '(A, I2)') " cudaMemcpy 1 error: ", ierr
      stop
  end if

! Previously used cusparseDcsrmv2
  ierr = cusparseDcsrmv(cusparseHandle,0, &
                   n,n,inz,minus_one,descrA,d_csrValA,d_csrRowPtrA,d_csrColIndA,&
                   d_x,one,d_r)
  if (ierr .ne. 0 ) then
      write (*, '(A, I2)') " cusparseDcsrmv: ", ierr
      stop
  end if

  ierr = cudaMemcpy(h_r,d_r,Arow_d_size,cudaMemcpyDeviceToHost)
  if (ierr .ne. 0 ) then
      write (*, '(A, I2)') " cudaMemcpy 5 error: ", ierr
      stop
  end if

!  write(*,'(A)') ' Residual'
!  write(*,*) max(abs(residual(1:n)))

end if ! end resid

write(*,*) 'Copy solution back to CPU'
ierr = cudaMemcpy(h_x,d_x,Acol_d_size,cudaMemcpyDeviceToHost)
if (ierr .ne. 0 ) then
    write (*, '(A, I2)') " cudaMemcpy 6 error: ", ierr
    stop
end if

75 if (ireorder.gt.0) then
  write(*,*) 'Reordering solution vector'
  do ii=1,n
    reorder_rhs(permute_vec(ii)+1) = x(ii)
  end do
  do ii=1,n
    x(ii) = reorder_rhs(ii)
  end do
end if 

if (icpu_check.eq.1) go to 85

ierr = cudafree(d_csrRowPtrA)
ierr2 = ierr2 + ierr
ierr = cudafree(d_csrColIndA)
ierr2 = ierr2 + ierr
ierr = cudafree(d_csrValA)
ierr2 = ierr2 + ierr
ierr = cudafree(d_x)
ierr2 = ierr2 + ierr
ierr = cudafree(d_b)
ierr2 = ierr2 + ierr
if (iresid.eq.1) then
   ierr = cudafree(d_r)
   ierr2 = ierr2 + ierr
end if
if (ierr2.ne.0) then
  write(*,'(A, I2)') 'Error during cudafree: ',ierr2
  stop
end if 

85 ierr = cusolverSpDestroy(handle)
ierr2 = ierr2 + ierr
ierr = cusparseDestroy(cusparseHandle)
ierr2 = ierr2 + ierr
ierr = cudaStreamDestroy(stream)
ierr2 = ierr2 + ierr
ierr = cusparseDestroyMatDescr(descrA)
ierr2 = ierr2 + ierr
if (ierr2.ne.0) then
  write(*,'(A, I2)') 'Error during cuda handle destruction: ',ierr2
  stop
end if 

if (ireorder.gt.0) then
deallocate(buffer_space)
end if

return
end subroutine cuda_sparse_solve_qr