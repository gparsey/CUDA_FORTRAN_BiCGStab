!
! CUDA 
!
module cuda_cusolve_map_reduced

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


 integer(c_int) function cudaDeviceSynchronize() bind(C,name="cudaDeviceSynchronize")

 use iso_c_binding
 implicit none
 end function cudaDeviceSynchronize


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
 type(c_ptr) :: csrValA
 type(c_ptr) :: csrRowPtrA
 type(c_ptr) :: csrColIndA
 type(c_ptr), value :: info
 end function cusparseDcsrsv_analysis

 integer(c_int) function cusparseDestroySolveAnalysisInfo(info) &
                 bind(C,name="cusparseDestroySolveAnalysisInfo")

 use iso_c_binding
 implicit none
 type(c_ptr),value::info
 end function cusparseDestroySolveAnalysisInfo

 end interface  

end module cuda_cusolve_map_reduced
!
!======================================================================
!======================================================================
  program main
   implicit none
   integer n,inz,i
   parameter (n=5)
   parameter (inz=13)
   double precision x(n),x_known(n),rhs(n),b(inz)
   integer ib(n+1),jb(inz)

   write(*,'(A)') 'Setting up test system'
   b(1) = 1.0d0;b(2) = 1.0d0;b(3) = 5.0d0;b(4) = 2.0d0
   b(5) = 1.0d0;b(6) = 3.0d0;b(7) = 2.0d0;b(8) = 1.0d0
   b(9) = 6.0d0;b(10) = 3.0d0;b(11) = 1.0d0;b(12) = 2.0d0
   b(13) = 1.0d0
   rhs(1) = 1.0d0;rhs(2) = 2.0d0;rhs(3) = 1.0d0
   rhs(4) = 3.0d0;rhs(5) = 0.0d0

   ib(1) = 1;ib(2) = 5;ib(3) = 7
   ib(4) = 9;ib(5) = 12;ib(6) = 14

   jb(1) = 1;jb(2) = 2;jb(3) = 4;jb(4) = 5
   jb(5) = 2;jb(6) = 3;jb(7) = 2;jb(8) = 3
   jb(9) = 1;jb(10) = 3;jb(11) = 4;jb(12) = 4
   jb(13) = 5

   x_known(1) = 0.08d0;x_known(2) = 0.2d0;x_known(3) = 0.6d0
   x_known(4) = 0.72d0;x_known(5) = -1.44d0
   x(1)=1.0d0;x(2)=1.0d0;x(3)=1.0d0
   x(4)=1.0d0;x(5)=1.0d0


  write(*,'(A)') 'Starting iterative solve'
  call cuda_BiCGStab_error(n,rhs,x,inz,ib,jb,b)
  write(*,'(A)') 'Found and Known solutions'
  do 23 i = 1,n
     write(*,*) x(i),x_known(i)
23  continue

  end program main
!
!=========================================================
subroutine cuda_BiCGStab_error(n,rhs,x,inz,ib,jb,b)
!=========================================================
use iso_c_binding
use cuda_cusolve_map_reduced
implicit none
integer n, inz
double precision x(n), rhs(n), b(inz)
target rhs,b,x
integer ib(n+1),jb(inz)
target ib,jb
integer ii,ierr,ierr2

integer, parameter :: dp = kind(1.d0)

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

! -------------------- pointers to device memory    
type(c_ptr) :: devPtrArowsIndex
type(c_ptr) :: devPtrAcolsIndex
type(c_ptr) :: devPtrAval
type(c_ptr) :: devPtrMrowsIndex
type(c_ptr) :: devPtrMcolsIndex
type(c_ptr) :: devPtrMval
type(c_ptr) :: devPtrX
type(c_ptr) :: devPtrF

integer*8 Arow1_i_size,Arow_d_size,Acol_d_size,Annz_i_size,Annz_d_size
 
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

! define pointers to host memory
ArowsIndex = c_loc(ib)
AcolsIndex = c_loc(jb)
Aval = c_loc(b)
h_x  = c_loc(x)  ! x = A \ b
h_rhs  = c_loc(rhs)  ! b = ones(m,1)

Arow1_i_size=sizeof(ib(1:n+1))
Arow_d_size=sizeof(rhs(1:n))
Acol_d_size=sizeof(x(1:n))
Annz_i_size=sizeof(jb(1:inz))
Annz_d_size=sizeof(b(1:inz))

! Define the CUDA stream and matrix parameters
ierr = cusparseCreate(cusparseHandle)
ierr2 = ierr2 + ierr
ierr = cusparseCreateMatDescr(descrA)
ierr2 = ierr2 + ierr
ierr = cusparseCreateMatDescr(descrM)
ierr2 = ierr2 + ierr
ierr = cudaStreamCreate(stream) 
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
ierr = cusparseSetMatFillMode(descrM,CUBLAS_OP_N)
ierr2 = ierr2 + ierr
ierr = cusparseSetMatDiagType(descrM,CUBLAS_OP_T)
ierr2 = ierr2 + ierr
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
if (ierr2.ne.0) then
  write(*,'(A, I2)') 'Error during cudafree: ',ierr2
  stop
end if 

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
end subroutine cuda_BiCGStab_error
