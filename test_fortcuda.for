c======================================================================
c======================================================================
       program main

       implicit none

       integer n,inz,i
       parameter (n=5)
       parameter (inz=13)
       double precision tol
       double precision x(n),x_known(n),rhs(n),b(inz)
       integer ib(n+1),jb(inz)
       logical method1,method2,method3

       method1=.true.
       method2=.true.
       method3=.false.

       tol = 1.0d-8

       write(*,'(A)') 'Setting up test system'
       b(1) = 1.0d0
       b(2) = 1.0d0
       b(3) = 5.0d0
       b(4) = 2.0d0
       b(5) = 1.0d0
       b(6) = 3.0d0
       b(7) = 2.0d0
       b(8) = 1.0d0
       b(9) = 6.0d0
       b(10) = 3.0d0
       b(11) = 1.0d0
       b(12) = 2.0d0
       b(13) = 1.0d0

       rhs(1) = 1.0d0
       rhs(2) = 2.0d0
       rhs(3) = 1.0d0
       rhs(4) = 3.0d0
       rhs(5) = 0.0d0

       ib(1) = 1
       ib(2) = 5
       ib(3) = 7
       ib(4) = 9
       ib(5) = 12
       ib(6) = 14

       jb(1) = 1
       jb(2) = 2
       jb(3) = 4
       jb(4) = 5
       jb(5) = 2
       jb(6) = 3
       jb(7) = 2
       jb(8) = 3
       jb(9) = 1
       jb(10) = 3
       jb(11) = 4
       jb(12) = 4
       jb(13) = 5

       x_known(1) = 0.08d0
       x_known(2) = 0.2d0
       x_known(3) = 0.6d0
       x_known(4) = 0.72d0
       x_known(5) = -1.44d0

       if (method1) then
       write(*,'(A)') 'Resetting solution vector'
       call reset_x(n,1.0d0,x)
       write(*,'(A)') 'Starting direct solve'
       call cuda_sparse_solve_qr(n,rhs,x,inz,ib,jb,b,tol)
       write(*,'(A)') 'Found and Known solutions'
       do 22 i = 1,n
         write(*,*) x(i),x_known(i)
   22  continue
       end if

       if (method2) then
       write(*,'(A)') 'Resetting solution vector'
       call reset_x(n,1.0d0,x)
       write(*,'(A)') 'Starting iterative solve 1'
       call cuda_BiCGStab(n,rhs,x,inz,ib,jb,b,tol)
       write(*,'(A)') 'Found and Known solutions'
       do 23 i = 1,n
         write(*,*) x(i),x_known(i)
   23  continue
       end if

       if (method3) then
       write(*,'(A)') 'Resetting solution vector'
       call reset_x(n,1.0d0,x)
       write(*,'(A)') 'Starting iterative solve 2'
       call cuda_BiCGStab2(n,rhs,x,inz,ib,jb,b,tol)
       write(*,'(A)') 'Found and Known solutions'
       do 24 i = 1,n
         write(*,*) x(i),x_known(i)
   24  continue
       end if

       end program main

c-----------------------------------------------------------------------
       subroutine reset_x(n,rval,x)
       implicit none
       integer n,i
       double precision rval
       double precision x(n)

       do 19 i = 1,n
         x(i) = rval
  19   continue

       end subroutine reset_x
