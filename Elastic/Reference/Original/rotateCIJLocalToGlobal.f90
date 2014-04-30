subroutine rotateCIJLocalToGlobal(nx, ny, nz, ilo, jlo, klo, iup, jup, kup,&
          & azim, dip, rake, &
          & c11, c22, c33, c44, c55, c66, &
          & c12, c13, c14, c15, c16, &
          & c23, c24, c25, c26, &
          & c34, c35, c36, &
          & c45, c46, &
          & c56)

!  Subroutine transforms the CIJ matrix from its local frame (eTTI or eTOr)
!  to the global (FD) frame using the Bond 6x6 transformation matrix
!  and the Z(2)-Y(1)-Z Euler rotation matrix.
!
!
!  o Last Modified:  
!   04-MAR-14  Ver. 1.1: Corrected Euler rotation matrix B convention in
!              rotateCIJLocalToGlobal.f90 to be consistent with my local 
!              to global rotations.
!    03-03-13  Written.
!
!  o Written by Kurt T. Nihei


implicit none

integer :: nx, ny, nz
integer :: ilo, jlo, klo
integer :: iup, jup, kup
real, dimension(ilo:iup,jlo:jup,klo:kup) :: azim, dip, rake
integer*2, dimension(ilo:iup,jlo:jup,klo:kup) :: c11, c22, c33, &
&                                              & c44, c55, c66, &
                                          & c12, c13, c14, c15, c16, &
                                          & c23, c24, c25, c26, &
                                          & c34, c35, c36, &
                                          & c45, c46, &
                                          & c56

integer :: i, j, k
integer :: ii
real :: azim_, dip_, rake_
real :: cosa, cosd, cosr
real :: sina, sind, sinr
real :: b(3,3)  ! Euler Z(2)-Y(1)-Z rotation tensor
real :: m(6,6)  ! Bond 6x6 transformation matrix local: local to global (FD)
real :: mc(6,6) ! Bond matrix x cIJ matrix 


!$OMP PARALLEL DO PRIVATE(i, j, ii, azim_, dip_, rake_, &
!$OMP& cosa, cosd, cosr, sina, sind, sinr, b, m, mc)
do k = 1, nz
  do j = 1, ny
    do i = 1, nx

      ! Rotate cIj's from local frame to global (FD) frame:

      !..direction cosines:
      azim_ = azim(i,j,k)
      dip_  = dip(i,j,k)
      rake_ = rake(i,j,k)

      cosa = cos(azim_)
      cosd = cos(dip_)
      cosr = cos(rake_)
      sina = sin(azim_)
      sind = sin(dip_)
      sinr = sin(rake_)

      !..Euler (Z(2)-Y(1)-Z-order) rotation matrix b: local to global (FD)
      b(1,1) =  cosa*cosd*cosr - sina*sinr
      b(2,2) = -sina*cosd*sinr + cosa*cosr
      b(3,3) =  cosd

      !!!! ..this is correct for rot from global to local:
      !!!!b(1,2) = -cosa*cosd*sinr - sina*cosr
      !!!!b(1,3) =  cosa*sind
      !!!!b(2,1) =  sina*cosd*cosr + cosa*sinr
      !!!!b(2,3) =  sina*sind
      !!!!b(3,1) = -sind*cosr
      !!!!b(3,2) =  sind*sinr
      ! ..this is correct for rot from local to global:
      b(2,1) = -cosa*cosd*sinr - sina*cosr
      b(3,1) =  cosa*sind
      b(1,2) =  sina*cosd*cosr + cosa*sinr
      b(3,2) =  sina*sind
      b(1,3) = -sind*cosr
      b(2,3) =  sind*sinr

      !..Bond transformation matrix M:
      m(1,1) = b(1,1)*b(1,1)
      m(1,2) = b(1,2)*b(1,2)
      m(1,3) = b(1,3)*b(1,3)
      m(1,4) = 2*b(1,2)*b(1,3)
      m(1,5) = 2*b(1,1)*b(1,3)
      m(1,6) = 2*b(1,1)*b(1,2)
 
      m(2,1) = b(2,1)*b(2,1)
      m(2,2) = b(2,2)*b(2,2)
      m(2,3) = b(2,3)*b(2,3)
      m(2,4) = 2*b(2,2)*b(2,3)
      m(2,5) = 2*b(2,1)*b(2,3)
      m(2,6) = 2*b(2,1)*b(2,2)

      m(3,1) = b(3,1)*b(3,1)
      m(3,2) = b(3,2)*b(3,2)
      m(3,3) = b(3,3)*b(3,3)
      m(3,4) = 2*b(3,2)*b(3,3)
      m(3,5) = 2*b(3,1)*b(3,3)
      m(3,6) = 2*b(3,1)*b(3,2)

      m(4,1) = b(2,1)*b(3,1)
      m(4,2) = b(2,2)*b(3,2)
      m(4,3) = b(2,3)*b(3,3)
      m(4,4) = b(2,2)*b(3,3) + b(2,3)*b(3,2)
      m(4,5) = b(2,1)*b(3,3) + b(2,3)*b(3,1)
      m(4,6) = b(2,2)*b(3,1) + b(2,1)*b(3,2)
        
      m(5,1) = b(1,1)*b(3,1)
      m(5,2) = b(1,2)*b(3,2)
      m(5,3) = b(1,3)*b(3,3)
      m(5,4) = b(1,2)*b(3,3) + b(1,3)*b(3,2)
      m(5,5) = b(1,1)*b(3,3) + b(1,3)*b(3,1)
      m(5,6) = b(1,1)*b(3,2) + b(1,2)*b(3,1)
       
      m(6,1) = b(1,1)*b(2,1)
      m(6,2) = b(1,2)*b(2,2)
      m(6,3) = b(1,3)*b(2,3)
      m(6,4) = b(1,3)*b(2,2) + b(1,2)*b(2,3)
      m(6,5) = b(1,1)*b(2,3) + b(1,3)*b(2,1)
      m(6,6) = b(1,1)*b(2,2) + b(1,2)*b(2,1)

      !..[M] [CIJ_local] product:
      do ii = 1, 6
        mc(ii,1) = m(ii,1)*c11(i,j,k) + m(ii,2)*c12(i,j,k) + &
                 & m(ii,3)*c13(i,j,k)
        mc(ii,2) = m(ii,1)*c12(i,j,k) + m(ii,2)*c22(i,j,k) + &
                 & m(ii,3)*c23(i,j,k)
        mc(ii,3) = m(ii,1)*c13(i,j,k) + m(ii,2)*c23(i,j,k) + &
                 & m(ii,3)*c33(i,j,k)
        mc(ii,4) = m(ii,4)*c44(i,j,k)
        mc(ii,5) = m(ii,5)*c55(i,j,k)
        mc(ii,6) = m(ii,6)*c66(i,j,k)
      enddo

      !..[CIJ_global] = [M] [CIJ_local] [M]^T
      c11(i,j,k) = mc(1,1)*m(1,1) + mc(1,2)*m(1,2) + mc(1,3)*m(1,3) + &
                 & mc(1,4)*m(1,4) + mc(1,5)*m(1,5) + mc(1,6)*m(1,6)
      c12(i,j,k) = mc(1,1)*m(2,1) + mc(1,2)*m(2,2) + mc(1,3)*m(2,3) + &
                 & mc(1,4)*m(2,4) + mc(1,5)*m(2,5) + mc(1,6)*m(2,6)
      c13(i,j,k) = mc(1,1)*m(3,1) + mc(1,2)*m(3,2) + mc(1,3)*m(3,3) + &
                 & mc(1,4)*m(3,4) + mc(1,5)*m(3,5) + mc(1,6)*m(3,6)
      c14(i,j,k) = mc(1,1)*m(4,1) + mc(1,2)*m(4,2) + mc(1,3)*m(4,3) + &
                 & mc(1,4)*m(4,4) + mc(1,5)*m(4,5) + mc(1,6)*m(4,6)
      c15(i,j,k) = mc(1,1)*m(5,1) + mc(1,2)*m(5,2) + mc(1,3)*m(5,3) + &
                 & mc(1,4)*m(5,4) + mc(1,5)*m(5,5) + mc(1,6)*m(5,6)
      c16(i,j,k) = mc(1,1)*m(6,1) + mc(1,2)*m(6,2) + mc(1,3)*m(6,3) + &
                 & mc(1,4)*m(6,4) + mc(1,5)*m(6,5) + mc(1,6)*m(6,6)

      c22(i,j,k) = mc(2,1)*m(2,1) + mc(2,2)*m(2,2) + mc(2,3)*m(2,3) + &
                 & mc(2,4)*m(2,4) + mc(2,5)*m(2,5) + mc(2,6)*m(2,6)
      c23(i,j,k) = mc(2,1)*m(3,1) + mc(2,2)*m(3,2) + mc(2,3)*m(3,3) + &
                 & mc(2,4)*m(3,4) + mc(2,5)*m(3,5) + mc(2,6)*m(3,6)
      c24(i,j,k) = mc(2,1)*m(4,1) + mc(2,2)*m(4,2) + mc(2,3)*m(4,3) + &
                 & mc(2,4)*m(4,4) + mc(2,5)*m(4,5) + mc(2,6)*m(4,6)
      c25(i,j,k) = mc(2,1)*m(5,1) + mc(2,2)*m(5,2) + mc(2,3)*m(5,3) + &
                 & mc(2,4)*m(5,4) + mc(2,5)*m(5,5) + mc(2,6)*m(5,6)
      c26(i,j,k) = mc(2,1)*m(6,1) + mc(2,2)*m(6,2) + mc(2,3)*m(6,3) + &
                 & mc(2,4)*m(6,4) + mc(2,5)*m(6,5) + mc(2,6)*m(6,6)

      c33(i,j,k) = mc(3,1)*m(3,1) + mc(3,2)*m(3,2) + mc(3,3)*m(3,3) + &
                 & mc(3,4)*m(3,4) + mc(3,5)*m(3,5) + mc(3,6)*m(3,6)
      c34(i,j,k) = mc(3,1)*m(4,1) + mc(3,2)*m(4,2) + mc(3,3)*m(4,3) + &
                 & mc(3,4)*m(4,4) + mc(3,5)*m(4,5) + mc(3,6)*m(4,6)
      c35(i,j,k) = mc(3,1)*m(5,1) + mc(3,2)*m(5,2) + mc(3,3)*m(5,3) + &
                 & mc(3,4)*m(5,4) + mc(3,5)*m(5,5) + mc(3,6)*m(5,6)
      c36(i,j,k) = mc(3,1)*m(6,1) + mc(3,2)*m(6,2) + mc(3,3)*m(6,3) + &
                 & mc(3,4)*m(6,4) + mc(3,5)*m(6,5) + mc(3,6)*m(6,6)

      c44(i,j,k) = mc(4,1)*m(4,1) + mc(4,2)*m(4,2) + mc(4,3)*m(4,3) + &
                 & mc(4,4)*m(4,4) + mc(4,5)*m(4,5) + mc(4,6)*m(4,6)
      c45(i,j,k) = mc(4,1)*m(5,1) + mc(4,2)*m(5,2) + mc(4,3)*m(5,3) + &
                 & mc(4,4)*m(5,4) + mc(4,5)*m(5,5) + mc(4,6)*m(5,6)
      c46(i,j,k) = mc(4,1)*m(6,1) + mc(4,2)*m(6,2) + mc(4,3)*m(6,3) + &
                 & mc(4,4)*m(6,4) + mc(4,5)*m(6,5) + mc(4,6)*m(6,6)

      c55(i,j,k) = mc(5,1)*m(5,1) + mc(5,2)*m(5,2) + mc(5,3)*m(5,3) + &
                 & mc(5,4)*m(5,4) + mc(5,5)*m(5,5) + mc(5,6)*m(5,6)
      c56(i,j,k) = mc(5,1)*m(6,1) + mc(5,2)*m(6,2) + mc(5,3)*m(6,3) + &
                 & mc(5,4)*m(6,4) + mc(5,5)*m(6,5) + mc(5,6)*m(6,6)

      c66(i,j,k) = mc(6,1)*m(6,1) + mc(6,2)*m(6,2) + mc(6,3)*m(6,3) + &
                 & mc(6,4)*m(6,4) + mc(6,5)*m(6,5) + mc(6,6)*m(6,6)

    enddo
  enddo
enddo
!$OMP END PARALLEL DO


return
end subroutine