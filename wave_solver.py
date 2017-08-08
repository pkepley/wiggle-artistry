import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib import animation

def taper(x):
    # Very crude taper for the periodic bc case
    nx = len(x.flatten())
    x0 = np.linspace(0,1,nx)
    y = np.cumsum(np.exp(-(x0-.05)**2/.0100**2) - np.exp(-(x0-.95)**2/.0100**2), 
                  axis=0)
    y = y / y[nx/2]
    
    return y

class wave_solver:
    def __init__(self, x, w0, v0, dt, dx, bc_type = 'Dirichlet'):
        if len(w0.shape) == 1:
            w0 = w0.reshape((1, w0.shape[0]))
            v0 = v0.reshape((1, v0.shape[0]))
            nr, nc = w0.shape
        else:
            nr, nc = w0.shape
        
        self.x = x
        self.w0 = w0
        self.v0 = v0
        self.dx = dx
        self.dt = dt
        self.dtsq_over_dxsq = dt**2 / dx**2
        self.t_cur = 0
        self.nr = nr
        self.nc = nc
        self.bc_type = bc_type

        # Taper bc in periodic BC case to avoid discontinuities
        if self.bc_type == 'Periodic':
            self.taper_for_periodic()
        
        # Allocate space for waves, and use ic's
        self.w_new = np.zeros(self.w0.shape)
        self.w_cur = self.w0 + self.v0 * dt
        self.w_old = self.w0
        
        # Spatial 2nd derivative stencil
        self.K = np.zeros((self.nc,self.nc))

        # Handle periodic 2nd derivative
        if self.bc_type == 'Periodic':
            # wrapped 2nd derivative at left edge
            self.K[0,self.nc-1] = -1
            self.K[0,0] = 2
            self.K[0,1] = -1
            # wrapped 2nd derivative at right edge
            self.K[self.nc-1,self.nc-2] = -1
            self.K[self.nc-1,self.nc-1] = 2
            self.K[self.nc-1,0] = -1
        # Don't bother with the boundary values in the non-periodic case 
        else:
            self.K[0,0] = 1.0
            self.K[self.nc-1,self.nc-1] = 1.0

        # Fill the 2nd derivative stencil in the interior
        for i in range(1,self.nc-1):
            self.K[i,i-1] = -1.0
            self.K[i,i] = 2.0
            self.K[i,i+1] = -1.0

        # sparsify the matrix
        self.K = sparse.csr_matrix(self.K)

    def taper_for_periodic(self):
        # In the periodic case, taper the starting wave at 
        # the end-points to zero to avoid discontinuities
        tp = taper(self.x)
        for i in range(self.nr):
            m = np.mean(self.w0[i,:])
            self.w0[i,:] = (self.w0[i,:] - m) * tp + m

    def step(self):
        for i in range(self.nr):
             self.w_new[i,:] = ((2 * self.w_cur[i,:] - self.w_old[i,:]) -
                               self.dtsq_over_dxsq * self.K.dot(self.w_cur[i,:]).flatten())
        self.update_bcs()

        self.w_old = self.w_cur.copy()
        self.w_cur = self.w_new.copy()
        self.t_cur = self.t_cur + self.dt

    def update_bcs(self):
        if self.bc_type == 'Dirichlet':
            # Hold the boundary conditions fixed to the same value for all time
            self.w_new[:,0] = self.w_cur[:,0]
            self.w_new[:,self.nc-1] = self.w_cur[:,self.nc-1]

        elif self.bc_type == 'Neumann':
            # Set the derivative to 0 at the boundary using 
            #   f'(x) ~ (f(x+dx) - f(x))/dx 
            self.w_new[:,0] = self.w_new[:,1]
            self.w_new[:,self.nc-1] = self.w_new[:,self.nc-2]

        elif self.bc_type == 'Periodic':
            # Periodic bc's are handled by the 2nd derivative operator
            pass
