import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib import animation

class dirichlet_wave_solver:
    def __init__(self, x, w0, v0, dt, dx):
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

        # Allocate space for waves, and use ic's
        self.w_new = np.zeros(self.w0.shape)
        self.w_cur = self.w0 + self.v0 * dt
        self.w_old = self.w0
        
        # Spatial 2nd derivative stencil
        self.K = np.zeros((self.nc,self.nc))

        # Fill the diagonal and off-diags
        self.K[0,0] = 1.0
        self.K[self.nc-1,self.nc-1] = 1.0
        for i in range(1,self.nc-1):
            self.K[i,i-1] = -1.0
            self.K[i,i] = 2.0
            self.K[i,i+1] = -1.0

        # sparsify the matrix
        self.K = sparse.csr_matrix(self.K)

    def step(self):
        for i in range(self.nr):
             self.w_new[i,:] = ((2 * self.w_cur[i,:] - self.w_old[i,:]) -
                               self.dtsq_over_dxsq * 
                               self.K.dot(self.w_cur[i,:]).flatten())
        self.w_new[:,0] = self.w_cur[:,0]
        self.w_new[:,self.nc-1] = self.w_cur[:,self.nc-1]

        self.w_old = self.w_cur.copy()
        self.w_cur = self.w_new.copy()
        self.t_cur = self.t_cur + self.dt
