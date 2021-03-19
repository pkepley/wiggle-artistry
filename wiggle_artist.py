import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from wave_animator import wave_animator
from matplotlib import animation

#####################################################################################
## Compute moving_average. credit to:
## https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
#####################################################################################
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


#####################################################################################
## wiggle_artist
#####################################################################################
class wiggle_artist:
    def __init__(self, image_file, block_height, bg_color="b", line_color="w"):
        self.im = imread(image_file, as_gray=True)
        self.nr, self.nc = self.im.shape

        self.block_height = block_height
        self.n_block_rows = int(np.ceil(self.nr / self.block_height))
        self.n_block_cols = self.nc - int(np.ceil(self.block_height / 2.0)) + 1

        self.line_width = np.sqrt(block_height)
        self.line_color = line_color
        self.bg_color = bg_color

        self.sigma = np.sqrt(np.var(self.im))

        self.wiggles = np.zeros((self.n_block_rows, self.n_block_cols))

        for i in range(0, self.n_block_rows):
            row = i * self.block_height
            row_height = self.nr - row
            mu = np.mean(self.im[row, :])
            self.wiggles[i, :] = moving_average(
                self.im[row, :], int(np.ceil(self.block_height / 2.0))
            )
            self.wiggles[i, :] = row_height + np.sqrt(self.block_height) * (
                (self.wiggles[i, :] - mu) / self.sigma
            )
        self.fig, self.ax = plt.subplots()
        self.ax.axis(
            [
                self.block_height,
                self.nc - self.block_height,
                self.block_height,
                self.nr - self.block_height,
            ]
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.patch.set_facecolor(self.bg_color)

    def draw(self):
        for i in range(0, self.n_block_rows):
            self.ax.plot(self.wiggles[i, :], c=self.line_color, lw=self.line_width)
        return self.fig, self.ax

    def init_wiggler(self, solver_class, steps_per_frame=1, bc_type="Dirichlet"):
        self.x = np.linspace(0, self.n_block_cols, self.n_block_cols)
        self.w0 = self.wiggles
        self.v0 = np.zeros(self.w0.shape)
        self.dt = self.x[1] - self.x[0]
        self.dx = self.x[1] - self.x[0]
        self.steps_per_frame = steps_per_frame
        self.bc_type = bc_type

        self.solver = solver_class(
            x=self.x,
            w0=self.w0,
            v0=self.v0,
            dt=self.dt,
            dx=self.dx,
            bc_type=self.bc_type,
        )

        self.waver = wave_animator(
            self.solver,
            self.fig,
            self.ax,
            self.line_width,
            self.line_color,
            self.steps_per_frame,
        )


if __name__ == "__main__":
    from wave_solver import wave_solver

    # # Plot Cage
    image_file = "./imgs/NickCage.jpg"
    # wa = wiggle_artist(image_file, block_height = 4, line_color= (0,0,1,0.5), bg_color='w')
    # wa.draw()

    # Plot Cage
    # image_file = './imgs/NickCage.jpg'
    # wa = wiggle_artist(image_file, block_height = 4, line_color= (0,0,1,0.5), bg_color='w')
    # wa.draw()

    # Plot Lena
    # image_file = './imgs/Lena.jpg'
    wa = wiggle_artist(
        image_file, block_height=3, line_color=(0, 0, 1, 0.5), bg_color="w"
    )
    wa.init_wiggler(wave_solver, bc_type="Periodic")
    anim = animation.FuncAnimation(
        wa.fig,
        wa.waver.animate,
        init_func=wa.waver.init,
        frames=25,
        interval=0.1,
        blit=True,
    )
    plt.show()
