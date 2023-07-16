import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from matplotlib import animation
from WaveAnimator import WaveAnimator
from wiggle_utilities import moving_average


###############################################################################
# WiggleArtist
###############################################################################
class WiggleArtist:
    def __init__(
            self,
            image_file,
            block_height = None,
            bg_color="b",
            line_color="w",
            fig = None,
            ax = None,
            n_block_rows = None
    ):
        self.im = imread(image_file, mode='F')
        self.nr, self.nc = self.im.shape

        if fig is None or ax is None:
            print("One of ax or fig is missing.\nCreating new fig & ax.\nPass BOTH to avoid")
            self.fig, self.ax = plt.subplots()
        else:
            self.fig = fig
            self.ax = ax

        if n_block_rows is None:
            self.block_height = block_height
            self.n_block_rows = int(np.ceil(self.nr / self.block_height))
            self.n_block_cols = self.nc - int(np.ceil(self.block_height / 2.0)) + 1

        else:
            self.n_block_rows = n_block_rows
            self.block_height = int(np.floor(self.nr / self.n_block_rows))
            self.n_block_cols = self.nc - int(np.ceil(self.block_height / 2.0)) + 1


        self.line_width = np.sqrt(self.block_height)
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

        self.wiggles_initial = self.wiggles.copy()
        self.init_ax()

    def init_ax(self):
        self.ax.axis([
            self.block_height,
            self.nc - self.block_height,
            self.block_height,
            self.nr - self.block_height,
        ])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.patch.set_facecolor(self.bg_color)

    def draw(self):
        for i in range(0, self.n_block_rows):
            self.ax.plot(self.wiggles[i, :], c=self.line_color,
                         lw=self.line_width)
        return self.fig, self.ax

    def init_wiggler(
            self,
            solver_class,
            steps_per_frame=1,
            bc_type="Dirichlet"
    ):
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

        self.waver = WaveAnimator(
            self.solver,
            self.fig,
            self.ax,
            self.line_width,
            self.line_color,
            self.steps_per_frame,
        )


if __name__ == "__main__":
    from WaveSolver import WaveSolver

    # Plot Cage
    image_file = "./imgs/NickCage.jpg"
    wa = WiggleArtist(
        image_file, block_height=3, line_color=(0, 0, 1, 0.5), bg_color="w",
        n_block_rows = 100
    )
    wa.init_wiggler(WaveSolver, bc_type="Periodic")
    anim = animation.FuncAnimation(
        wa.fig,
        wa.waver.animate,
        init_func=wa.waver.init,
        frames=25,
        interval=0.1,
        blit=True,
    )
    plt.show()
