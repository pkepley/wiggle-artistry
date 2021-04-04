import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class WaveAnimator:
    def __init__(self, solver, fig, ax, lw=2, color="b", steps_per_frame=1):
        self.solver = solver
        self.fig = fig
        self.ax = ax
        self.lines = []
        self.lw = lw
        self.color = color
        self.steps_per_frame = steps_per_frame

        for i in range(solver.nr):
            lobj = self.ax.plot([], [], lw=self.lw, color=self.color)[0]
            self.lines.append(lobj)

    def init(self):
        for line in self.lines:
            line.set_data([], [])
        return self.lines

    def animate(self, i):
        for j in range(self.steps_per_frame):
            self.solver.step()
        for j, line in enumerate(self.lines):
            line.set_data(self.solver.x, self.solver.w_cur[j, :])
        return self.lines


if __name__ == "__main__":
    from WaveSolver import WaveSolver

    nx = 1000
    nt = 1000
    ny = 60

    x = np.linspace(0, 2, 1000)
    y = np.arange(0, ny, dtype="float")
    xx, yy = np.meshgrid(x, y)

    w0 = np.zeros((ny, nx))
    w0 = np.exp(-((xx - ((yy + 1) / ny)) ** 2) / 0.1 ** 2) + (-2 + (3 * yy) / ny)

    fig = plt.figure()
    ax = plt.axes(xlim=(x[0], x[-1]), ylim=(-3, 3))
    ax.set_xticks([])
    ax.set_yticks([])

    bc_type = "Dirichlet"
    solver = WaveSolver(
        x, w0, np.zeros(w0.shape), dt=x[1] - x[0], dx=x[1] - x[0],
        bc_type=bc_type
    )

    wa = WaveAnimator(solver, fig, ax, steps_per_frame=20)

    anim = animation.FuncAnimation(
        wa.fig, wa.animate, init_func=wa.init, frames=2, interval=2, blit=True
    )

    plt.show()
