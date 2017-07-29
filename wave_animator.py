import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

class wave_animator:
	def __init__(self, solver, fig, ax, lw = 2, color='b', steps_per_frame = 1):
		self.solver = solver		
		self.fig = fig
		self.ax = ax
		self.lines = []
		self.lw = lw
		self.color = color
		self.steps_per_frame = steps_per_frame

		for i in range(solver.nr):
			lobj = self.ax.plot([],[],lw = self.lw,color = self.color)[0]
			self.lines.append(lobj)
		
	def init(self):
		for line in self.lines:
			line.set_data([],[])
		return self.lines

	def animate(self,i):
		for j in range(self.steps_per_frame):
			self.solver.step()
		for j, line in enumerate(self.lines):
			line.set_data(self.solver.x, self.solver.w_cur[j,:])
		return self.lines


if __name__ == '__main__':
	from dirichlet_wave_solver import dirichlet_wave_solver
	nx = 1000
	nt = 1000
	x = np.linspace(0,2,1000)
	w0 = []
	mm = 60

	fig = plt.figure()
	ax = plt.axes(xlim=(x[0], x[-1]), ylim=(-3, 3))

	for i in range(mm):
		w0.append(np.array([np.exp(-(x-((i+1)*(1.0/mm)))**2 / .1**2)+(-2+ (3.0*i)/mm)]).flatten())
	w0 = np.array(w0)

	solver = dirichlet_wave_solver(x,w0, np.zeros(w0.shape), 
								dt = x[1] - x[0], 
								dx = x[1] - x[0])

	wa = wave_animator(solver, fig, ax, steps_per_frame = 20)

	anim = animation.FuncAnimation(wa.fig, wa.animate, init_func=wa.init,
								   frames=2, interval=2, blit=True)


	plt.show()
