import matplotlib.pyplot as plt
from wiggle_artist import wiggle_artist
from wave_solver import wave_solver
from matplotlib import animation

############################### Animate Lena ################################
image_file = './imgs/Lena.jpg'
wa = wiggle_artist(image_file = image_file, 
                   block_height = 8, 
                   line_color= (0,0,1,0.5), 
                   bg_color='w')
wa.init_wiggler(wave_solver, steps_per_frame = 20, bc_type='Periodic')
wa.fig.set_size_inches(8,8)
wa.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

n_frames = 2 * int(wa.n_block_cols / wa.waver.steps_per_frame) + 1

anim = animation.FuncAnimation(wa.fig, wa.waver.animate, init_func = wa.waver.init,
                               frames = n_frames, interval = .1, blit = True)

anim.save('./imgs/Lena_{0}_move.gif'.format(wa.block_height),
          dpi = 50,
          writer = 'imagemagick')

############################# Animate Nick Cage #############################
image_file = './imgs/NickCage.jpg'
wa = wiggle_artist(image_file = image_file, 
                   block_height = 4, 
                   line_color= (0,0,1,0.5), 
                   bg_color='w')
wa.init_wiggler(wave_solver, steps_per_frame = 20)
wa.fig.set_size_inches(8,8)
wa.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

n_frames = 2 * int(wa.n_block_cols / wa.waver.steps_per_frame) + 1

anim = animation.FuncAnimation(wa.fig, wa.waver.animate, init_func = wa.waver.init,
                               frames = n_frames, interval = .1, blit = True)

anim.save('./imgs/NickCage_{0}_move.gif'.format(wa.block_height),
          dpi = 50,
          writer = 'imagemagick')
