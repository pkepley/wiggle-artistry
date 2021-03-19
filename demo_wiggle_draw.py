import matplotlib.pyplot as plt
from wiggle_artist import wiggle_artist

# Plot Cage
image_file = "./imgs/NickCage.jpg"
wa = wiggle_artist(
    image_file=image_file, block_height=4, line_color=(0, 0, 1, 0.5), bg_color="w"
)
wa.draw()
wa.fig.set_size_inches(12, 12)
plt.savefig(
    "./imgs/NickCage_{0}.png".format(wa.block_height), bbox_inches="tight", dpi=100
)

# Plot Lena
image_file = "./imgs/Lena.jpg"
wa = wiggle_artist(
    image_file=image_file, block_height=8, line_color=(0, 0, 1, 0.5), bg_color="w"
)
wa.draw()
wa.fig.set_size_inches(12, 12)
plt.savefig("./imgs/Lena_{0}.png".format(wa.block_height), bbox_inches="tight", dpi=100)

# Show results
plt.show()
