# ADAM optimizer

# objective function
def objective(x, y):
 return x**2.0 + y**2.0

# 3d plot of the test function
import numpy as np
import matplotlib.pyplot as plt

# objective function
def objective(x, y):
    return x**2.0 + y**2.0

# define range for input
r_min, r_max = -1.0, 1.0
# sample input range uniformly at 0.1 increments
xaxis = np.arange(r_min, r_max, 0.1)
yaxis = np.arange(r_min, r_max, 0.1)
# create a mesh from the axis
x, y = np.meshgrid(xaxis, yaxis)
# compute targets
results = objective(x, y)

# create a surface plot with the jet color scheme
figure = plt.figure(figsize=(18,12))
axis = figure.add_subplot(111, projection='3d')
axis.plot_surface(x, y, results, cmap='jet')

# show the plot
plt.show()
