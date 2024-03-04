import numpy as np
import matplotlib.pyplot as plt

# Original function
x = np.linspace(-5, 9, 1000)
y = -2 * np.abs(x - 2)

# Transformed function
y2 = 4 * np.abs(x - 2)

# Plot the functions
plt.plot(x, y, label='Original function')
plt.plot(x, y2, label='Transformed function')
plt.legend()
plt.grid()
plt.show()
