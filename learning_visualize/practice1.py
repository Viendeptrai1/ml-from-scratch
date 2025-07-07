import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 100)
y = x**2
z = x**3
plt.plot(x, y)
plt.title("Hàm số y = x^2")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend(["y = x^2", "y = x^3"])
plt.plot(x, z)
plt.show()