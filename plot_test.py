import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 200)
y = np.sin(x)

plt.figure()
plt.plot(x, y)
plt.title("Sine wave")
plt.show()

rng = np.random.default_rng(0) #random number generator 
samples = rng.normal(1, 1, size=1000) ## mea μ, stdσ, size= 

plt.figure()
plt.hist(samples, bins=30)
plt.title("Histogram")
plt.show()