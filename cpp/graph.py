import seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy


dt = pd.read_csv("out.txt")
matrix_name = Mass
# seaborn.set_theme(style="darkgrid")
seaborn.set_theme(style="ticks")

g = seaborn.catplot(x="degree", y="time", hue="method", col="compiler", kind="bar", data=dt)

max_degree = max(dt["degree"])
P = numpy.arange(1, max_degree + 1)
Asize = ((P + 1) * (P + 2) * (P + 3) / 6)**2

plt.plot(P - 1, Asize / Asize[0] * min(dt["time"]), "r-o")
plt.legend([r"size($A_e$)"])

plt.yscale("log")
plt.title(f"{matrix_name} Matrix")
plt.show()
