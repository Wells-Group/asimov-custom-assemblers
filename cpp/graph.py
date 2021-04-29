import seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy


dt = pd.read_csv("results.txt")
matrix_name = "Stiffness"

seaborn.set(style="ticks")
seaborn.set_style("darkgrid")

dt1 = dt[dt["method"] == "main"].copy()
dt2 = dt[dt["method"] == "custom"].copy()

dt1["speedup"] = dt1.time/dt2.time.array


g = seaborn.catplot(x="degree", y="speedup", hue="compiler", col="compiler", kind="bar", data=dt1)


# plt.yscale("log")
# plt.title(f"{matrix_name} Matrix")
plt.show()
