import matplotlib.pyplot as plt




if __name__ == "__main__":



plt.style.use('_mpl-gallery')
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

ax.scatter(points[0,:], points[1,:], points[2,:])

ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])
       


plt.show()


