# other imports
import matplotlib.pyplot as plt
import numpy as np

# import the package
import forcelayout as fl

# Need a dataset to visualise
dataset = np.array([[1, 1],
                    [1, 3],
                    [2, 2]])

# Need to use the brute force algorithm on a dataset this small
# (not recommended for larger datasets)
layout = fl.draw_spring_layout(dataset=dataset, algorithm=fl.SpringForce)

plt.show()