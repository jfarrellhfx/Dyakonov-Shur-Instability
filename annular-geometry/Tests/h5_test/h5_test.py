import h5py
import numpy as np

a = np.linspace(0, 1, 100)
b = np.linspace(100, 200, 100)

print(a)
print(b)
f = h5py.File("h5_test2.h5", "a")
f.create_dataset("test1", data = a)
f.close()

f = h5py.File("h5_test2.h5", "a")
f.create_dataset("test2", data = b)
f.close()
