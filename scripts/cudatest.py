#cuda test
from numba import cuda
import numpy as np

print(cuda.gpus)


@cuda.jit
def my_kernel():

    j = 0
    while j < 10000000:
        j+=1
        a = 3 + 2



# Create the data array - usually initialized some other way
data = np.ones(256)

# Set the number of threads in a block
threadsperblock = 32

# Calculate the number of thread blocks in the grid
blockspergrid = (data.size + (threadsperblock - 1)) // threadsperblock

# Now start the kernel
my_kernel[blockspergrid, threadsperblock]()

# Print the result
print(data)
