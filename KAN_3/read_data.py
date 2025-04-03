import numpy as np

# Load the data from the .npy file
data = np.load("data.npy", allow_pickle=True)

# Print the shape of the array to know its dimensions (rows, columns)
print("Data shape:", data.shape)

# Print the first 5 rows of data to inspect the values
print("First 5 rows of data:")
print(data)
