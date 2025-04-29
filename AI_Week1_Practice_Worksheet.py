
# ===============================
# AI Week 1 Practice Worksheet
# Topics: NumPy, Pandas, Matplotlib
# ===============================

# -------- NumPy Basics --------
import numpy as np

print("NumPy Practice")
a = np.array([[1, 2, 3], [4, 5, 6]])
print("Array:\n", a)
print("Shape:", a.shape)
print("Sum of all elements:", np.sum(a))
print("Mean of all elements:", np.mean(a))
print("Dot product of row 0 and row 1:", np.dot(a[0], a[1]))
print("Transpose:\n", a.T)

print("\n---------------------------\n")

# -------- Pandas Basics --------
import pandas as pd

print("Pandas Practice")
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)
print("First 5 rows of the dataset:")
print(df.head(10))

print("\nSpecies counts:")
print(df['species'].value_counts())

print("\nMean of each column by species:")
print(df.groupby('species').mean())

print("\n---------------------------\n")
