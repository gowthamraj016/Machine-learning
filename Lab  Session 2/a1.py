import pandas as pd
import numpy as np


data = pd.read_excel(r"/content/Lab Session Data.xlsx", sheet_name="Purchase data")

A = data.loc[:, ['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
C = data[['Payment (Rs)']].values


print(f"A = {A}")
print(f"C = {C}")


dim = A.shape[1]
num_vectors = A.shape[0]
rank_A = np.linalg.matrix_rank(A)
A_pinv = np.linalg.pinv(A)
cost_vector = A_pinv @ C


print(f"The dimensionality of the vector space is = {dim}")
print(f"The number of vectors in the vector space is = {num_vectors}")
print(f"The rank of the matrix A is = {rank_A}")
print(f"The pseudo-inverse of matrix A is =\n{A_pinv}")
print(f"The cost of each product that is available for sale is = {cost_vector.flatten()}")
