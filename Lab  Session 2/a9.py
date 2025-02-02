import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


file_path = r"/content/Lab Session Data.xlsx"
data = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")


numeric_columns = data.select_dtypes(include=["number"]).columns
numeric_vectors = data.loc[:1, numeric_columns]


vec1, vec2 = numeric_vectors.iloc[0].values.reshape(1, -1), numeric_vectors.iloc[1].values.reshape(1, -1)

cosine_sim = cosine_similarity(vec1, vec2)[0][0]

print("\nCosine Similarity between the first two observations:", round(cosine_sim, 4))


if cosine_sim > 0.8:
    print(" The vectors are highly similar.")
elif cosine_sim > 0.5:
    print(" The vectors have moderate similarity.")
else:
    print(" The vectors are not very similar.")
