import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('IPL_Ball_by_Ball_2008_2022.csv')

# Perform label encoding for categorical columns
cat_cols = ['batter', 'bowler', 'non-striker', 'extra_type', 'player_out', 'kind', 'fielders_involved', 'BattingTeam']
le = LabelEncoder()
for col in cat_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

# Extract the features from the dataset
X = df.iloc[:, 1:]  # Exclude the ID column or any other irrelevant columns

# Perform PCA with different values of n_components
explained_variances = []
n_components_range = np.arange(1, X.shape[1] + 1)  # Range of n_components to try

for n_components in n_components_range:
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    explained_variances.append(np.sum(pca.explained_variance_ratio_))

# Plot the explained variance ratio
plt.plot(n_components_range, explained_variances, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Explained Variance Ratio')
plt.grid(True)
plt.show()
