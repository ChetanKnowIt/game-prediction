from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('IPL_Ball_by_Ball_2008_2022.csv')
df['runs_per_inning'] = df.groupby(['ID', 'innings'])['total_run'].cumsum()

# Specify the columns to be included in each step
# Select columns with numerical data types
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

cat_cols = ['batter', 'bowler', 'non-striker', 'extra_type', 'player_out', 'kind', 'fielders_involved', 'BattingTeam']
le = LabelEncoder()
for col in cat_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

# Extract the features from the dataset
X = df.drop('runs_per_inning', axis=1)  # Update column names as needed
y = df['runs_per_inning']
# Create column transformer for preprocessing categorical and numerical features separately
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', 'passthrough', cat_cols)
    ])

# Create a pipeline with dimensionality reduction, scaling, and XGBoost regressor
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('reduce_dim', PCA(n_components=4)),
    ('regressor', GradientBoostingRegressor())
])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions on test data
y_pred = pipeline.predict(X_test)

# Evaluate the pipeline on test data
mse = mean_squared_error(y_test, y_pred)
