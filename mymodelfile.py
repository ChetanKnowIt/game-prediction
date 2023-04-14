import pandas as pd
import pprint
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.tsa.seasonal as sm
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

def display(text: str) -> None:
    # ANSI escape code for bold text
    BOLD = '\033[1m'
    # ANSI escape code for resetting text formatting
    RESET = '\033[0m'
    
    # Print the text with bold formatting and reset the formatting after
    print(BOLD + text + RESET)

def reduce_dtype(df):
    """
    Reduce data types of DataFrame columns to more memory-efficient types.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame with reduced data types.
    """
    # Copy the input DataFrame to avoid modifying the original DataFrame
    df = df.copy()

    # Loop through each column in the DataFrame
    for col in df.columns:
        col_dtype = df[col].dtype  # Get the current data type of the column

        # Check if the column contains numerical data
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if the column can be downcasted to int8
            if df[col].min() >= -128 and df[col].max() <= 127:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            # Check if the column can be converted to categorical
            elif len(df[col].unique()) / len(df[col]) < 0.5:
                df[col] = df[col].astype('category')
            # Otherwise, try to downcast to float16 or int16
            else:
                df[col] = pd.to_numeric(df[col], downcast='float')

    return df

class MyModel:
    def __init__(self, model_name: str) -> None:
        """
        This method to define your model and include any preprocessing steps
        """
        self.model = DecisionTreeRegressor(random_state=42) 
        self.label_encoders = {}  # Store label encoders for categorical columns


    def fit(self, X_train, y_train):
        """
        This method trains the parameters of the model. 
        It receives training data as a pair of pandas dataframes, 
        trains the model and returns reference to the MyModel object itself.
        """
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X_test):
        """
        This method receives test data as pandas dataframe and returns the predictions in specified format.
        """
        y_pred = self.model.predict(X_test)
        # Perform any necessary formatting of the predictions
        # and return the results
        return y_pred
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """All preprocessing required for mymodel.

        Args:
            data (pd.DataFrame): A DataFrame containing the data to be processed.

        Returns:
            pd.DataFrame: A new DataFrame containing the processed data.
        """
        display("=======================================================BASIC EDA=======================================================================")
        display("Contents from IPL_Ball_by_Ball_2008_2022.csv: ")
        print(data.head(20))
        display("Columns: ")
        pprint.pprint(data.columns, indent=4)
        display("Shape: ")
        print(data.shape)
        display("Info: ")
        print(data.info())
        display("Statistical Description: ")
        print(data.describe(include=["int64"]))
        reduce_data = reduce_dtype(data)
        display("Reduced DataFrame Info: ")
        print(reduce_data.info())
        print(reduce_data.describe(include=["int8"]))
        print(data.describe(include=["int64"]))
        #print(data['ID'].value_counts())
        #print(data.describe(include=["int64"])) statistically data and reduce_data will never be same but we can use it to improve development speed
        
        # countplot for testing
        #sns.countplot(x="batsman_run", data = reduce_data)
        #sns.set(style = "whitegrid")
        #plt.show()
        
        # seasonal decompose for testing time series 
        # result = sm.seasonal_decompose(reduce_data['batsman_run'], model = 'additive', period = 6)
        #with plt.style.context("seaborn-whitegrid"):
            #fig_size = plt.rcParams["figure.figsize"]
            #fig_size[0] = 12 # X scaling of fig 
            #fig_size[1] = 8 # Y scaling of fig
            #plt.show()

        columns_to_keep = ['innings', 'overs', 'ballnumber', 'batter', 'bowler', 'non-striker',
                   'batsman_run', 'extras_run', 'total_run', 'BattingTeam']
        print(columns_to_keep)
        train_full = reduce_data[columns_to_keep]
        print(train_full)
        data_preprocessed = train_full.copy()
        # Perform label encoding for categorical columns
        categorical_cols = ['ID', 'batter', 'bowler', 'non-striker', 'extra_type', 'player_out', 'kind', 'fielders_involved', 'BattingTeam']
        for col in categorical_cols:
            if col in data_preprocessed.columns:
                if col not in self.label_encoders:
                    # Initialize label encoder for the column if not already done
                    self.label_encoders[col] = LabelEncoder()
                # Apply label encoder to the column
                data_preprocessed[col] = self.label_encoders[col].fit_transform(data_preprocessed[col].astype(str))
        return data_preprocessed

    def evaluate(self, X_test, y_test):
        """
        This method receives test data and ground truth labels,
        and returns the Mean Squared Error (MSE) for the predictions.
        """
        y_pred = self.predict(X_test)  # Make predictions using the trained model
        mse = mean_squared_error(y_test, y_pred)  # Calculate MSE
        return mse
    

def main():
    # Load your data into a pandas DataFrame
    # Assume 'df' is the DataFrame containing the selected columns

    # Initialize your model
    model = MyModel(model_name='DecitionTreeRegression')

    # Preprocess the data
    df = pd.read_csv('IPL_Ball_by_Ball_2008_2022.csv')
    df_preprocessed = model.preprocess(df)

    # Split the data into training and testing sets
    X = df_preprocessed.drop('total_run', axis=1)  # Features
    y = df_preprocessed['total_run']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the model
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Evaluate the model
    display("=========================================Decision Tree Regressor Prediction===================================================================")
    # Evaluate the model
    mse = model.evaluate(X_test, y_test)
    print('Mean Squared Error (MSE):', mse)
if __name__ == "__main__":
    main()
