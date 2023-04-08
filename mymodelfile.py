import pandas as pd
import pprint
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.tsa.seasonal as sm
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

class mymodel:
    def __init__(self, model_name: str) -> None:
        """
        This method to define your model and include any preprocessing steps
        """
        self.model_name = model_name
        df1 = pd.read_csv('IPL_Ball_by_Ball_2008_2022.csv')
        self.preprocess(data=df1)

    def fit(self):
        """
        This method trains the parameters of the model. 
        It receives training data as a pair of pandas dataframes, 
        trains the model and returns reference to the MyModel object itself.
        """
        return 
    
    def predict(self):
        """
        This method receives test data as pandas dataframe and returns the predictions in specified format.
        """
        d1 = pd.read_csv('sample_prediction.csv')
        return d1
    
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


        display("=======================================================BASIC EDA=======================================================================")

        return 

        
def main():
    m1 = mymodel('LGBM_test')
    


if __name__ == "__main__":
    main()
