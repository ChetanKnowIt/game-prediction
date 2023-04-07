import pandas as pd


class mymodel:
    def __init__(self, model_name, preprocess, data):
        """
        This method to define your model and include any preprocessing steps
        """
        self.model_name = model_name
        self.preprocess = preprocess
        self.data = data
        pre1 = preprocess()

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
    
    def preprocess():
        """
        all preprocessing required for mymodel
        """
        return

def main():
    d1 = pd.read_csv('IPL_Ball_by_Ball_2008_2022.csv')
    m1 = mymodel('nn_test','yes', d1)
    m1.preprocess()
    m1.fit()
    m1.predict()


if __name__ == "__main__":
    main()
