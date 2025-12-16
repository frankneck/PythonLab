from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from DataFrameHandler import DataFrameHandler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Program:
    
    path = r"Linear_regression\Inputs\kc_house_data.csv"

    def ReadInputs(path):
        df = pd.read_csv(path)
        return df

    @staticmethod
    def UseRegression(df, model_type="Linear", alpha=1.0):
        # Target variable is Y and properties is X
        X = df.drop(columns=['price'])
        y = df['price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Models choice
        if model_type == "Linear":
            model = LinearRegression()
        elif model_type == "Ridge":
            model = Ridge(alpha=alpha)
        elif model_type == "Lasso":
            model = Lasso(alpha=alpha, max_iter=10000)
        else:
            raise ValueError(f"Unknown regression type: {model_type}")

        # Learn and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrixes
        coef_df = DataFrameHandler.Compute_coefficients(model, X_train.columns)
        r2 = DataFrameHandler.CalculateR2(y_test=y_test, y_pred=y_pred)
        rmse = DataFrameHandler.CalculateRMSE(y_test=y_test, y_pred=y_pred)
        mean_y = y_test.mean()
        relative_error = (rmse / mean_y) * 100
        
        # Output
        print(f"\n=== {model_type} Regression ===")
        print(f"Coefficients\n{coef_df}")
        print(f"alpha = {alpha}")
        print(f"R² = {r2:.4f}")
        print(f"RMSE = {rmse:.4f}")
        print(f"Relative Error = {relative_error:.4f}")

        # Visualization
        DataFrameHandler.Plot_coefficients(coef_df, y_test=y_test, y_pred=y_pred, model_type=model_type)

        return model, coef_df
    

if __name__ == "__main__":
    # Reading
    df = Program.ReadInputs(Program.path)
    origin_count = df.shape[0]
    
    # Filling the missing values
    DataFrameHandler.FillMissingValues(df)
    
    # IQR
    skip_cols = [col for col in df.select_dtypes(include=['number']).columns if col != 'price']
    df = DataFrameHandler.DetectOutliersIQR(df, skip_cols)
    
    # Skip cols
    skip_cols = ["id", "date", "zipcode", "lat", "long", "sqft_basement"]
    df = DataFrameHandler.RemoveCols(df, skip_cols)
    df = DataFrameHandler.Standartize(df, skip_cols=['price'])

    current_count = df.shape[0]

    # Adding new features
    new_features = ['sqft_per_room', 'age', 'yrs_since_renovated', 'total_rooms', 'density']
    df = DataFrameHandler.FeatureExtraction(df)
    # Standardization only needed for extracted features
    df = DataFrameHandler.Standartize(df, skip_cols=['price'])
    print(df)

    print(f"Изначальное кол-во строк в DF : {origin_count}")
    print(f"Оставшееся кол-во строк в DF : {current_count}")
    
    # Use of linear regression
    Program.UseRegression(df, model_type="Linear")
    Program.UseRegression(df, model_type="Ridge", alpha=3)
    Program.UseRegression(df, model_type="Lasso", alpha=120)
