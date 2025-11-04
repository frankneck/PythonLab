from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from DataFrameHandler import DataFrameHadler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Program :
    
    path = r"Linear_regression\Inputs\kc_house_data.csv"

    def ReadInputs(path) :
        df = pd.read_csv(path)
        
        return df

    @staticmethod
    def UseRegression(df, model_type="Linear", alpha=1.0):
        # Target variable is Y and properties is X
        X = df.drop(columns=['price'])
        y = df['price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

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
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Metrixes
        coef_df = DataFrameHadler.Compute_coefficients(model, X_train.columns)
        r2 = DataFrameHadler.CalculateR2(y_test=y_test, y_pred=y_pred)
        rmse = DataFrameHadler.CalculateRMSE(y_test=y_test, y_pred=y_pred)
        
        # Output
        print(f"\n=== {model_type} Regression ===")
        print(f"Coefficents\n{coef_df}")
        print(f"alpha = {alpha}")
        print(f"R² = {r2:.4f}")
        print(f"RMSE = {rmse:.4f}")

        # Vizualization
        DataFrameHadler.Plot_coefficients(coef_df, y_test=y_test, y_pred=y_pred)

        return model, coef_df


    if __name__ == "__main__" :
        # Reading
        df = ReadInputs(path)
        
        # Filling the misssing values
        DataFrameHadler.FillMissingValues(df)
        
        # IQR
        skip_cols = [col for col in df.select_dtypes(include=['number']).columns if col != 'price']
        df = DataFrameHadler.DetectOutliersIQR(df, skip_cols)
        
        # Skip cols
        skip_cols = ["id", "date", "zipcode", "lat", "long", "sqft_basement"]
        df = DataFrameHadler.RemoveCols(df, skip_cols)
        df = DataFrameHadler.Standartize(df, skip_cols=['price'])

        # Adding new features
        new_features = ['sqft_per_room', 'age', 'yrs_since_renovated', 'total_rooms', 'density']
        df = DataFrameHadler.FeatureExtraction(df)
        # Standartization only needed extraction features
        df = DataFrameHadler.Standartize(df, skip_cols=['price'])

        # Use of linear regression
        UseRegression(df, model_type="Linear")
        UseRegression(df, model_type="Ridge", alpha=0.5)
        UseRegression(df, model_type="Lasso", alpha=0.01)




