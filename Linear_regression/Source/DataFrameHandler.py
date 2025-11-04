import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


class DataFrameHadler :
    @staticmethod
    def FillMissingValues(df, zero_columns = None, strategy="mean", fill_text = "unknown") :
        """
        Parametres: 
            df (pd.DataFrame)
            zero_columns (list[str]) : columns where 0 is meaningful
            strategy is wheither Mean or Median
            fill_text is a placeholder
        """

        if (df.isnull().values.any()) :
            print("Value of missing valuse is " + df.isnull().sum())
            
            for col in df.select_dtypes(include="number") :
                if (zero_columns and col in zero_columns) :
                    df[col].fillna(0, inplace=True)
                else :
                    if (strategy == "mean") :
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif (strategy == "median") :
                        df[col].fillna(df[col].median(), inplace=True)

            for col in df.select_dtypes(include="object") :
                df[col].fillna("None", inplace=True)
        
        return df
    
    @staticmethod
    def DetectOutliersIQR(df, skip_cols = None, multiplier = 1.5):
        if (skip_cols is None) :
            skip_cols = []

        # Select numeric columns excluding skipped columns
        numeric_cols = df.select_dtypes(include=['number'])
        numeric_cols = numeric_cols.drop(columns = [col for col in skip_cols if col in numeric_cols.columns])


        # Find IQR bounds
        Q1 = numeric_cols.quantile(0.25)
        Q3 = numeric_cols.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        for col in lower_bound.index :
            print(f"Lower bound for {col} =", lower_bound[col])
        
        print()
        
        for col in upper_bound.index :
            print(f"Upper bound for {col} =", upper_bound[col])


        # Create a mask (~ is NOT)
        mask = ~((numeric_cols < lower_bound) | (numeric_cols > upper_bound)).any(axis=1)
        outliers = df[~mask]

        # Without outliers
        return df[mask]
    
    def RemoveCols(df, skipped_cols = None) :
        if skipped_cols != None :
            return df.drop(columns = skipped_cols)
        else :
            return df
        
    @staticmethod
    def Standartize(df, skip_cols = None):
        if skip_cols is None:
            skip_cols = []
        
        numeric_cols = [col for col in df.select_dtypes(include=['number']).columns if col not in skip_cols]
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        return df
    
    @staticmethod
    def FeatureExtraction(df):
        """
        Creating of new features
        """

        current_year = datetime.datetime.now().year

        # Sqare per one room
        df['sqft_per_room'] = df['sqft_living'] / (df['bedrooms'] + df['bathrooms'])

        # House's year
        df['age'] = current_year - df['yr_built']

        # The time when renovation has been done 
        df['yrs_since_renovated'] = current_year - df['yr_renovated']
        df['yrs_since_renovated'] = df['yrs_since_renovated'].where(df['yr_renovated'] > 0, df['age'])

        # Sum of rooms
        df['total_rooms'] = df['bedrooms'] + df['bathrooms']

        # Density near houses
        df['density'] = df['sqft_living15'] / df['sqft_lot15']

        return df
    
    @staticmethod
    def Compute_coefficients(model, feature_names):
        """
        Вычисление коэффициентов регрессии и организация в DataFrame.
        """
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': model.coef_
        })
        coef_df = coef_df.sort_values(by='Coefficient', ascending=True)
        
        return coef_df
    
    @staticmethod
    def Plot_coefficients(coef_df, y_test=None, y_pred=None, new_features=None, figsize=(10, 8), base_color='skyblue', new_feature_color='orange', title='The influence of attributes on price'):
        # histogramm
        colors = []
        for feature in coef_df['Feature']:
            if new_features and feature in new_features:
                colors.append(new_feature_color)
            else:
                colors.append(base_color)

        plt.figure(figsize=figsize)
        plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
        plt.xlabel('Коэффициент регрессии')
        plt.title(title)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.show()

        # predict VS real
        if y_test is not None and y_pred is not None:
            plt.figure(figsize=(8, 8))
            plt.scatter(y_pred, y_test, alpha=0.5, color='blue')
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
            plt.xlabel('Предсказанное значение')
            plt.ylabel('Реальное значение')
            plt.title('Предсказанное vs Реальное')
            plt.grid(True)
            plt.show()
    
    @staticmethod
    def CalculateR2(y_test, y_pred):
        r2 = r2_score(y_test, y_pred)
        return r2
        
    @staticmethod
    def CalculateRMSE(y_test, y_pred):
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return rmse
        