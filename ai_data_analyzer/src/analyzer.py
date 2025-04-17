import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import warnings
import io

class DataAnalyzer:
    def __init__(self, file):
        self.file = file
        self.data = None
        self.observations = {}
        self.summary = ""
        self.ml_readiness = {}
        warnings.filterwarnings('ignore')
        
    def analyze(self):
        self._load_data()
        self._basic_analysis()
        self._check_ml_readiness()
        self._create_summary()
        
    def _load_data(self):
        file_name = self.file.name.lower()
        
        if file_name.endswith('.csv'):
            self.data = pd.read_csv(self.file)
        elif file_name.endswith(('.xlsx', '.xls')):
            self.data = pd.read_excel(self.file)
        elif file_name.endswith('.json'):
            self.data = pd.read_json(self.file)
        elif file_name.endswith('.txt'):
            try:
                self.data = pd.read_csv(self.file)
            except:
                content = self.file.getvalue().decode('utf-8')
                self.data = pd.DataFrame({'text': content.split('\n')})
        
        self.data = self.data.dropna(how='all').reset_index(drop=True)
        self.data = self.data.loc[:, ~self.data.columns.str.contains('^Unnamed')]
        
    def _basic_analysis(self):
        self.observations = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.value_counts().to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'duplicates': self.data.duplicated().sum()
        }
    
    def _check_ml_readiness(self):
        numeric_cols = self.data.select_dtypes(include=np.number).columns
        self.ml_readiness = {
            'missing_values_analysis': (self.data.isnull().mean() * 100).to_dict(),
            'outlier_analysis': self._detect_outliers(),
            'feature_importance': self._estimate_feature_importance()
        }
    
    def _detect_outliers(self):
        outliers = {}
        numeric_cols = self.data.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            try:
                q1 = self.data[col].quantile(0.25)
                q3 = self.data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (1.5 * iqr)
                upper_bound = q3 + (1.5 * iqr)
                outliers[col] = {
                    'outlier_count': ((self.data[col] < lower_bound) | (self.data[col] > upper_bound)).sum(),
                    'outlier_pct': ((self.data[col] < lower_bound) | (self.data[col] > upper_bound)).mean() * 100
                }
            except:
                continue
        return outliers
    
    def _estimate_feature_importance(self):
        """Calculate feature importance using correlation with target"""
        numeric_cols = self.data.select_dtypes(include=np.number).columns
        
        # Skip if no numeric columns or only 1 column (target)
        if len(numeric_cols) < 2:
            return None
        
        try:
            # Ensure target is numeric
            if hasattr(self, 'target_col') and self.target_col in numeric_cols:
                corr_with_target = self.data[numeric_cols].corr()[self.target_col].abs()
                return corr_with_target.drop(self.target_col).to_dict()
            
            # If no target specified, use mean absolute correlation
            corr_matrix = self.data[numeric_cols].corr().abs()
            return corr_matrix.mean().sort_values(ascending=False).to_dict()
            
        except Exception as e:
            print(f"Error calculating feature importance: {e}")
            return None
        
    def _create_summary(self):
        self.summary = f"""
        Dataset Summary:
        - Rows: {self.data.shape[0]}
        - Columns: {self.data.shape[1]}
        - Missing Values: {self.data.isnull().sum().sum()}
        """