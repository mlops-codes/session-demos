"""
Data preprocessing module for McDonald's financial data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
from typing import Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class McDonaldsDataProcessor:
    """Data processor for McDonald's financial dataset"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.target_name = None
        
    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        """Load raw McDonald's financial data"""
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data shape: {df.shape}")
        return df
    
    def clean_and_restructure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and restructure the raw data into a usable format"""
        logger.info("Cleaning and restructuring data...")
        
        cleaned_data = []
        
        for _, row in df.iterrows():
            table = row['table']
            heading = row['heading']
            subheading = row['subheading']
            
            # Extract yearly data
            for year in ['2021', '2022', '2023', '2024']:
                if year in df.columns:
                    value = row[year]
                    # Clean numeric values (remove commas, quotes)
                    if pd.notna(value) and str(value).strip() != '':
                        try:
                            # Remove quotes and commas
                            clean_value = str(value).replace('"', '').replace(',', '')
                            numeric_value = float(clean_value)
                            
                            cleaned_data.append({
                                'year': int(year),
                                'table': table,
                                'heading': heading,
                                'subheading': subheading,
                                'metric': f"{table}_{heading}_{subheading}",
                                'value': numeric_value
                            })
                        except:
                            pass
        
        clean_df = pd.DataFrame(cleaned_data)
        logger.info(f"Cleaned data shape: {clean_df.shape}")
        return clean_df
    
    def create_pivot_features(self, clean_df: pd.DataFrame) -> pd.DataFrame:
        """Create pivot table with metrics as features"""
        logger.info("Creating pivot features...")
        
        pivot_df = clean_df.pivot_table(
            index='year', 
            columns='metric', 
            values='value', 
            aggfunc='first'
        )
        
        # Fill missing values
        pivot_df = pivot_df.fillna(method='ffill').fillna(method='bfill')
        
        # Select features with good completeness
        feature_completeness = pivot_df.count() / len(pivot_df)
        good_features = feature_completeness[feature_completeness > 0.5].index.tolist()
        
        ml_df = pivot_df[good_features]
        logger.info(f"Selected {len(good_features)} features")
        
        return ml_df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features for better predictions"""
        logger.info("Engineering additional features...")
        
        engineered_df = df.copy()
        
        # Add year as a feature
        engineered_df['year_numeric'] = engineered_df.index
        
        # Calculate year-over-year growth rates
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                growth_col = f"{col}_growth"
                engineered_df[growth_col] = df[col].pct_change()
        
        # Calculate moving averages (2-year)
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                ma_col = f"{col}_ma2"
                engineered_df[ma_col] = df[col].rolling(window=2, min_periods=1).mean()
        
        # Remove infinite and NaN values
        engineered_df = engineered_df.replace([np.inf, -np.inf], np.nan)
        engineered_df = engineered_df.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Engineered features shape: {engineered_df.shape}")
        return engineered_df
    
    def prepare_ml_data(self, df: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for machine learning"""
        logger.info("Preparing ML data...")
        
        # If no target specified, create one by predicting next year's total revenue
        if target_column is None:
            # Find a revenue column to predict
            revenue_cols = [col for col in df.columns if 'revenue' in col.lower()]
            if revenue_cols:
                target_column = revenue_cols[0]
                logger.info(f"Using {target_column} as target")
                
                # Shift target to predict next year
                target = df[target_column].shift(-1)
                features = df.drop(columns=[target_column])
                
                # Remove last row (no target available)
                features = features[:-1]
                target = target[:-1]
            else:
                raise ValueError("No suitable target column found")
        else:
            if target_column not in df.columns:
                raise ValueError(f"Target column {target_column} not found")
            
            target = df[target_column]
            features = df.drop(columns=[target_column])
        
        self.feature_names = features.columns.tolist()
        self.target_name = target_column
        
        logger.info(f"Features shape: {features.shape}")
        logger.info(f"Target shape: {target.shape}")
        
        return features, target
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features using StandardScaler"""
        logger.info("Scaling features...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled, None
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """Split data into train and test sets"""
        logger.info(f"Splitting data with test_size={test_size}")
        
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def save_preprocessor(self, save_dir: str):
        """Save the fitted preprocessor"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(save_dir, 'scaler.pkl'))
        
        # Save feature names and target
        metadata = {
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'label_encoders': self.label_encoders
        }
        joblib.dump(metadata, os.path.join(save_dir, 'preprocessor_metadata.pkl'))
        
        logger.info(f"Preprocessor saved to {save_dir}")
    
    def load_preprocessor(self, save_dir: str):
        """Load a fitted preprocessor"""
        # Load scaler
        self.scaler = joblib.load(os.path.join(save_dir, 'scaler.pkl'))
        
        # Load metadata
        metadata = joblib.load(os.path.join(save_dir, 'preprocessor_metadata.pkl'))
        self.feature_names = metadata['feature_names']
        self.target_name = metadata['target_name']
        self.label_encoders = metadata['label_encoders']
        
        logger.info(f"Preprocessor loaded from {save_dir}")


def main():
    """Main preprocessing pipeline"""
    # Initialize processor
    processor = McDonaldsDataProcessor()
    
    # Load and process data
    raw_df = processor.load_raw_data('../data/mc-donalds.csv')
    clean_df = processor.clean_and_restructure(raw_df)
    pivot_df = processor.create_pivot_features(clean_df)
    engineered_df = processor.engineer_features(pivot_df)
    
    # Prepare for ML
    X, y = processor.prepare_ml_data(engineered_df)
    
    # Split data
    X_train, X_test, y_train, y_test = processor.split_data(X, y)
    
    # Scale features
    X_train_scaled, X_test_scaled = processor.scale_features(X_train, X_test)
    
    # Save processed data
    os.makedirs('../../data/processed', exist_ok=True)
    
    np.save('../../data/processed/X_train_scaled.npy', X_train_scaled)
    np.save('../../data/processed/X_test_scaled.npy', X_test_scaled)
    np.save('../../data/processed/y_train.npy', y_train.values)
    np.save('../../data/processed/y_test.npy', y_test.values)
    
    # Save preprocessor
    processor.save_preprocessor('../../models/preprocessor')
    
    # Save feature DataFrames for reference
    X_train.to_csv('../../data/processed/X_train.csv')
    X_test.to_csv('../../data/processed/X_test.csv')
    y_train.to_csv('../../data/processed/y_train.csv')
    y_test.to_csv('../../data/processed/y_test.csv')
    
    print(f"Preprocessing complete!")
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    print(f"Target: {processor.target_name}")


if __name__ == "__main__":
    main()