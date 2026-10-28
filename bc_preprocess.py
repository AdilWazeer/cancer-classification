import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler



class FeaturePreparation:
    BINARY_COLS  = ['family_history', 'menopause_status', 'tumor_location']
    ORDINAL_COLS = {
        'alcohol_intake':          ['Low', 'Moderate', 'High'],
        'physical_activity_level': ['Low', 'Moderate', 'High'],
        'diet_quality':            ['Poor', 'Average', 'Good'],
        'smoking_status':          ['Never', 'Former', 'Current'],
    }
    NOMINAL_COLS = ['genetic_mutation']

    def __init__(self, df):
        self.df = df.copy()
        self.scaler = StandardScaler() # standardise to mean=0, std=1 for better model convergence
        self.transformer = self._build_transformer()

    def _build_transformer(self):
        ord_keys = list(self.ORDINAL_COLS.keys())
        ord_order = list(self.ORDINAL_COLS.values())
        numeric_cols = self.df.select_dtypes(include='number').columns.drop('patient_id').tolist()

        return ColumnTransformer(transformers=[ # apply different transformations to different column groups
                ('binary', OrdinalEncoder(), self.BINARY_COLS),
                ('ordinal', OrdinalEncoder(categories=ord_order), ord_keys),
                ('nominal', OneHotEncoder(drop='first', sparse_output=False), self.NOMINAL_COLS),
                ('scale', self.scaler, numeric_cols),
            ], remainder='passthrough')

    def prepare(self):
        X = self.df.drop(columns=['diagnosis', 'patient_id']) # drop target and identifer
        y = self.df['diagnosis'].map({'Benign': 0, 'Malignant': 1}).to_numpy() 

        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                        test_size=0.2, stratify=y, random_state=2026, shuffle=True)
        X_train = self.transformer.fit_transform(X_train)
        X_test = self.transformer.transform(X_test)

        feature_names = self.transformer.get_feature_names_out()
        np.save('./splits/feature_names.npy', feature_names)
        feature_index = {name: i for i, name in enumerate(feature_names)}
        np.save('./splits/feature_index.npy', feature_index)

        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = pd.read_csv('./cleaned_breast_cancer_dataset.csv') 
    X_train, X_test, y_train, y_test = FeaturePreparation(df).prepare() 

    split_dir = './splits/'
    np.save(f'{split_dir}X_train.npy', X_train)
    np.save(f'{split_dir}X_test.npy',  X_test)
    np.save(f'{split_dir}y_train.npy', y_train)
    np.save(f'{split_dir}y_test.npy',  y_test)

    print(f"\n{'='*75}\nSPLIT SUMMARY\n{'='*75}")
    print(f"Train : X={X_train.shape}  |  Malignant={y_train.sum()}  Benign={(y_train==0).sum()}")
    print(f"Test  : X={X_test.shape}   |  Malignant={y_test.sum()}   Benign={(y_test==0).sum()}")

    