import numpy as np
import pandas as pd
from sklearn.svm import SVC

'''
Problem: University Admission Classification using SVMs

Instructions:
1. Do not use any additional libraries. Your code will be tested in a pre-built environment with only 
   the library specified in question instruction available. Importing additional libraries will result in 
   compilation errors and you will lose marks.

2. Fill in the skeleton code precisely as provided. You may define additional 
   default arguments or helper functions if necessary, but ensure the input/output format matches.
'''
class DataLoader:
    '''
    Put your call to class methods in the __init__ method. Autograder will call your __init__ method only. 
    '''
    
    def __init__(self, data_path: str):
        """
        Initialize data processor with paths to train dataset. You need to have train and validation sets processed.
        
        Args:
            data_path: absolute path to your data file
        """
        self.train_data = pd.DataFrame()
        self.val_data = pd.DataFrame()
        
        # TODO：complete your dataloader here!
        
        # read data
        df = pd.read_csv(data_path)

        # create the label
        df = self.create_binary_label(df)

        # Drop non-feature columns
        drop_cols = []
        if "Serial No." in df.columns:
            drop_cols.append("Serial No.")
        if "Chance of Admit" in df.columns:
            drop_cols.append("Chance of Admit")
        df = df.drop(columns=drop_cols)

        # 80/20 split
        idx = np.arange(len(df))
        split = int(0.8 * len(df))
        train_idx, val_idx = idx[:split], idx[split:]

        self.train_data = df.iloc[train_idx].reset_index(drop=True)
        self.val_data = df.iloc[val_idx].reset_index(drop=True)

        # Standardize features using TRAIN stats only
        self.feature_cols = [c for c in df.columns if c != "label"]
        means = self.train_data[self.feature_cols].mean()
        stds = self.train_data[self.feature_cols].std().replace(0.0, 1.0)

        self.train_data[self.feature_cols] = (self.train_data[self.feature_cols] - means) / stds
        self.val_data[self.feature_cols] = (self.val_data[self.feature_cols] - means) / stds

    
    def create_binary_label(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Create a binary label for the training data.
        '''
        med = df["Chance of Admit"].median()
        df = df.copy()
        df["label"] = (df["Chance of Admit"] > med).astype(int)
        return df

class SVMTrainer:
    def __init__(self):
        pass

    def train(self, X_train: np.ndarray, y_train: np.ndarray, kernel: str, **kwargs) -> SVC:
        '''
        Train the SVM model with the given kernel and parameters.

        Parameters:
            X_train: Training features
            y_train: Training labels
            kernel: Kernel type
            **kwargs: Additional arguments you may use
        Returns:
            SVC: Trained sklearn.svm.SVC model
        '''
        model = SVC(kernel=kernel, **kwargs)
        model.fit(X_train, y_train)
        return model         

    def get_support_vectors(self,model: SVC) -> np.ndarray:
        '''
        Get the support vectors from the trained SVM model.
        '''
        return model.support_vectors_

 # Initialize 3 different SVM models with the following kernels
svm_linear = SVC(kernel="linear")
svm_rbf = SVC(kernel="rbf")
svm_poly3 = SVC(kernel="poly", degree=3) 

# 2.1.3 (c) Feature Selection and Model Training
# Train each kernel (linear/rbf/poly3) with each requested feature pair.

def train_models_for_feature_pairs(
    train_df: pd.DataFrame,
    feature_pairs=None,
    # random_state: int = 42
):
    """
    Returns:
        models: dict with keys (kernel, (feat1, feat2)) and values = trained SVC
    """
    if feature_pairs is None:
        feature_pairs = [
            ("CGPA", "SOP"),
            ("CGPA", "GRE Score"),
            ("SOP", "LOR"),
            ("LOR", "GRE Score"),
        ]

    trainer = SVMTrainer()

    kernels = [
        ("linear", dict()),                 
        ("rbf", dict()),                    
        ("poly", dict(degree=3)), 
    ]

    models = {}

    y_train = train_df["label"].to_numpy(dtype=int)

    for feat1, feat2 in feature_pairs:
        X_train = train_df[[feat1, feat2]].to_numpy(dtype=float)

        for kernel_name, params in kernels:
            model = trainer.train(X_train, y_train, kernel=kernel_name, **params)
            models[(kernel_name, (feat1, feat2))] = model

    return models

# 2.1.4 (d) Support Vectors
# Identify support vectors for each trained model + feature combination.

def get_support_vectors_for_models(trained_models: dict) -> dict:
    """
    Args:
        trained_models: dict from train_models_for_feature_pairs()
                       keys: (kernel_name, (feat1, feat2))
                       values: trained SVC

    Returns:
        support_vectors: dict with same keys as trained_models,
                         values: np.ndarray of support vectors (in that feature space)
    """
    trainer = SVMTrainer()
    support_vectors = {}

    for key, model in trained_models.items():
        support_vectors[key] = trainer.get_support_vectors(model)

    return support_vectors

'''
Initialize my_best_model with the best model you found.
'''
my_best_model = SVC()

if __name__ == "__main__":
    print("Hello, World!")