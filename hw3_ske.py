from matplotlib import pyplot as plt
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

# 2.1.5 (e) Result Visualization 
# Visualize the predictions for each kernel-feature combination on the training set. Use color coding 
# for labels. Include this figure in your report.
def plot_kernel_feature_predictions(train_df, trained_models):
    feature_pairs = [
        ("CGPA", "SOP"),
        ("CGPA", "GRE Score"),
        ("SOP", "LOR"),
        ("LOR", "GRE Score"),
    ]
    kernels = ["linear", "rbf", "poly"]

    fig, axes = plt.subplots(len(feature_pairs), len(kernels), figsize=(15, 16))
    fig.suptitle("SVM Predictions on Training Set (by Kernel + Feature Pair)", y=1.02)

    for i, (f1, f2) in enumerate(feature_pairs):
        X = train_df[[f1, f2]].to_numpy(dtype=float)
        y = train_df["label"].to_numpy(dtype=int)

        # Mesh grid for decision regions
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 300),
            np.linspace(y_min, y_max, 300),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]

        for j, k in enumerate(kernels):
            ax = axes[i, j]
            model = trained_models[(k, (f1, f2))]

            Z = model.predict(grid).reshape(xx.shape)

            # Background = predicted class
            ax.contourf(xx, yy, Z, alpha=0.25, levels=[-0.5, 0.5, 1.5], cmap="coolwarm")

            # Points = true labels (color-coded)
            ax.scatter(X[y == 0, 0], X[y == 0, 1], s=18, c="tab:blue", label="label=0")
            ax.scatter(X[y == 1, 0], X[y == 1, 1], s=18, c="tab:orange", label="label=1")

            ax.set_title(f"{k} | {f1} vs {f2}")
            ax.set_xlabel(f1)
            ax.set_ylabel(f2)

    # Single legend for the full figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    plt.tight_layout()
    plt.show()



def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return float((y_true == y_pred).mean())

# 2.1.6 (f) Result Analysis 
# Determine the best feature-kernel combination based on training set figures. Validate the model on 
# the validation split and find the best performing combination with respect to accuracy. Initialize 
# my best model variable using your best model combination. Tune hyperparameter and aim for 
# 0.83 accuracy on our reserved test set. 
def select_best_model_and_set_global(loader: DataLoader):
    """
    Trains/tunes all kernel-feature combinations, evaluates on validation,
    selects best by validation accuracy, and sets global my_best_model.

    Returns:
        best_model: SVC
        best_info: dict with details
    """
    global my_best_model

    trainer = SVMTrainer()

    feature_pairs = [
        ("CGPA", "SOP"),
        ("CGPA", "GRE Score"),
        ("SOP", "LOR"),
        ("LOR", "GRE Score"),
    ]

    # Small manual hyperparameter grids
    grid = {
        "linear": [
            {"C": 0.1},
            {"C": 1.0},
            {"C": 10.0},
        ],
        "rbf": [
            {"C": 0.5, "gamma": "scale"},
            {"C": 1.0, "gamma": "scale"},
            {"C": 10.0, "gamma": "scale"},
            {"C": 10.0, "gamma": 0.5},
            {"C": 10.0, "gamma": 0.1},
        ],
        "poly": [
            {"C": 0.1, "degree": 3, "gamma": "scale", "coef0": 0.0},
            {"C": 1.0, "degree": 3, "gamma": "scale", "coef0": 0.0},
            {"C": 1.0, "degree": 3, "gamma": "scale", "coef0": 1.0},
            {"C": 10.0, "degree": 3, "gamma": "scale", "coef0": 1.0},
        ],
    }

    best = {
        "val_acc": -1.0,
        "kernel": None,
        "features": None,
        "params": None,
        "model": None,
    }

    # Evaluate each combo on validation
    for feats in feature_pairs:
        X_tr = loader.train_data[list(feats)].to_numpy(dtype=float)
        y_tr = loader.train_data["label"].to_numpy(dtype=int)

        X_va = loader.val_data[list(feats)].to_numpy(dtype=float)
        y_va = loader.val_data["label"].to_numpy(dtype=int)

        for kernel in ["linear", "rbf", "poly"]:
            for params in grid[kernel]:
                model = trainer.train(X_tr, y_tr, kernel=kernel, **params)
                pred = model.predict(X_va)
                acc = _accuracy(y_va, pred)

                if acc > best["val_acc"]:
                    best["val_acc"] = acc
                    best["kernel"] = kernel
                    best["features"] = feats
                    best["params"] = params
                    best["model"] = model

    # Retrain best on training split 
    X_tr_best = loader.train_data[list(best["features"])].to_numpy(dtype=float)
    y_tr_best = loader.train_data["label"].to_numpy(dtype=int)
    my_best_model = trainer.train(X_tr_best, y_tr_best, kernel=best["kernel"], **best["params"])

    return my_best_model, best

'''
Initialize my_best_model with the best model you found.
'''
my_best_model = SVC()

if __name__ == "__main__":
    print("Hello, World!")
    loader = DataLoader(data_path="./hw3-univ-app-data.csv")
    trained_models = train_models_for_feature_pairs(loader.train_data)

    support_vectors = get_support_vectors_for_models(trained_models)
    sv_linear_cgpa_sop = support_vectors[("linear", ("CGPA", "SOP"))]

    plot_kernel_feature_predictions(loader.train_data, trained_models)
    # plt.savefig("svm_kernel_feature_grid.png", dpi=300)

    best_model, best_info = select_best_model_and_set_global(loader)

    print("Best validation accuracy:", best_info["val_acc"])
    print("Best kernel:", best_info["kernel"])
    print("Best features:", best_info["features"])
    print("Best params:", best_info["params"])