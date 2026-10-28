from itertools import product
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from bc_result_manager import ModelEvaluator

# ── Results management and visualisation ───────────────────────────────
RM = ModelEvaluator()
SPLIT_DIR = './splits/'

# ── Model parameters ───────────────────────────────────────────────
MIN_RECALL = 0.9  # clinical floor — helps prioritise catching cancers

SELECTED_FEATURES = [  # numeric features only, justified by EDA
    'scale__mean_radius', 'scale__tumor_size_log', 'scale__mean_texture',
    'scale__mean_area', 'scale__age', 'scale__smoothness', 'scale__symmetry',
    'scale__concavity', 'scale__compactness', 'scale__mean_perimeter', 
]
 

# C: regularisation strength, helps prevent overfitting
# solver: controls the descent algo, saga's better for larger datasets
# class_weight: tries w/wo giving more weight to minority class (malignant)
# max_iter: max iterations for descent (sklearn alerts if not converged)
LR_PARAM_GRID = {
    'C': [0.001, 0.005, 0.01, 0.1, 0.5, 1],
    'solver': ['lbfgs', 'saga'],
    'class_weight': ['balanced', None],
    'max_iter': [500],
} 

# estimator: base decision tree, tried with different depths / weights
# max_depth: controls tree depth, shallow reduces overfit but may underfit
# class_weight: tries w/wo giving more weight to minority class (malignant)
# n_estimators: number of refinements rounds correcting prev errors
# learning_rate: weak learner contribution, helps prevent overfitting
ADA_PARAM_GRID = {
    'estimator': [DecisionTreeClassifier(max_depth=d, class_weight=cw)
        for d, cw in product([1, 2, 3, 4], ['balanced', None])],
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.005, 0.01, 0.05],
}


# ── Main training pipelines ───────────────────────────────────────────────
class ModelArchitect:
    def __init__(self, classifier, param_grid):
        self.cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=2026)
        self.search = GridSearchCV(classifier, param_grid,
            scoring=self._precision_at_recall_floor, cv=self.cv, n_jobs=-1  
        )  # selects params that max precision @ recall ≥ MIN_RECALL

    def run_model(self, name, X_tr, y_tr, X_te, y_te):
        model = self.search.fit(X_tr, y_tr).best_estimator_ 
        train_proba = model.predict_proba(X_tr)[:, 1] #
        thresh = self._find_best_threshold(y_tr, train_proba) #
        
        proba = model.predict_proba(X_te)[:, 1] # gets proba predictions
        pred = (proba >= thresh).astype(int)  # final predictions 

        RM.evaluate(name, self, X_tr, y_tr, y_te, proba, thresh)
        RM.save_artifacts(name, model, thresh, SELECTED_FEATURES)
        return model, proba, thresh, pred 

    def _find_best_threshold(self, y_true, y_proba):
        precision, recall, th = precision_recall_curve(y_true, y_proba)
        precision, recall = precision[:-1], recall[:-1]  # remove sentinel
        mask = (recall >= MIN_RECALL)  # retrieve points that pass MIN_RECALL
        if not mask.any():  # no valid points, fallback to highest recall
            return th[np.argmax(recall)]
        return th[mask][np.argmax(precision[mask])]  # max P @ MIN_RECALL
    
    
    @staticmethod
    def _precision_at_recall_floor(estimator, X, y_true):
        y_proba = estimator.predict_proba(X)[:, 1]
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        precision, recall = precision[:-1], recall[:-1]  # remove sentinel
        mask = (recall >= MIN_RECALL) 
        return np.max(precision[mask]) if mask.any() else 0.0 # reward only high P @ min R
    


def logistic_regression_pipeline(X_tr, y_tr, X_te, y_te):
    ma = ModelArchitect(LogisticRegression(random_state=2026), LR_PARAM_GRID)
    return ma.run_model('Logistic_Regression', X_tr, y_tr, X_te, y_te)


def adaboost_pipeline(X_tr, y_tr, X_te, y_te):
    ma = ModelArchitect(AdaBoostClassifier(random_state=2026), ADA_PARAM_GRID)
    return ma.run_model('AdaBoost', X_tr, y_tr, X_te, y_te)


if __name__ == '__main__':
    X_tr, y_tr = np.load(f'{SPLIT_DIR}X_train.npy'), np.load(f'{SPLIT_DIR}y_train.npy')
    X_te, y_te = np.load(f'{SPLIT_DIR}X_test.npy'), np.load(f'{SPLIT_DIR}y_test.npy')
    feature_index = np.load(f'{SPLIT_DIR}feature_index.npy', allow_pickle=True).item()
    # SELECTED_FEATURES = list(feature_index.keys())

    top_idx = [feature_index[f] for f in SELECTED_FEATURES]
    # X_tr, X_te = X_tr[:, top_idx], X_te[:, top_idx]

    lr, lr_proba, lr_thresh, lr_pred = logistic_regression_pipeline(X_tr, y_tr, X_te, y_te)
    ada, ada_proba, ada_thresh, ada_pred = adaboost_pipeline(X_tr, y_tr, X_te, y_te)

    # RM.generate_visuals(y_te, y_tr, X_tr, X_te, features = SELECTED_FEATURES,
    #     lr_results = {'pred': lr_pred, 'proba': lr_proba, 'thresh': lr_thresh, 'model': lr},
    #     ada_results = {'pred': ada_pred, 'proba': ada_proba, 'thresh': ada_thresh, 'model': ada},)

