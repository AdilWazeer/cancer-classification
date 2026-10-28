import json
import joblib
from matplotlib.pylab import f
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import cross_validate
from bc_visualiser import ModelVisualiser


class ModelEvaluator:
    def __init__(self):
        self.cv_score_list = []

    def evaluate(self, name, ma, X_tr, y_tr, y_te, proba, thresh):
        model, best_params, cv = ma.search.best_estimator_, ma.search.best_params_, ma.cv
        pred = (proba >= thresh).astype(int) # Derived from proba and thresh
        cv_scores = cross_validate(model, X_tr, y_tr, cv=cv, scoring=['accuracy', 'f1', 'roc_auc'])
        self.cv_score_list.append(cv_scores)
        tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
        accuracy, f1, auc = accuracy_score(y_te, pred), f1_score(y_te, pred), roc_auc_score(y_te, proba)

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        specificity = tn / (tn + fp) if tn + fp > 0 else 0
        cv_auc, cv_f1, cv_acc = cv_scores['test_roc_auc'], cv_scores['test_f1'], cv_scores['test_accuracy']
        
        print(f"\n{'='*75}\n"
            f"{name.upper()}\t(AUC: {auc:.3f}  |  Threshold: {thresh:.3f}  |  F1: {f1:.3f})"
            f"\n{'='*75}")
        print(f"Cancers Caught   : {tp:>3} / {tp+fn}  — {recall*100:.1f}% recall")
        print(f"Overall Accuracy : {tp+tn:>3} / {len(y_te)}  — {accuracy*100:.1f}% accuracy")
        print(f"Correct Flags    : {tp:>3} / {tp+fp}  — {precision*100:.1f}% precision")
        print(f"Healthy Cleared  : {tn:>3} / {tn+fp}  — {specificity*100:.1f}% specificity")
        print(f"CV (mean ± std)  : AUC {cv_auc.mean():.3f} ± {cv_auc.std():.3f}  |  F1 {cv_f1.mean():.3f} ± {cv_f1.std():.3f}  |  Acc {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
        print(f"Best Hyperparams : {best_params}")


    @staticmethod
    def save_artifacts(name, model, thresh, features, out_dir='./models/'):
        slug = name.lower().replace(' ', '_')
        joblib.dump(model, f'{out_dir}model_{name}_all.pkl')
        json.dump(
            {'threshold': round(thresh, 6), 'features': features},
            open(f'{out_dir}{slug}_meta_all.json', 'w'),
            indent=4
        )

    def generate_visuals(self, y_test, y_train, X_tr, X_te, lr_results, ada_results, features):
        lr_cv, ada_cv = self.cv_score_list
        viz = ModelVisualiser(
            y_test=y_test,
            y_train=y_train,
            X_tr=X_tr,
            X_te=X_te,
            feature_names=features,
            # Unpacking LR
            lr_pred=lr_results["pred"],
            lr_proba=lr_results["proba"],
            lr_thresh=lr_results["thresh"],
            lr_cv=lr_cv,
            lr_model=lr_results["model"],
            # Unpacking Ada
            ada_pred=ada_results["pred"],
            ada_proba=ada_results["proba"],
            ada_thresh=ada_results["thresh"],
            ada_cv=ada_cv,
            ada_model=ada_results["model"],
        )
        viz.plot_all()
