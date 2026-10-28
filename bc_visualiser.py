# visualiser.py
from __future__ import annotations

import os
from matplotlib import patches
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
)


class ModelVisualiser:
    """
    Encapsulates all post-training plots for LR vs AdaBoost.

    Parameters
    ----------
    y_test          : array-like  — true test labels
    lr_pred         : array-like  — LR hard predictions
    ada_pred        : array-like  — AdaBoost hard predictions
    lr_proba        : array-like  — LR positive-class probabilities
    ada_proba       : array-like  — AdaBoost positive-class probabilities
    lr_thresh       : float       — operating threshold for LR (PR dot)
    ada_thresh      : float       — operating threshold for AdaBoost (PR dot)
    lr_cv           : dict        — cross_validate() output for LR
    ada_cv          : dict        — cross_validate() output for AdaBoost
    ada_model       : AdaBoostClassifier — fitted model (staged_predict_proba)
    lr_model        : LogisticRegression — fitted model (coefficients)
    X_tr            : array-like  — training features (plateau sweep)
    X_te            : array-like  — test features    (plateau sweep + threshold sweep)
    y_train         : array-like  — training labels  (plateau sweep)
    feature_names   : list[str]   — display names aligned to X_te columns
    min_recall      : float       — minimum recall constraint line (default 0.90)
    out_dir         : str         — root directory for saved figures
    """

    # ── shared palette ────────────────────────────────────────
    _COLOURS = {'lr': '#4C72B0', 'ada': '#DD8452'}
    _C_SWEEP  = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 25, 50]

    def __init__(
        self,
        y_test, lr_pred, ada_pred,
        lr_proba, ada_proba,
        lr_thresh, ada_thresh,
        lr_cv, ada_cv,
        ada_model, lr_model,
        X_tr, X_te, y_train,
        feature_names: list[str],
        min_recall: float = 0.90,
        out_dir:    str   = './figs/results/',
    ):
        self.y_test       = np.asarray(y_test)
        self.lr_pred      = np.asarray(lr_pred)
        self.ada_pred     = np.asarray(ada_pred)
        self.lr_proba     = np.asarray(lr_proba)
        self.ada_proba    = np.asarray(ada_proba)
        self.lr_thresh    = lr_thresh
        self.ada_thresh   = ada_thresh
        self.lr_cv        = lr_cv
        self.ada_cv       = ada_cv
        self.ada_model    = ada_model
        self.lr_model     = lr_model
        self.X_tr         = X_tr
        self.X_te         = X_te
        self.y_train      = np.asarray(y_train)
        self.feature_names = feature_names
        self.min_recall   = min_recall
        self.out_dir      = out_dir

        sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
        os.makedirs(self.out_dir, exist_ok=True)

    # ── public API ────────────────────────────────────────────

    def plot_all(self) -> None:
        """Render and save every figure, then print a summary."""
        self._fig_confusion_matrices()
        self._fig_roc_curves()
        self._fig_pr_curves()
        self._fig_cv_distribution()
        self._fig_plateau()
        self._fig_feature_importance()
        self._fig_threshold_sensitivity()
        print(f'\nAll figures saved to {self.out_dir}/')

    # ── private helpers ───────────────────────────────────────

    def _save(self, filename: str) -> None:
        path = os.path.join(self.out_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

    def _model_pairs(self) -> list[tuple]:
        """Return (label, proba, threshold, colour) for each model."""
        return [
            ('Logistic Regression', self.lr_proba,  self.lr_thresh,  self._COLOURS['lr']),
            ('AdaBoost',            self.ada_proba,  self.ada_thresh, self._COLOURS['ada']),
        ]

    # ─────────────────────────────────────────────────────────
    # FIG 1 — Confusion Matrices
    # ─────────────────────────────────────────────────────────
    def _fig_confusion_matrices(self) -> None:
            # Keep height at 6 to prevent the squishing we saw earlier
            fig, axes = plt.subplots(1, 2, figsize=(12, 6)) 
            total = len(self.y_test)

            # Using hex codes as you updated them
            GREEN = '#8fbc8f'
            RED   = '#B85C5C'

            # --- Label Determination Logic ---
            unique = np.unique(self.y_test)
            if len(unique) != 2:
                raise ValueError("y_test must contain exactly two classes")

            if np.issubdtype(unique.dtype, np.number):
                benign_label, malignant_label = 0, 1
            else:
                str_unique = [str(x).lower() for x in unique]
                if 'benign' in str_unique:
                    idx = str_unique.index('benign')
                    benign_label, malignant_label = unique[idx], unique[1 - idx]
                else:
                    sorted_unique = sorted(unique)
                    benign_label, malignant_label = sorted_unique[0], sorted_unique[1]

            class_order = [malignant_label, benign_label]
            
            for ax, name, pred in zip(axes, ['LR Confusion Matrix', 'AdaBoost Confusion Matrix'], [self.lr_pred, self.ada_pred]):
                cm = confusion_matrix(self.y_test, pred, labels=class_order)
                tp, fn, fp, tn = cm.ravel()

                for r in range(2):
                    for c in range(2):
                        color = GREEN if r == c else RED
                        rect  = patches.Rectangle((c, 1 - r), 1, 1, facecolor=color, edgecolor='none')
                        ax.add_patch(rect)

                counts = [[tp, fn], [fp, tn]]
                metric_names = [['Recall', 'Miss Rate'], ['FP Rate', 'Specificity']]
                malignant_total, benign_total = tp + fn, fp + tn

                for r in range(2):
                    for c in range(2):
                        count = counts[r][c]
                        row_total = malignant_total if r == 0 else benign_total
                        pct = count / row_total if row_total > 0 else 0
                        ax.text(c + 0.5, 1 - r + 0.5, f'{count}\n({pct:.1%} {metric_names[r][c]})',
                                ha='center', va='center', fontsize=12, fontweight='bold', color='white')

                # --- Labels and Formatting ---
                ax.set_xticks([0.5, 1.5])
                ax.set_xticklabels(['Malignant', 'Benign'], fontsize=11)
                ax.xaxis.set_label_position('top')
                ax.xaxis.tick_top()
                ax.set_xlabel('Predicted', fontsize=11, labelpad=10)
                
                ax.set_yticks([0.5, 1.5])
                ax.set_yticklabels(['Benign', 'Malignant'], fontsize=11)
                ax.set_ylabel('True', fontsize=11, labelpad=10)
                
                ax.set_title(name, fontsize=13, fontweight='bold', pad=20) 
                
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.tick_params(which='both', bottom=False, left=False, top=False)
                ax.set_xlim(0, 2)
                ax.set_ylim(0, 2)
                ax.set_aspect('equal')

                # Metrics calculation
                m_acc = (tp + tn) / total
                m_f1  = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
                m_pre = tp / (tp + fp) if (tp + fp) > 0 else 0

                # transform=ax.transAxes ensures this text stays locked to the bottom of the matrix
                ax.text(0.5, -0.12, f'Acc: {m_acc:.3f}  |  F1: {m_f1:.3f}  |  Precision: {m_pre:.3f}',
                        ha='center', va='top', fontsize=10, color='#444', transform=ax.transAxes)

            # --- CLEANUP & SPACING ---
            # w_pad increases the horizontal space between the two matrices
            # h_pad/pad help manage the top/bottom margins now that suptitle is gone
            fig.tight_layout(pad=3.0, w_pad=6.0)
            
            self._save('models_confusion_matrices.png')

    # ─────────────────────────────────────────────────────────
    # FIG 2 — ROC Curves
    # ─────────────────────────────────────────────────────────
    def _fig_roc_curves(self) -> None:
        fig, ax = plt.subplots(figsize=(7, 6))

        for name, proba, _, colour in self._model_pairs():
            fpr, tpr, _ = roc_curve(self.y_test, proba)
            auc          = roc_auc_score(self.y_test, proba)
            ax.plot(fpr, tpr, lw=2.5, color=colour, label=f'{name}  (AUC = {auc:.4f})')
            ax.fill_between(fpr, tpr, alpha=0.06, color=colour)

        ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='Random classifier')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves — LR vs AdaBoost', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        plt.tight_layout()
        self._save('models_roc_curves.png')

    # ─────────────────────────────────────────────────────────
    # FIG 3 — Precision-Recall Curves
    # ─────────────────────────────────────────────────────────
    def _fig_pr_curves(self) -> None:
        fig, ax = plt.subplots(figsize=(7, 6))

        for name, proba, thresh, colour in self._model_pairs():
            prec, rec, thresholds = precision_recall_curve(self.y_test, proba)
            ap          = average_precision_score(self.y_test, proba)
            chosen_idx  = np.argmin(np.abs(thresholds - thresh))
            ax.plot(rec, prec, lw=2.5, color=colour, label=f'{name}  (AP = {ap:.4f})')
            ax.scatter(
                rec[chosen_idx], prec[chosen_idx],
                s=100, color=colour, zorder=5,
                edgecolors='black', linewidths=0.8,
            )

        ax.axvline(
            x=self.min_recall, color='#4a9e6b', linestyle='--',
            lw=1.5, alpha=0.7, label=f'Min recall = {self.min_recall}',
        )
        ax.axhline(
            y=self.y_test.mean(), color='grey', linestyle='--',
            lw=1.2, alpha=0.6, label='No-skill baseline',
        )
        ax.set_xlabel('Recall (Malignant)', fontsize=12)
        ax.set_ylabel('Precision (Malignant)', fontsize=12)
        ax.set_title('Precision-Recall Curves — LR vs AdaBoost', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        plt.tight_layout()
        self._save('models_pr_curves.png')

    # ─────────────────────────────────────────────────────────
    # FIG 4 — CV Score Distribution
    # ─────────────────────────────────────────────────────────
    def _fig_cv_distribution(self) -> None:
        cv_data = pd.DataFrame({
            'LR — Accuracy':       self.lr_cv['test_accuracy'],
            'AdaBoost — Accuracy': self.ada_cv['test_accuracy'],
            'LR — F1':             self.lr_cv['test_f1'],
            'AdaBoost — F1':       self.ada_cv['test_f1'],
            'LR — AUC':            self.lr_cv['test_roc_auc'],
            'AdaBoost — AUC':      self.ada_cv['test_roc_auc'],
        })

        fig, ax = plt.subplots(figsize=(11, 5))
        sns.boxplot(
            data=cv_data,
            palette=[self._COLOURS['lr'], self._COLOURS['ada']] * 3,
            width=0.5, linewidth=1.5, ax=ax,
        )
        sns.stripplot(data=cv_data, color='black', size=7, jitter=True, alpha=0.6, ax=ax)

        for i, col in enumerate(cv_data.columns):
            ax.text(
                i, cv_data[col].mean() + 0.004,
                f'{cv_data[col].mean():.4f}',
                ha='center', fontsize=9, fontweight='bold',
            )

        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim([0.70, 1.0])
        ax.set_title('5-Fold CV Score Distribution — LR vs AdaBoost', fontsize=13, fontweight='bold')
        plt.xticks(rotation=15)
        plt.tight_layout()
        self._save('models_cv_comparison.png')

    # ─────────────────────────────────────────────────────────
    # FIG 5 — Performance Plateau
    # ─────────────────────────────────────────────────────────
    def _fig_plateau(self) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # AdaBoost: staged AUC on test set
        staged_auc = [
            roc_auc_score(self.y_test, p[:, 1])
            for p in self.ada_model.staged_predict_proba(self.X_te)
        ]
        peak_n   = staged_auc.index(max(staged_auc)) + 1
        peak_auc = max(staged_auc)

        axes[0].plot(range(1, len(staged_auc) + 1), staged_auc, lw=2, color='steelblue')
        axes[0].axhline(y=peak_auc, color='gray', linestyle='--', lw=1.2,
                        label=f'Peak AUC: {peak_auc:.4f}')
        axes[0].axvline(x=peak_n, color='tomato', linestyle=':', lw=1.2,
                        label=f'Peak at n={peak_n}')
        axes[0].set_xlabel('n_estimators')
        axes[0].set_ylabel('Test AUC')
        axes[0].set_title('AdaBoost: AUC vs n_estimators')
        axes[0].legend()

        # Logistic Regression: AUC across C values
        lr_aucs = [
            roc_auc_score(
                self.y_test,
                LogisticRegression(C=c, solver='saga', max_iter=5000)
                .fit(self.X_tr, self.y_train)
                .predict_proba(self.X_te)[:, 1],
            )
            for c in self._C_SWEEP
        ]
        peak_c_idx = lr_aucs.index(max(lr_aucs))
        peak_c     = self._C_SWEEP[peak_c_idx]
        axes[1].plot(range(len(self._C_SWEEP)), lr_aucs, marker='o', lw=2,
                     markersize=7, color='steelblue')
        axes[1].axhline(y=max(lr_aucs), color='gray', linestyle='--', lw=1.2,
                        label=f'Peak AUC: {max(lr_aucs):.4f}')
        axes[1].axvline(x=peak_c_idx, color='tomato', linestyle=':', lw=1.2,
                        label=f'Peak at C={peak_c}')
        axes[1].set_xticks(range(len(self._C_SWEEP)))
        axes[1].set_xticklabels([str(c) for c in self._C_SWEEP], rotation=45, ha='right')
        axes[1].set_xlabel('C value')
        axes[1].set_ylabel('Test AUC')
        axes[1].set_title('Logistic Regression: AUC vs C')
        axes[1].legend()

        plt.suptitle('Performance Plateau — Both Models', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save('plateau.png')

    # ─────────────────────────────────────────────────────────
    # FIG 6 — Feature Importance
    # ─────────────────────────────────────────────────────────
    def _fig_feature_importance(self) -> None:
        """
        Generates a standardized comparison of feature importance between 
        Logistic Regression (Coefficients) and AdaBoost (Gini Importance).
        """
        def get_merged_stats(raw_values, is_lr=True):
            merged = {}
            for name, val in zip(self.feature_names, raw_values):
                # 1. Standardize Names: Strip prefixes and handle nominal grouping
                clean_name = name.replace('scale__', '').replace('binary__', '').replace('ordinal__', '')
                if 'nominal__' in name:
                    # Group 'nominal__genetic_mutation_BRCA2' -> 'genetic_mutation'
                    clean_name = name.split('nominal__')[-1].rsplit('_', 1)[0]
                
                if is_lr:
                    # Use L2 Norm (Sum of Squares) for grouping to capture total magnitude 
                    # without directional cancellation
                    merged[clean_name] = merged.get(clean_name, 0) + (val**2)
                else:
                    # Sum Gini Importance for AdaBoost
                    merged[clean_name] = merged.get(clean_name, 0) + val
            
            # 2. Finalize LR values: square root the sum of squares to return to coefficient scale
            if is_lr:
                for k in merged:
                    merged[k] = np.sqrt(merged[k])

            # 3. Sort by magnitude (lowest to highest for horizontal bar chart display)
            sorted_keys = sorted(merged, key=lambda k: merged[k])
            return sorted_keys, [merged[k] for k in sorted_keys]

        # Process models
        lr_labels, lr_vals = get_merged_stats(self.lr_model.coef_[0], is_lr=True)
        ada_labels, ada_vals = get_merged_stats(self.ada_model.feature_importances_, is_lr=False)

        fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=False)

        # ── Left Plot: Logistic Regression (Magnitude of Impact) ────────────────
        axes[0].barh(lr_labels, lr_vals, color=self._COLOURS['lr'], height=0.7, alpha=0.9)
        axes[0].set_title('Logistic Regression\nGlobal Impact Magnitude ($|\\beta|$)', fontweight='bold', pad=15)
        axes[0].set_xlabel('Square Root of Sum of Squared Coefficients')
        
        for i, v in enumerate(lr_vals):
            axes[0].text(v + (max(lr_vals)*0.01), i, f'{v:.3f}', va='center', fontsize=9)

        # ── Right Plot: AdaBoost (Gini Importance) ──────────────────────────────
        axes[1].barh(ada_labels, ada_vals, color=self._COLOURS['ada'], height=0.7, alpha=0.9)
        axes[1].set_title('AdaBoost\nFeature Importance (Gini)', fontweight='bold', pad=15)
        axes[1].set_xlabel('Mean Decrease in Impurity')
        
        for i, v in enumerate(ada_vals):
            axes[1].text(v + (max(ada_vals)*0.01), i, f'{v:.3f}', va='center', fontsize=9)

        # ── Global Refinements ──────────────────────────────────────────────────
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='x', linestyle='--', alpha=0.3)
            # Ensure x-axis starts at 0 for honest magnitude comparison
            ax.set_xlim(left=0, right=ax.get_xlim()[1] * 1.1) 

        plt.suptitle('Model Feature Interpretation: Statistical vs Ensemble', 
                    fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()
        self._save('models_feature_importance.png')

    # ─────────────────────────────────────────────────────────
    # FIG 7 — Threshold Sensitivity
    # ─────────────────────────────────────────────────────────
    def _fig_threshold_sensitivity(self) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, (name, proba, chosen_thresh, colour) in zip(axes, self._model_pairs()):
            colour = '#4C72B0' # looks better
            thresholds = np.linspace(0.01, 0.99, 200)
            recalls, precisions, specificities = [], [], []

            for t in thresholds:
                pred = (proba >= t).astype(int)
                tn, fp, fn, tp = confusion_matrix(self.y_test, pred).ravel()
                precisions.append(tp / (tp + fp) if tp + fp >= 5 else np.nan)
                recalls.append(tp / (tp + fn) if tp + fn >= 5 else np.nan)
                specificities.append(tn / (tn + fp) if tn + fp >= 5 else np.nan)

            ax.plot(thresholds, recalls,       lw=2,   color='#2ecc71', label='Recall')
            ax.plot(thresholds, precisions,    lw=2,   color='#3498db', label='Precision')
            ax.plot(thresholds, specificities, lw=2,   color='#9b59b6', label='Specificity')

            # operating point
            ax.axvline(chosen_thresh, color=colour, linestyle='--', lw=1.8,
                       label=f'Chosen threshold ({chosen_thresh:.3f})')

            # min recall floor
            ax.axhline(self.min_recall, color='#e74c3c', linestyle=':', lw=1.4,
                       alpha=0.8, label=f'Min recall = {self.min_recall}')

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel('Threshold', fontsize=11)
            ax.set_ylabel('Score', fontsize=11)
            ax.set_title(f'{name}\nThreshold Sensitivity', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9, loc='center left')

        plt.suptitle('Threshold Sensitivity — LR vs AdaBoost', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save('models_threshold_sensitivity.png')


