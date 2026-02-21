"""
PASO 9 — Evaluación final en test set
"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    average_precision_score, roc_auc_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score, cohen_kappa_score,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve,
)


def evaluate_model(model, X_test_scaled, y_test, threshold, plot=True):
    """
    Calcula métricas de desempeño en el conjunto de test retenido.

    Parámetros
    ----------
    model : modelo ajustado.
    X_test_scaled : pd.DataFrame  (ya estandarizado).
    y_test : pd.Series
    threshold : float  (obtenido en el paso de optimización).
    plot : bool  — genera matriz de confusión, curva PR y ROC.

    Retorna
    -------
    metrics : dict
    y_test_proba : np.array
    y_test_pred  : np.array
    """
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    metrics = {
        'AUPRC':             average_precision_score(y_test, y_proba),
        'AUROC':             roc_auc_score(y_test, y_proba),
        'Balanced_Accuracy': balanced_accuracy_score(y_test, y_pred),
        'F1_Score':          f1_score(y_test, y_pred, zero_division=0),
        'Precision':         precision_score(y_test, y_pred, zero_division=0),
        'Recall':            recall_score(y_test, y_pred, zero_division=0),
        'Cohen_Kappa':       cohen_kappa_score(y_test, y_pred),
        'Threshold':         threshold,
    }

    print(f"{'='*60}")
    print("EVALUACIÓN FINAL EN TEST SET")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k:22s}: {v:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Clase 0', 'Clase 1'], digits=4))

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Pred 0', 'Pred 1'],
                    yticklabels=['Real 0', 'Real 1'], ax=axes[0])
        axes[0].set_title('Matriz de Confusión', fontweight='bold')

        # Precision-Recall
        prec_c, rec_c, _ = precision_recall_curve(y_test, y_proba)
        axes[1].plot(rec_c, prec_c, linewidth=2)
        axes[1].axhline(y_test.mean(), color='red', linestyle='--',
                        label=f'Baseline: {y_test.mean():.2f}')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title(f'Precision-Recall (AUPRC={metrics["AUPRC"]:.3f})',
                          fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        axes[2].plot(fpr, tpr, linewidth=2)
        axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[2].set_xlabel('FPR')
        axes[2].set_ylabel('TPR')
        axes[2].set_title(f'ROC (AUROC={metrics["AUROC"]:.3f})', fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return metrics, y_proba, y_pred
