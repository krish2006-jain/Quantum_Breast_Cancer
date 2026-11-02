import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set writable cache dir for matplotlib
os.environ["MPLCONFIGDIR"] = str(Path.cwd() / ".mplconfig")

results_dir = Path("d:/quantumgroup/wdbc-qsvm/results")
results_dir.mkdir(exist_ok=True)

# -------------------------------------------------------------
# FLOWCHART 1: End-to-End System Pipeline
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 11), dpi=220)
ax.set_xlim(0, 14)
ax.set_ylim(0, 11)
ax.axis("off")

# Title
ax.text(7, 10.5, "End-to-End WDBC Classification & Quantum Kernel Pipeline", 
        ha="center", va="center", fontsize=15, fontweight="bold", color="#1E293B")

def draw_box(ax, x, y, w, h, title, subtitle="", bg_color="#EEF2F6", border_color="#334155", title_color="#0F172A"):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15,rounding_size=0.15",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=1.5
    )
    ax.add_patch(box)
    if subtitle:
        ax.text(x + w/2, y + h*0.62, title, ha="center", va="center", fontsize=10, fontweight="bold", color=title_color)
        ax.text(x + w/2, y + h*0.30, subtitle, ha="center", va="center", fontsize=8.5, color="#475569")
    else:
        ax.text(x + w/2, y + h/2, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=title_color)

def draw_arrow(ax, x1, y1, x2, y2, color="#64748B", style="->", text=""):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=1.8, shrinkA=3, shrinkB=3)
    )
    if text:
        ax.text((x1+x2)/2 + 0.15, (y1+y2)/2, text, fontsize=8, color="#475569", va="center")

# Step 1: Raw Data
draw_box(ax, 4.5, 9.3, 5.0, 0.75, "WDBC Dataset (UCI ID 17 / Scikit-Learn)", "569 Patients, 30 Continuous Features, 0 Missing Values", "#EFF6FF", "#3B82F6", "#1D4ED8")

# Arrow
draw_arrow(ax, 7.0, 9.3, 7.0, 8.5)

# Step 2: Ingestion & Split
draw_box(ax, 4.25, 7.7, 5.5, 0.8, "Data Cleaning & Stratified 80/20 Split", "Train: 455 Patients | Test: 114 Held-out Patients", "#F8FAFC", "#64748B")

# Split arrows
draw_arrow(ax, 7.0, 7.7, 7.0, 6.9)

# Step 3: Leakage-Free Preprocessing
draw_box(ax, 3.5, 6.1, 7.0, 0.8, "Leakage-Free Standard Scaling & PCA", "Fitted strictly on train fold; 30 features reduced to 4 Principal Components", "#F1F5F9", "#475569")

# Branching arrows
draw_arrow(ax, 5.0, 6.1, 3.2, 5.2, text="Classical")
draw_arrow(ax, 9.0, 6.1, 10.8, 5.2, text="Quantum")

# Branch A: Classical Suite (Left)
draw_box(ax, 0.8, 4.4, 4.8, 0.8, "Branch A: Classical ML Suite", "7 Algorithms: SVM, RF, XGB, LogReg, NaiveBayes, Tree", "#FEF3C7", "#D97706", "#92400E")
draw_arrow(ax, 3.2, 4.4, 3.2, 3.6)
draw_box(ax, 0.8, 2.8, 4.8, 0.8, "Hyperparameter Grid Search", "5-Fold Stratified Cross-Validation on Train Data", "#FFFBEB", "#F59E0B")
draw_arrow(ax, 3.2, 2.8, 3.2, 2.0)
draw_box(ax, 0.8, 1.2, 4.8, 0.8, "Classical Test Predictions", "14 Configurations (All 30 Features vs. 4 PCA Components)", "#FEF9C3", "#CA8A04")

# Branch B: Quantum Suite (Right)
draw_box(ax, 8.4, 4.4, 4.8, 0.8, "Branch B: Quantum Feature Engineering", "Angle Scaling to [0, pi] for Phase Rotations", "#F3E8FF", "#9333EA", "#6B21A8")
draw_arrow(ax, 10.8, 4.4, 10.8, 3.6)
draw_box(ax, 8.4, 2.8, 4.8, 0.8, "Feature Maps & Statevector Kernel", "ZFeatureMap (unentangled) vs. ZZFeatureMap (entangled)", "#FAF5FF", "#A855F7")
draw_arrow(ax, 10.8, 2.8, 10.8, 2.0)
draw_box(ax, 8.4, 1.2, 4.8, 0.8, "Quantum Support Vector Classifier", "SVC with Precomputed Quantum Overlap Kernel & C Tuning", "#F3E8FF", "#9333EA")

# Converge Arrows to Center Evaluation
draw_arrow(ax, 5.6, 1.6, 6.3, 0.6)
draw_arrow(ax, 8.4, 1.6, 7.7, 0.6)

# Bottom Final Evaluation & Synthesis Box
draw_box(ax, 3.0, 0.1, 8.0, 0.7, "Unified Model Benchmarking & Explainable AI", "Accuracy, Precision, Recall, F1, ROC-AUC + SHAP TreeExplainer & 5-Seed Stability", "#ECFDF5", "#059669", "#065F46")

plt.tight_layout()
plt.savefig(results_dir / "flowchart_system_architecture.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("Saved flowchart_system_architecture.png")

# -------------------------------------------------------------
# FLOWCHART 2: Quantum Kernel Mechanics Flowchart
# -------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(13, 5), dpi=220)
ax2.set_xlim(0, 13)
ax2.set_ylim(0, 5)
ax2.axis("off")

# Title
ax2.text(6.5, 4.6, "Quantum Kernel Construction & SVM Decision Architecture", 
         ha="center", va="center", fontsize=14, fontweight="bold", color="#1E293B")

# Step 1: Classical Vectors
draw_box(ax2, 0.5, 2.2, 2.2, 1.6, "Input Patient Data", "Vectors x and x'\n(4 PCA Components)", "#EFF6FF", "#3B82F6", "#1D4ED8")
draw_arrow(ax2, 2.7, 3.0, 3.4, 3.0)

# Step 2: Angle Mapping
draw_box(ax2, 3.4, 2.2, 2.3, 1.6, "MinMax Scaling", "Map components to\nangles theta in [0, pi]", "#F1F5F9", "#64748B")
draw_arrow(ax2, 5.7, 3.0, 6.4, 3.0)

# Step 3: Quantum Feature Map
draw_box(ax2, 6.4, 2.0, 2.6, 2.0, "Quantum Circuit U(x)", "Z / ZZ Feature Map\nHadamard + Rz Gates\nLinear CNOT Entanglement", "#F3E8FF", "#9333EA", "#6B21A8")
draw_arrow(ax2, 9.0, 3.0, 9.7, 3.0)

# Step 4: Quantum Overlap
draw_box(ax2, 9.7, 2.2, 2.8, 1.6, "Fidelity Kernel Overlap", "K(x, x') = |<phi(x)|phi(x')>|^2\nStatevector simulation", "#FEF3C7", "#D97706", "#92400E")

# Bottom connector to SVM
draw_arrow(ax2, 11.1, 2.2, 11.1, 1.4)
draw_box(ax2, 3.4, 0.4, 8.5, 0.9, "Precomputed Support Vector Machine (SVC)", "Solves quadratic programming on K matrix: f(x) = sgn(sum alpha_i y_i K(x_i, x) + b) -> Benign / Malignant", "#ECFDF5", "#059669", "#065F46")
draw_arrow(ax2, 11.1, 0.85, 11.9, 0.85)

# Output badge
box_out = patches.FancyBboxPatch(
    (11.9, 0.4), 0.9, 0.9,
    boxstyle="round,pad=0.1", facecolor="#10B981", edgecolor="#047857", lw=1.5
)
ax2.add_patch(box_out)
ax2.text(12.35, 0.85, "Diagnosis", ha="center", va="center", fontsize=9, fontweight="bold", color="white")

plt.tight_layout()
plt.savefig(results_dir / "flowchart_quantum_kernel.png", dpi=220, bbox_inches="tight")
plt.close(fig2)
print("Saved flowchart_quantum_kernel.png")

