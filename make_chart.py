import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
r = json.load(open("results/logreg_results.json"))["controlled_80_20"]
names = list(r); train = [r[k]["train"] for k in names]; test = [r[k]["test"] for k in names]
x = np.arange(len(names)); w = .36
fig, ax = plt.subplots(figsize=(8.2, 4), dpi=160)
ax.bar(x - w/2, train, w, color="#9ca3af", label="Train"); ax.bar(x + w/2, test, w, color="#1d4ed8", label="Test")
for i in range(len(names)):
    ax.text(x[i] - w/2, train[i] + .005, f"{train[i]:.3f}", ha="center", fontsize=8); ax.text(x[i] + w/2, test[i] + .005, f"{test[i]:.3f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([n.replace(", ", "\n") for n in names], fontsize=9); ax.set_ylim(.8, 1.04); ax.set_ylabel("Accuracy")
ax.set_title("Spark logistic regression, same 80/20 split (breast cancer, 569 rows)", fontsize=10.5); ax.legend(frameon=False, loc="lower right")
[ax.spines[s].set_visible(False) for s in ("top", "right")]
plt.tight_layout(); plt.savefig("assets/logreg_comparison.png")
