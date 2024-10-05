import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LEGEND_FONTSIZE = 12
X_TITLE_FONTSIZE = 16
Y_TITLE_FONTSIZE = 16
TITLE_FONTSIZE = 18
CLIP_FONTSIZE = 18
NUM_FONTSIZE = 16
FIGSIZE = (4.6, 4.0)

save_dir = "fewshot_curves"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

path = "AWT_results.xlsx"  # this is the excel file containing the results (like the one we released)
file = pd.read_excel(path, sheet_name="few-shot")

datasets = [
    "OxfordPets", "Flowers102", "FGVCAircraft", "DTD",
    "EuroSAT", "StanfordCars", "Food101", "SUN397",
    "Caltech101", "UCF101", "ImageNet"
]

shots = [1, 2, 4, 8, 16]

COLORS = {
    "zs": "C4",
    "provp": "#FF3E96",
    "ape": "#8470FF",
    "coop": "#4a86e8",
    "plotpp": "C2",
    "promptsrc": "#ff9900",
    "awt": "C3"
}
MS = 3
ALPHA = 1
plt.rcParams.update({"font.size": NUM_FONTSIZE})

average = {
    "zs": 0.,
    "coop": np.array([0., 0., 0., 0., 0.]),
    "plotpp": np.array([0., 0., 0., 0., 0.]),
    "promptsrc": np.array([0., 0., 0., 0., 0.]),
    "awt": np.array([0., 0., 0., 0., 0.]),
    "ape": np.array([0., 0., 0., 0., 0.]),
    "provp": np.array([0., 0., 0., 0., 0.])
}

for dataset in datasets:
    print(f"Processing {dataset} ...")

    zs = file[dataset][0]

    coop = file[dataset][2:7]
    coop = [float(num) for num in coop]

    plotpp = file[dataset][7:12]
    plotpp = [float(num) for num in plotpp]

    promptsrc = file[dataset][12:17]
    promptsrc = [float(num) for num in promptsrc]

    provp = file[dataset][17:22]
    provp = [float(num) for num in provp]

    ape = file[dataset][22:27]
    ape = [float(num) for num in ape]

    awt = file[dataset][27:32]
    awt = [float(num) for num in awt]

    average["zs"] += zs
    average["coop"] += np.array(coop)
    average["plotpp"] += np.array(plotpp)
    average["promptsrc"] += np.array(promptsrc)
    average["provp"] += np.array(provp)
    average["ape"] += np.array(ape)
    average["awt"] += np.array(awt)

    # Plot
    values = [zs]
    values += provp
    values += ape
    values += coop
    values += plotpp
    values += promptsrc
    values += awt
    val_min, val_max = min(values), max(values)
    diff = val_max - val_min
    val_bot = val_min - diff*0.05
    val_top = val_max + diff*0.05

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.set_xticks([0] + shots)
    ax.set_xticklabels([0] + shots)
    ax.set_xlabel("# shots per class", fontsize=X_TITLE_FONTSIZE)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=Y_TITLE_FONTSIZE)
    ax.grid(axis="x", color="#DCDCDC", linewidth=1)
    ax.axhline(zs, color="#DCDCDC", linewidth=1)
    # ax.set_title(dataset)
    ax.set_title(dataset, fontweight="bold", fontsize=TITLE_FONTSIZE)
    ax.set_ylim(val_bot, val_top)

    ax.plot(
        0, zs,
        marker="*",
        markersize=MS*4,
        color=COLORS["zs"],
        markerfacecolor='white',
        markeredgecolor=COLORS['zs'],
        alpha=ALPHA
    )
    ax.plot(
        shots, coop,
        marker="s",
        markersize=MS,
        color=COLORS["coop"],
        label="CoOp",
        alpha=ALPHA,
        linestyle='--'
    )
    ax.plot(
        shots, plotpp,
        marker="s",
        markersize=MS,
        color=COLORS["plotpp"],
        label="PLOT++",
        alpha=ALPHA,
        linestyle=':'
    )
    ax.plot(
        shots, promptsrc,
        marker="s",
        markersize=MS,
        color=COLORS["promptsrc"],
        label="PromptSRC",
        alpha=ALPHA,
        linestyle='--'
    )
    ax.plot(
        shots, ape,
        marker="s",
        markersize=MS,
        color=COLORS["ape"],
        label="APE-T",
        alpha=ALPHA,
        linestyle='-.'
    )
    ax.plot(
        shots, provp,
        marker="s",
        markersize=MS,
        color=COLORS["provp"],
        label="ProVP-Ref",
        alpha=ALPHA,
        linestyle='-.'
    )
    ax.plot(
        shots, awt,
        marker="*",
        markersize=MS*3.5,
        color=COLORS["awt"],
        markerfacecolor='#FFD700',
        markeredgecolor=COLORS["awt"],
        label="AWT",
        alpha=ALPHA
    )

    ax.text(-0.66, zs+diff*0.04, "CLIP", color=COLORS["zs"], fontsize=CLIP_FONTSIZE)
    ax.legend(loc="lower right", prop={'size': LEGEND_FONTSIZE})

    fig.savefig(f"{save_dir}/{dataset}.pdf", bbox_inches="tight")


# Plot
average = {k: v/len(datasets) for k, v in average.items()}
zs = average["zs"]
provp = list(average["provp"])
ape = list(average["ape"])
coop = list(average["coop"])
plotpp = list(average["plotpp"])
promptsrc = list(average["promptsrc"])
awt = list(average["awt"])

values = [zs]
values += provp
values += ape
values += coop
values += plotpp
values += promptsrc
values += awt
val_min, val_max = min(values), max(values)
diff = val_max - val_min
val_bot = val_min - diff*0.05
val_top = val_max + diff*0.05

fig, ax = plt.subplots(figsize=FIGSIZE)

ax.set_xticks([0] + shots)
ax.set_xticklabels([0] + shots)
ax.set_xlabel("# shots per class", fontsize=X_TITLE_FONTSIZE)
ax.set_ylabel("Top-1 Accuracy (%)", fontsize=Y_TITLE_FONTSIZE)
ax.grid(axis="x", color="#DCDCDC", linewidth=1)
ax.axhline(zs, color="#DCDCDC", linewidth=1)
ax.set_title("Average over 11 datasets", fontweight="bold", fontsize=TITLE_FONTSIZE)
ax.set_ylim(val_bot, val_top)

ax.plot(
    0, zs,
    marker="*",
    markersize=MS*4,
    color=COLORS["zs"],
    markerfacecolor='white',
    markeredgecolor=COLORS['zs'],
    alpha=ALPHA
)
ax.plot(
    shots, coop,
    marker="s",
    markersize=MS,
    color=COLORS["coop"],
    label="CoOp",
    alpha=ALPHA,
    linestyle='--'
)
ax.plot(
    shots, plotpp,
    marker="s",
    markersize=MS,
    color=COLORS["plotpp"],
    label="PLOT++",
    alpha=ALPHA,
    linestyle=':'
)
ax.plot(
    shots, promptsrc,
    marker="s",
    markersize=MS,
    color=COLORS["promptsrc"],
    label="PromptSRC",
    alpha=ALPHA,
    linestyle='--'
)
ax.plot(
    shots, ape,
    marker="s",
    markersize=MS,
    color=COLORS["ape"],
    label="APE-T",
    alpha=ALPHA,
    linestyle='-.'
)
ax.plot(
    shots, provp,
    marker="s",
    markersize=MS,
    color=COLORS["provp"],
    label="ProVP-Ref",
    alpha=ALPHA,
    linestyle='-.'
)
ax.plot(
    shots, awt,
    marker="*",
    markersize=MS*3.5,
    color=COLORS["awt"],
    markerfacecolor='#FFD700',
    markeredgecolor=COLORS["awt"],
    label="AWT",
    alpha=ALPHA
)

ax.text(-0.66, zs+diff*0.04, "CLIP", color=COLORS["zs"], fontsize=CLIP_FONTSIZE)
ax.legend(loc="lower right", prop={'size': LEGEND_FONTSIZE})

fig.savefig(f"{save_dir}/average.pdf", bbox_inches="tight")
