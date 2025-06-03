import matplotlib.pyplot as plt
import matplotlib.patches as patches



plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'

blip = {'10': 701, '01': 1338 , '11': 191}
git = {'10': 2121, '01': 1548, '11': 276}
ofa = {'10': 1343, '01': 1973, '11': 233}
oscar = {'10': 1456, '01': 1802, '11': 177}

fig, ax = plt.subplots(1, 4, figsize=(14, 7))

def draw_ellipse(ax, venn_data, title):
    ax.set_title(title, fontsize=27, pad=0, fontname='Times New Roman', fontweight='bold')

    ellipse1 = patches.Ellipse((0.35, 0.5), 0.6, 0.8, color='lightgreen', alpha=0.5)
    ellipse2 = patches.Ellipse((0.65, 0.5), 0.6, 0.8, color='lightblue', alpha=0.5)

    ax.add_patch(ellipse1)
    ax.add_patch(ellipse2)

    ax.text(0.2, 0.5, f"{venn_data['10']}", fontsize=20, ha='center', va='center',
            fontname='Times New Roman', fontweight='bold')
    ax.text(0.8, 0.5, f"{venn_data['01']}", fontsize=20, ha='center', va='center',
            fontname='Times New Roman', fontweight='bold')
    ax.text(0.5, 0.5, f"{venn_data['11']}", fontsize=20, ha='center', va='center',
            fontname='Times New Roman', fontweight='bold')

    ax.set_axis_off()

draw_ellipse(ax[0], blip, 'BLIP')
draw_ellipse(ax[1], git, 'GIT')
draw_ellipse(ax[2], ofa, 'OFA')
draw_ellipse(ax[3], oscar, 'OSCAR')

oamt_patch = patches.Patch(color='lightgreen', alpha=0.5, label='OAMT Errors')
rome_patch = patches.Patch(color='lightblue', alpha=0.5, label='ROME Errors')

legend = fig.legend(handles=[oamt_patch, rome_patch],
                    loc='center',
                    ncol=2,
                    fontsize=20,
                    frameon=False,
                    bbox_to_anchor=(0.5, 0.01),
                    prop={'family': 'Times New Roman', 'weight': 'bold','size':20})  # 设置字体属性


plt.tight_layout()
plt.subplots_adjust(bottom=0.05)
plt.savefig("venn.pdf", format="pdf", bbox_inches="tight")
plt.show()