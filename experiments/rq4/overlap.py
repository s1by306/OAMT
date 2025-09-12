

import ast
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_image_ids_from_list_file(list_file, tsv_file):
    # Read only the first line from the list file
    with open(list_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        # Safely parse the list using ast.literal_eval
        line_numbers = ast.literal_eval(first_line)

    # Read all lines from the TSV file
    with open(tsv_file, 'r', encoding='utf-8') as f:
        tsv_lines = f.readlines()

    # Extract image_id from the specified line numbers
    image_ids = []
    for i in line_numbers:
        if 0 <= i < len(tsv_lines):
            image_id = tsv_lines[i].strip().split('\t')[0]
            image_ids.append(image_id)
        else:
            print(f"Line number {i} is out of range, skipping.")

    return image_ids

import re

def extract_img_ids_from_OAMT_issue_file(file_path):
    img_ids = []

    # Regular expression to match image URLs and extract the numeric ID
    pattern = re.compile(r'(\d{12})\.jpg')

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Search for image ID in the line
            match = pattern.search(line)
            if match:
                img_id = match.group(1)  # Extract 12-digit image ID as string
                img_ids.append(img_id)

    return img_ids



def plot_model_venns(blip_sets, git_sets, ofa_sets, oscar_sets, save_path="../rq_results/overlap.pdf"):
    """
    Draw four Venn-like diagrams for BLIP, GIT, OFA, and OSCAR models.

    Parameters:
        blip_sets, git_sets, ofa_sets, oscar_sets: tuple of sets (set_A, set_B)
            Each tuple contains two sets of image IDs for two methods.
        save_path: str
            File path to save the generated plot as a PDF.
    """
    # Configure font
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.weight'] = 'bold'

    def compute_venn_counts(set_a, set_b):
        return {
            '10': len(set_a - set_b),   # Only in set A
            '01': len(set_b - set_a),   # Only in set B
            '11': len(set_a & set_b)    # In both sets
        }

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

    # Compute data for each model
    venn_blip = compute_venn_counts(*blip_sets)
    venn_git = compute_venn_counts(*git_sets)
    venn_ofa = compute_venn_counts(*ofa_sets)
    venn_oscar = compute_venn_counts(*oscar_sets)

    fig, ax = plt.subplots(1, 4, figsize=(14, 7))

    draw_ellipse(ax[0], venn_blip, 'BLIP')
    draw_ellipse(ax[1], venn_git, 'GIT')
    draw_ellipse(ax[2], venn_ofa, 'OFA')
    draw_ellipse(ax[3], venn_oscar, 'OSCAR')

    oamt_patch = patches.Patch(color='lightgreen', alpha=0.5, label='OAMT Errors')
    rome_patch = patches.Patch(color='lightblue', alpha=0.5, label='ROME Errors')

    legend = fig.legend(handles=[oamt_patch, rome_patch],
                        loc='center',
                        ncol=2,
                        fontsize=20,
                        frameon=False,
                        bbox_to_anchor=(0.5, 0.01),
                        prop={'family': 'Times New Roman', 'weight': 'bold','size':20})

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()



model_name =['blip','git','ofa','oscar']
all_sets=[]
for name in model_name:
    with open(f'TestIC/ROME/suspicious_list/{name}_base/report_issues', 'r') as list_file :
        with open(f'TestIC/ROME/error_detection/{name}_base_ancestor.tsv', 'r') as tsv_file :
            ROME_img_ids = get_image_ids_from_list_file(list_file,tsv_file)

    with open(f'../rq_results/issues/{name}_issues','r') as issue_file:
        OAMT_img_ids = extract_img_ids_from_OAMT_issue_file(issue_file)
    all_sets.append((set(OAMT_img_ids),set(ROME_img_ids)))

plot_model_venns(
    blip_sets=all_sets[0],
    git_sets=all_sets[1],
    ofa_sets=all_sets[2],
    oscar_sets=all_sets[3]
)


