from Config import Config
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages # Import necessario
main_root = Path(Config.Paths.mainRooot)
root = (
    main_root / 
    Config.Paths.DataRoots.ResourcesRoot / 
    Config.Paths.DataRoots.StreamRoot / 
    Config.Paths.DataRoots.CaseStudyRoot() /
    Config.Packages.Drivers.__name__ / 
    Config.Packages.Drivers.Phases.Phase4.__name__ / 
    Config.Packages.Drivers.Phases.Phase4.Modules.Module2.__name__ / 
    Config.Packages.Drivers.Phases.Phase4.Modules.Module2.Tasks.Task1.__name__ / 
    Config.Packages.Drivers.Phases.Phase4.Modules.Module2.Tasks.Task1.MetaData.OutputExt)
data = pd.read_pickle(root)
df_plot = pd.concat(data.values(), keys=data.keys(), names=['Dataset', 'key']).reset_index()
sns.set_theme(style="darkgrid")
plt.rcParams.update({'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 16})
num_blades = df_plot['Blade'].nunique()
custom_palette = sns.color_palette(["#40E0D0", "#20B2AA", "#008B8B", "#006080", "#000080"])[:num_blades]

# Salvataggio in PDF
with PdfPages('damages.pdf') as pdf:
    plt.figure(figsize=(14, 8))
    plot = sns.lineplot(
        data=df_plot, 
        x='Dataset', 
        y='Area', 
        hue='Blade', 
        style='Blade',
        markers=True,
        dashes=False,
        palette=custom_palette,
        linewidth=2.5,
        errorbar=('pi', 25),
        err_kws={'alpha': 0.2}
    )
    plt.title('Damage Area Evolution per Blade', fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Dataset', fontsize=16)
    plt.ylabel('Area ($mm^2$)', fontsize=16)
    plt.xticks(rotation=45)
    plt.legend(
        title='Blade', 
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.2), 
        ncol=num_blades, 
        frameon=True,
        fontsize=12
    )
    plt.tight_layout()
    pdf.savefig()
    plt.close() 