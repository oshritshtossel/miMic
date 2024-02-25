import pandas as pd
import MIPMLP
import samba

try:
    from mimic import apply_mimic
except:
    from src.mimic import apply_mimic

if __name__ == '__main__':
    # Load the raw data and tag in the csv format => apply 'preprocess' mode => apply mimic with 'test' mode => apply mimic with 'plot' mode
    df = pd.read_csv("example_data/for_process.csv")
    tag = pd.read_csv("example_data/tag.csv", index_col=0)
    folder = "example_data/2D_images"

    # Apply MIPMLP and samba automatically
    apply_mimic(folder=folder, tag=tag, mode='preprocess', rowData=df, taxnomy_group='sub PCA')

    # If you want to apply the MIPMLP and Samba by yourself instead of apply "preprocess" mode as above, you can use the following code:
    # processed = MIPMLP.preprocess(df, taxnomy_group="sub PCA")
    # # micro2matrix and saving the images in a prepared folder
    # samba.micro2matrix(processed, folder, save=True)

    # Apply miMic test
    taxonomy_selected = apply_mimic(folder, tag, eval="corr", threshold_p=0.05, save=True)
    if taxonomy_selected is not None:
        apply_mimic(folder, tag, mode="plot", tax=taxonomy_selected, eval="corr", sis='fdr_bh', save=False,
                    threshold_p=0.05, THRESHOLD_edge=0.5)
