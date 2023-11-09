import pandas as pd
import MIPMLP
import samba

try:
    from miMic_test import apply_mimic
except:
    from src import apply_mimic

if __name__ == '__main__':

    # Load the raw data in the required format
    df = pd.read_csv("between/ibd_for_process.csv")
    tag = pd.read_csv("between/ibd_tag.csv", index_col=0)

    # Apply the MIPMLP with the defaultive parameters
    processed = MIPMLP.preprocess(df, taxnomy_group="sub PCA")

    # micro2matrix and saving the images in a prepared folder
    folder = "between/2D_ibd"
    samba.micro2matrix(processed, folder, save=True)

    # Apply miMic test
    taxonomy_selected = apply_mimic(folder, tag, eval="man")
    if taxonomy_selected is not None:
        apply_mimic(folder, tag, mode="plot", tax=taxonomy_selected, eval="man", save=False)

    c = 0
