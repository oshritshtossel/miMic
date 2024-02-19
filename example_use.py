import pandas as pd
import MIPMLP
import samba

# try:
#     from mimic import apply_mimic
# except:
from src.mimic import apply_mimic


if __name__ == '__main__':
    # Load the raw data in the required format
    # df = pd.read_csv("between/ibd_for_process.csv")
    tag = pd.read_csv("/home/shanif3/Codes/MIPMLP/data_to_compare/PRJEB14674/tag.csv", index_col=0)
    # processed = pd.read_csv("/home/shanif3/mimic_git/check/aftermipmlp.csv", index_col=0)

    # Apply the MIPMLP with the defaultive parameters
    # processed = MIPMLP.preprocess(df, taxnomy_group="sub PCA")

    # micro2matrix and saving the images in a prepared folder
    folder = "between/PRJEB14674"
    # samba.micro2matrix(processed, folder, save=True)

    # Apply miMic test
    taxonomy_selected = apply_mimic(folder, tag, eval="man", threshold_p=0.05, save=True)
    if taxonomy_selected is not None:
        apply_mimic(folder, tag, mode="plot", tax=taxonomy_selected, eval="man", sis='fdr_bh', save=False,
                    threshold_p=0.05, THRESHOLD_edge=0.5)
