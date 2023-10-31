import pandas as pd
import MIPMLP
import samba
from src import apply_mimic

if __name__ == '__main__':

    # Load the raw data in the required format
    df = pd.read_csv("between/ibd_for_process.csv")
    tag = pd.read_csv("between/ibd_tag.csv",index_col=0)


    # Apply the MIPMLP with the defaultive parameters
    processed = MIPMLP.preprocess(df)

    # micro2matrix and saving the images in a prepared folder
    folder = "between/2D_ibd"
    samba.micro2matrix(processed, folder, save=True)

    # Apply miMic test
    taxonomy_selected = apply_mimic(folder, tag,eval="man")
    if not taxonomy_selected :
        apply_mimic(folder, tag,mode="plot",tax=taxonomy_selected,eval="man")



    c = 0
