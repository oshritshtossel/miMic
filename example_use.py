import pandas as pd

try:
    from mimic import apply_mimic
except:
    from src.mimic_da import apply_mimic

if __name__ == '__main__':

    # Load the raw data and tag in the csv format => apply 'preprocess' mode => apply mimic_da with 'test' mode => apply mimic_da with 'plot' mode
    df = pd.read_csv("example_data/for_process.csv")
    tag = pd.read_csv("example_data/tag.csv", index_col=0)
    folder = "example_data/2D_images"

    # Apply MIPMLP and samba automatically
    processed = apply_mimic(folder=folder, tag=tag, mode="preprocess", preprocess=True, rawData=df,
                            taxnomy_group='sub PCA')
    # processed= Apply with your own processed data

    # Apply miMic test
    if processed is not None:
        taxonomy_selected = apply_mimic(folder, tag, eval="man", threshold_p=0.05, save=True, processed=processed)
        if taxonomy_selected is not None:
            apply_mimic(folder, tag, mode="plot", tax=taxonomy_selected, eval="man", sis='fdr_bh', save=False,
                        threshold_p=0.05, THRESHOLD_edge=0.5)
