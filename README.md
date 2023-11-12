# miMic (Mann-Whitney image microbiome)

This code is attached to the paper "miMic - a novel multi-layer statistical test for microbiome disease". 
miMic is a straightforward yet remarkably versatile and scalable approach for differential abundance analysis.
miMic consists of three main steps: 

(1)  Data preprocessing and translation to a cladogram of means.

(2)  An apriori nested ANOVA (or nested GLM for continuous labels to detect overall microbiome-label relations.

(3)  A post hoc test along the cladogram trajectories.


## How to apply miMic

miMic is available at this [GitHub](https://github.com/oshritshtossel/miMic), [PyPi](https://pypi.org/project/mimic-da/), and in the following [website](https://www.google.com).

### miMic's GitHub
There is an example in example_use.py. You should follow the following steps:

1. Load the raw ASVs table in the following format: the first column is named "ID", each row represents a sample and each column represents an ASV. The last row contains the taxonomy information, named "taxonomy".

```python
df = pd.read_csv("example_data/for_process.csv")
```

2. Load a tag table as CSV, such that the tag column is named "Tag".

  ```python
tag = pd.read_csv("example_data/tag.csv",index_col=0)
```

3. Apply the MIPMLP with the defaulting parameters, except for the taxnomy_group that the "sub PCA" method is preferred
 (see [MIPMLP PyPi](https://pypi.org/project/MIPMLP/) or [MIPMLP website](https://mip-mlp.math.biu.ac.il/Home) for more explanations).

```python
   processed = MIPMLP.preprocess(df,taxnomy_group="sub PCA")
```

4. micro2matrix (translate microbiome into matrix according to [iMic](https://www.tandfonline.com/doi/full/10.1080/19490976.2023.2224474), and save the images in a prepared folder.

  ```python
   folder = "example_data/2D_images"
    samba.micro2matrix(processed, folder, save=True)
   ```
  Note for more information on [SAMBA](https://github.com/oshritshtossel/SAMBA) and for further distance calculations.

5. Apply the miMic test.
   One can choose the following hyperparameters:

   - **eval** (evaluation method) Choose one of **"mann"** for binary labels, **"corr"** for continuous labels, and **"cat"** for categorical labels.
   - **sis** (apply sister correction) Choose one of **"bonferroni"** (defaulting value) or **"no"**.
   - **correct_first** (apply FDR correction to the starting taxonomy level) Choose one of **True** (defaulting value) or **False**.
   - **mode** (2 different formats of running) Choose one of **"test"** (defaulting value)  or **"plot"**. The "plot" mode should be applied only if the "test" mode is significant.
   - **save** (whether to save the corrs_df od the miMic test to computer) Choose one of **True** (defaulting value)  or **False**.
   - **tax** (Starting taxonomy of the post hoc test) Choose one of **None** ((defaulting value for "test" mode) or one of **1**, **2**, **3**, **4**, **5**, **6**, **7**. In the "plot" mode the tax is set automatically to the selected taxonomy of the "test" mode.

     ```python
      taxonomy_selected = apply_mimic(folder, tag, eval="man")
      if not taxonomy_selected:
        apply_mimic(folder, tag, mode="plot", tax=taxonomy_selected, eval="man")
   ```

### miMic's PyPi

1. Install the package

```python
pip install mimic-da
```

2. Load the raw ASVs table in the following format: the first column is named "ID", each row represents a sample and each column represents an ASV. The last row contains the taxonomy information, named "taxonomy".

```python
df = pd.read_csv("example_data/for_process.csv")
```

3. Load a tag table as CSV, such that the tag column is named "Tag".

  ```python
tag = pd.read_csv("example_data/tag.csv",index_col=0)
```

4.  Apply the MIPMLP with the defaulting parameters, except for the taxnomy_group that the "sub PCA" method is preferred
 (see [MIPMLP PyPi](https://pypi.org/project/MIPMLP/) or [MIPMLP website](https://mip-mlp.math.biu.ac.il/Home) for more explanations).

```python
   processed = MIPMLP.preprocess(df,taxnomy_group="sub PCA")
```

5.  micro2matrix (translate microbiome into matrix according to [iMic](https://www.tandfonline.com/doi/full/10.1080/19490976.2023.2224474), and save the images in a prepared folder.

  ```python
   folder = "example_data/2D_images"
    samba.micro2matrix(processed, folder, save=True)
   ```
  Note for more information on [SAMBA](https://github.com/oshritshtossel/SAMBA) and for further distance calculations.

6. Apply the miMic test.
   One can choose the following hyperparameters:

   - **eval** (evaluation method) Choose one of **"mann"** for binary labels, **"corr"** for continuous labels, and **"cat"** for categorical labels.
   - **sis** (apply sister correction) Choose one of **"bonferroni"** (defaulting value) or **"no"**.
   - **correct_first** (apply FDR correction to the starting taxonomy level) Choose one of **True** (defaulting value) or **False**.
   - **mode** (2 different formats of running) Choose one of **"test"** (defaulting value)  or **"plot"**. The "plot" mode should be applied only if the "test" mode is significant.
   - **save** (whether to save the corrs_df od the miMic test to computer) Choose one of **True** (defaulting value)  or **False**.
   - **tax** (Starting taxonomy of the post hoc test) Choose one of **None** ((defaulting value for "test" mode) or one of **1**, **2**, **3**, **4**, **5**, **6**, **7**. In the "plot" mode the tax is set automatically to the selected taxonomy of the "test" mode.

     ```python
     from miMic_test import apply_mimic
      taxonomy_selected = apply_mimic(folder, tag, eval="man")
      if not taxonomy_selected:
        apply_mimic(folder, tag, mode="plot", tax=taxonomy_selected, eval="man")
   ```

  ### Code example for GitHub or PyPi
  ```python
import pandas as pd
import MIPMLP
import samba

try:
    from mimic import apply_mimic
except:
    from src.mimic import apply_mimic

if __name__ == '__main__':

    # Load the raw data in the required format
    df = pd.read_csv("between/ibd_for_process.csv")
    tag = pd.read_csv("between/ibd_tag.csv", index_col=0)

    # Apply the MIPMLP with the defaultive parameters
    processed = MIPMLP.preprocess(df,taxnomy_group="sub PCA")

    # micro2matrix and saving the images in a prepared folder
    folder = "between/2D_ibd"
    samba.micro2matrix(processed, folder, save=True)

    # Apply miMic test
    taxonomy_selected = apply_mimic(folder, tag, eval="man")
    if taxonomy_selected is not None:
        apply_mimic(folder, tag, mode="plot", tax=taxonomy_selected, eval="man")
   ```

   
   
