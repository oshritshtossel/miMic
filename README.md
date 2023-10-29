# miMic (Mann-Whitney image microbiome)

This code is attached to the paper "miMic - a novel multi-layer statistical test for microbiome disease". 
miMic is a straightforward yet remarkably versatile and scalable approach for differential abundance analysis.
miMic consists of three main steps: 

(1)  Data preprocessing and translation to a cladogram of means.

(2)  An apriori nested ANOVA (or nested GLM for continuous labels to detect overall microbiome-label relations.

(3)  A post hoc test along the cladogram trajectories.


## How to apply miMic

miMic is available at this [GitHub](https://github.com/oshritshtossel/miMic), [PyPi](https://www.google.com), and in the following [website](https://www.google.com).

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

3. Apply the MIPMLP with the defaulting parameters (see [MIPMLP PyPi](https://pypi.org/project/MIPMLP/) or [MIPMLP website](https://mip-mlp.math.biu.ac.il/Home) for more explanations).

   ```python
   processed = MIPMLP.preprocess(df)
   ```

4. micro2matrix (translate microbiome into matrix according to [iMic](https://www.tandfonline.com/doi/full/10.1080/19490976.2023.2224474), and save the images in a prepared folder.

  ```python
   folder = "example_data/2D_images"
    samba.micro2matrix(processed, folder, save=True)
   ```
  Note for more information on [SAMBA](https://github.com/oshritshtossel/SAMBA) and for further distance calculations.

5. Apply miMic test.
   One can choose the following hyperparmaters:

   - **eval** (evaluation method) "mann" for binary labels, "corr" for continuous labels, and "cat" for categorical labels.
   
