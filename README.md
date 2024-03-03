# miMic (Mann-Whitney image microbiome) 

This repository is attached to the paper "miMic - a novel multi-layer statistical test for microbiome disease".    
miMic is a straightforward yet remarkably versatile and scalable approach for differential abundance analysis.

miMic consists of three main steps:

- Data preprocessing and translation to a cladogram of means.

-  An apriori nested ANOVA (or nested GLM for continuous labels) to detect overall microbiome-label relations.

-  A post hoc test along the cladogram trajectories.


##  miMic

miMic is available through the following platforms:
- [GitHub](https://github.com/oshritshtossel/miMic) 
- [PyPi](https://pypi.org/project/mimic-da/)
- [Website](https://micros.math.biu.ac.il/)

### Install the package
```python
pip install mimic-da
```
##  How to apply miMic 
See `example_use.py` for an example of how to use miMic.  
The example contains the following steps:

1.  Import miMic and additional packages.
    ```python
    from mimic_da import apply_mimic
    import pandas as pd
    ```

2. Load the raw ASVs table in the following format:    
   - The first column is named "ID"
   - Each row represents a sample and each column represents an ASV.  
   - The last row contains the taxonomy information, named "taxonomy".

    ```python
    df = pd.read_csv("example_data/for_process.csv")
    ```
   - <u>Note:</u> `for_process.csv` is a file that contains the raw ASVs table in the required format, you can find an exmaple file in `example_data` folder.   


3. Load a tag table as csv, such that the tag column is named "Tag".

      ```python
    tag = pd.read_csv("example_data/tag.csv",index_col=0)
      ```
   - <u>Note:</u>  `tag.csv` is a file that contains the tag table in the required format, you can find an example tag in `example_data` folder.


4. Specify a folder to save the output of the miMic test.   
   ```python
   folder = "example_data/2D_images"
   ```
   - <u>Note:</u>  `2D_images` is a folder that will be created in your current working directory, and the output of the miMic test will be saved there.


5. Apply MIPMLP.
   - MIPMLP using defaulting parameters, you can find more in 'Note' section below.
   - taxonomy_group: ["sub PCA", "mean", "sum"], "sub PCA" method is preferred.

   ```python
   processed = apply_mimic(folder=folder, tag=tag, mode="preprocess", preprocess=True, rawData=df,
                            taxnomy_group='sub PCA')
   ```
   - <u>Note:</u>  MIPMLP is a package that is used to preprocess the raw ASVs table, see [MIPMLP PyPi](https://pypi.org/project/MIPMLP/) or [MIPMLP website](https://mip-mlp.math.biu.ac.il/Home) for more explanations.   
<u>If you have your own processed data</u>, set `preprocess` to False, and use your processed data as input for `proceesed` parameter in the next step.


6. Apply miMic test.   
   miMic using the following hyperparameters:   
    - **eval**: evaluation method, ["man", "corr", "cat"]. Default is <u>"man"</u>.
      - "man" for binary labels.
      - "corr" for continuous labels.
      - "cat" for categorical labels.
    - **sis**: apply sister correction,["fdr_bh", "bonferroni", "no"]. Default is <u> "fdr_bh"</u>.
    - **correct_first**: apply FDR correction to the starting taxonomy level according to `sis` parameter,[True, False] Default is <u>True</u>.
    - **mode**: 2 different formats of running,["test", "plot"]. Default is <u>"test"</u>.
    - **save**: whether to save the corrs_df of the miMic test to computer,[True, False] Default is <u>True</u>.
    - **tax**: starting taxonomy of the post hoc test,["None", 1, 2, 3, "noAnova", "nosignifacnt"]   
      - In <u>"test"</u> mode the defaulting value is <u>"None"</u>. 
      - In the <u>"plot"</u> mode the tax is <u>set automatically</u> to the selected taxonomy of the "test" mode [1, 2, 3, "noAnova"].
      - "noAnova", where apriori nested ANOVA test is not significant.
      - "nosignificant", where apriori nested ANOVA test is not significant and miMic did not find any significant taxa in the leafs. In this case, the post hoc test will **not** be applied.
    - **colorful**: Determines whether to apply colorful mode on the plots [True, False]. <u>Default</u> is True.
    - **threshold_p**: the threshold for significant values. Default is <u>0.05</u>.
    - **THRESHOLD_edge**: the threshold for having an edge in "interaction" plot. Default is <u>0.5</u>.
    - **processed**: the processed data from the previous step. Default is <u>None</u>.
    - **apply_samba**: whether to apply samba or no. Default is True (Boolean).
    - **samba_output**: if you already have samba outputs- miMic will read it from the folder you specified,
    else miMic will apply samba and set `samba_output` to None.
   
     ```python
       if processed is not None:
            taxonomy_selected,samba_output = apply_mimic(folder, tag, eval="man", threshold_p=0.05, processed=processed, apply_samba=True, save=False)
            if taxonomy_selected is not None:
                apply_mimic(folder, tag, mode="plot", tax=taxonomy_selected, eval="man", sis='fdr_bh', samba_output=samba_output,save=False,
                            threshold_p=0.05, THRESHOLD_edge=0.5)
   ```
   - <u>Note:</u>  if `apply_samba` is set to True, miMic will apply samba-metric.   
   If `save` is set to True, the output will be saved to the folder you specified.   
   See [SAMBA PyPi](https://pypi.org/project/samba-metric/) for more explanations. 
##  miMic output
miMic will output the following:

- If `save` is set to True, samba outputs and the following csv will be saved to your specified folder:
  - **corrs_df**: a dataframe containing the results of the miMic test (including Utest results).
  - **just_mimic**: a dataframe containing the results of the miMic test without the Utest results.
  - **u_test_without_mimic**: a dataframe containing the results of the Utest without the miMic results.
  - **miMic&Utest**: a dataframe containing the joint results of miMic and Utest tests.


- If `mode` is set to "plot", plots will be saved in the folder named <u>'plots'</u> in your current working directory.    
The following plots will be saved:
   1.  **tax_vs_rp_sp_anova_p**: plot RP vs SP over the different taxonomy levels and color the background of the plot till the selected taxonomy, based on miMic test.  
  ![tax_vs_rp_sp_anova_p](https://github.com/oshritshtossel/miMic/raw/master/plots/tax_vs_rp_sp_anova_p.png)

   2. **rsp_vs_beta**: calculate RSP score for different betas and create the appropriate plot.   
  ![rsp_vs_beta](https://github.com/oshritshtossel/miMic/raw/master/plots/rsp_vs_beta.png)

   3. **hist**: a histogram of the ASVs in each taxonomy level.   
  ![hist](https://github.com/oshritshtossel/miMic/raw/master/plots/hist.png)

   4. **corrs_within_family**: a plot of the correlation between the significant ASVs within the family level, if `colorful` is set to True, each family will be colored.    
  ![corrs_within_family](https://github.com/oshritshtossel/miMic/raw/master/plots/corrs_within_family.png)

   5. **interaction**: a plot of the interaction between the significant ASVs.  
  ![interaction](https://github.com/oshritshtossel/miMic/raw/master/plots/interaction.png)

   6. **correlations_tree**: create correlation cladogram, such that tha size of each node is according to the -log(p-value), the color of 
       each node represents the sign of the post hoc test, the shape of the node (circle, square,sphere) is based on 
       miMic, Utest, or both results accordingly, and if `colorful` is set to True, the background color of the node will be colored based on the family color.  
   ![correlations_tree](https://github.com/oshritshtossel/miMic/raw/master/plots/correlations_tree.svg)





 
   
   
