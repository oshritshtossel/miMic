import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from scipy.stats import spearmanr
from statsmodels.formula.api import ols
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal
import networkx as nx
import ete3
from src import create_tax_tree
from ete3 import NodeStyle, TextFace, add_face_to_node, TreeStyle
from copy import deepcopy
import re
import statsmodels.stats.multitest as smt


def load_img(folder_path, tag):
    """
    Load images from the folder they are saved there, and create an list of loaded images and list of the names
    :param folder_path: Folder where the images are saved (str)
    :param tag: Tag dataframe with a column named "Tag" (dataframe)
    :return: final_array (list of loaded images), names (list of sample names)
    """
    arrays = []
    names = []
    for file in os.listdir(folder_path):
        if file.endswith(".npy"):
            if file == "bact_names.npy":
                continue
            file_path = os.path.join(folder_path, file)
            if file_path.split("\\")[-1].replace(".npy", "") in [str(i) for i in tag.index]:
                arrays.append(np.load(file_path, allow_pickle=True, mmap_mode='r'))
                names.append(file_path.split("\\")[-1].replace(".npy", ""))

    final_array = np.stack(arrays, axis=0)
    return final_array, names


def load_from_folder(folder, tag):
    """
    Load all images
    :param folder: Folder where the images are saved (str)
    :param tag: Tag dataframe with a column named "Tag" (dataframe)
    :return: img_arrays-ndarray of images (ndarray), bact_names-ndarray of taxa names (ndarray), tag (dataframe)
    """
    img_arrays, names = load_img(folder, tag)
    tag.index = [str(i) for i in tag.index]
    tag = tag.loc[names]
    index_name = tag.index.name
    if index_name == None:
        index_name = "index"
    tag = tag.reset_index()
    del tag[index_name]
    bact_names = np.load(f'{folder}/bact_names.npy', allow_pickle=True)

    return img_arrays, bact_names, tag


def build_img_from_table(table, col1, col2, names):
    """
    Build images of test's scores (img_s) and images of test's p-values (img_p).
    :param table: Post hoc test results (dataframe)
    :param col1: Name of the p-value column (str)
    :param col2: Name of the score column (str)
    :param names: List of sample names (list)
    :return: Images of test's scores (img_s) and images of test's p-values (img_p)
    """
    cols = len(names.columns)
    rows = len(names.index)
    img_p = np.zeros((rows, cols))
    img_s = np.zeros((rows, cols))
    unique_indexes = list(table.index)
    for target_str in unique_indexes:
        p_val = table.loc[target_str][col1]
        s_val = table.loc[target_str][col2]
        # Find row and column indices where entries in 'names' are equal to 'target_str'
        row_indices, col_indices = np.where(names == target_str)
        row_indices = list(set(row_indices))
        col_indices = list(set(col_indices))
        for r in row_indices:
            for c in col_indices:
                img_p[r, c] = p_val
                img_s[r, c] = s_val

    return img_p, img_s


def create_list_of_names(list_leaves):
    """
    Fix taxa names for tree plot.
    :param list_leaves: List of leaves names without the initials (list).
    :return: Corrected list taxa names.
    """
    list_lens = [len(i.split(";")) for i in list_leaves]
    otu_train_cols = list()
    for i, j in zip(list_leaves, list_lens):
        if j == 1:
            updated = "k__" + i.split(";")[0]
        elif j == 2:
            updated = "k__" + i.split(";")[0] + ";" + "p__" + i.split(";")[1]
        elif j == 3:
            updated = "k__" + i.split(";")[0] + ";" + "p__" + i.split(";")[1] + ";" + "c__" + i.split(";")[2]
        elif j == 4:
            updated = "k__" + i.split(";")[0] + ";" + "p__" + i.split(";")[1] + ";" + "c__" + i.split(";")[
                2] + ";" + "o__" + i.split(";")[3]
        elif j == 5:
            updated = "k__" + i.split(";")[0] + ";" + "p__" + i.split(";")[1] + ";" + "c__" + i.split(";")[
                2] + ";" + "o__" + i.split(";")[3] + ";" + "f__" + i.split(";")[4]
        elif j == 6:
            updated = "k__" + i.split(";")[0] + ";" + "p__" + i.split(";")[1] + ";" + "c__" + i.split(";")[
                2] + ";" + "o__" + i.split(";")[3] + ";" + "f__" + i.split(";")[4] + ";" + "g__" + i.split(";")[5]
        elif j == 7:
            updated = "k__" + i.split(";")[0] + ";" + "p__" + i.split(";")[1] + ";" + "c__" + i.split(";")[
                2] + ";" + "o__" + i.split(";")[3] + ";" + "f__" + i.split(";")[4] + ";" + "g__" + i.split(";")[
                          5] + ";" + "s__" + \
                      i.split(";")[6]
        otu_train_cols.append(updated)
    return otu_train_cols


def creare_tree_view(names, mean_0, mean_1, directory):
    """
    Create correlation cladogram, such that tha size of each node is according to the -log(p-value),
    the color of each node represents the sign of the post hoc test.
    :param names:  List of sample names (list)
    :param mean_0: 2D ndarray of the images filled with the post hoc p-values (ndarray).
    :param mean_1:  2D ndarray of the images filled with the post hoc scores (ndarray).
    :param directory: Folder to save the correlation cladogram (str)
    :return: None
    """
    T = ete3.PhyloTree()
    # Get the number of rows and columns in the ndarray
    num_rows, num_cols = names.shape

    # Create an empty list to store the first non-empty cell for each column
    first_non_empty_cells = [None] * num_cols

    # Iterate through the ndarray starting from the lowest row (row 7)
    for row_idx in reversed(range(num_rows)):
        for col_idx in range(num_cols):
            # Check if the current cell is empty (None or nan)
            if names[row_idx, col_idx] == '':
                continue
            # If the current cell is not empty, update the first non-empty cell for this column
            if first_non_empty_cells[col_idx] is None:
                first_non_empty_cells[col_idx] = names[row_idx, col_idx]

    otu_train_cols = create_list_of_names(first_non_empty_cells)
    g = create_tax_tree(pd.Series(index=otu_train_cols))
    epsilon = 1e-1000
    root = list(filter(lambda p: p[1] == 0, g.in_degree))[0][0]
    T.get_tree_root().species = root[0]
    T.get_tree_root().add_feature("max_0_grad", -np.log10(mean_0[0].mean() + epsilon))
    T.get_tree_root().add_feature("max_1_grad", mean_1[0].mean())
    T.get_tree_root().add_feature("max_2_grad", mean_0[0].mean())
    for node in g.nodes:
        for s in g.succ[node]:
            if s[0][-1] not in T or not any([anc.species == a for anc, a in
                                             zip(T.search_nodes(name=s[0][-1])[0].get_ancestors()[:-1],
                                                 reversed(s[0]))]):
                t = T
                if len(s[0]) != 1:
                    print(s[0])
                    t = T.search_nodes(full_name=s[0][:-1])[0]
                t = t.add_child(name=s[0][-1])
                t.species = s[0][-1]
                t.add_feature("full_name", s[0])
                t.add_feature("max_0_grad", -np.log10(mean_0[names == ";".join(s[0])].mean() + epsilon))
                t.add_feature("max_1_grad", mean_1[names == ";".join(s[0])].mean())
                t.add_feature("max_2_grad", mean_0[names == ";".join(s[0])].mean())

    for name, val in pd.Series(data=mean_0[-1], index=otu_train_cols).items():
        name = name.replace(" ", "")
        name = re.split("; |__|;", name)
        name = [i for i in name if len(i) > 2 or (len(i) > 0 and i[-1].isnumeric())]

        t = T.search_nodes(full_name=tuple(name))[0]
        t.dist = val
    T0 = T.copy("deepcopy")

    bound_0 = 0
    for t in T0.get_descendants():
        nstyle = NodeStyle()
        nstyle["size"] = 20
        nstyle["fgcolor"] = "gray"
        if t.max_1_grad > bound_0 and t.max_2_grad < 0.05:
            print("; ".join(reversed([t.name] + [i.name for i in t.get_ancestors() if len(i.name) > 0])))
            nstyle["fgcolor"] = "blue"
            nstyle["size"] = t.max_0_grad * 5
        elif t.max_1_grad < bound_0 and t.max_2_grad < 0.05:
            print("; ".join(reversed([t.name] + [i.name for i in t.get_ancestors() if len(i.name) > 0])))
            nstyle["fgcolor"] = "red"
            nstyle["size"] = t.max_0_grad * 5


        elif not t.is_root():
            if not any([anc.max_0_grad > bound_0 for anc in t.get_ancestors()[:-1]]) and not any(
                    [dec.max_0_grad > bound_0 for dec in t.get_descendants()]):
                t.detach()
        t.set_style(nstyle)

    for node in T0.get_descendants():
        if node.is_leaf():
            name = node.name.replace('_', '').capitalize()
            name = "".join([i for i in name if not i.isdigit()])
            if name == "":
                name = node.get_ancestors()[0].name.replace("_", "").capitalize()
                name = "".join([i for i in name if not i.isdigit()])
            node.name = name

    for node in T0.get_descendants():
        for sis in node.get_sisters():
            siss = []
            if sis.name == node.name:
                node.max_0_grad += sis.max_0_grad
                node.max_1_grad += sis.max_1_grad
                siss.append(sis)
            if len(siss) > 0:
                node.max_0_grad /= (len(sis) + 1)
                node.max_1_grad /= (len(sis) + 1)
                for s in siss:
                    node.remove_sister(s)

    ts = TreeStyle()
    ts.show_leaf_name = False
    ts.min_leaf_separation = 0.5
    ts.mode = "c"
    ts.root_opening_factor = 0.75
    ts.show_branch_length = False

    D = {1: "(k)", 2: "(p)", 3: "(c)", 4: "(o)", 5: "(f)", 6: "(g)", 7: "(s)"}

    def my_layout(node):
        """
        Design the cladogram layout.
        :param node: Node ETE object
        :return: None
        """
        if node.is_leaf():
            tax = D[len(node.full_name)]
            if len(node.full_name) == 7:
                name = node.up.name.replace("[", "").replace("]", "") + " " + node.name.lower()
            else:
                name = node.name
            F = TextFace(f"{name} {tax} ", fsize=60, ftype="Arial")  # {tax}
            add_face_to_node(F, node, column=0, position="branch-right")

    ts.layout_fn = my_layout
    T0.show(tree_style=deepcopy(ts))
    T0.render(f"{directory}/correlations_tree.png", tree_style=deepcopy(ts))


def build_interactions(all_ps, bact_names, img_array, save, THRESHOLD=0.5):
    """
    Plot interaction network between the significant taxa founded by miMic, such that each node color
    is according to the sigh of the post hoc test with the tag, its shape is according to its order, and
    its edge width is according to the correlation between the pair. There are edges only between pirs with
    correlation above the threshold.
    :param all_ps: All post hoc p-values (dataframe).
    :param bact_names: Dataframe with bact names according to the image order (dataframe)
    :param img_array: List of loaded iMic images (list).
    :param save: Name of folder to save the plot (str).
    :param THRESHOLD: The threshold for having an edge (float).
    :return: None. It creates a plot.
    """
    only_significant = all_ps[all_ps[0] < 0.05]

    # Initialize a DataFrame to store the first index
    df_index = pd.DataFrame(index=only_significant.index, columns=["index"])

    for pixel in only_significant.index:
        row = all_ps.loc[pixel]["len"]
        indexes = min([i for i, value in enumerate(bact_names.loc[row]) if value == pixel])
        df_index["index"][pixel] = indexes
    only_significant["index"] = df_index["index"]
    # only species
    only_significant = only_significant[only_significant["len"] == 7]
    only_significant.index = [
        node.replace("[", "").replace("]", "").replace(";_0", "").replace(";_1", "").replace(";_2", "").replace(";_3",
                                                                                                                "").replace(
            ";_4", "").replace(";_5", "").replace(";_6", "").replace(";_7", "").replace("_0", "") for node in
        only_significant.index]
    only_significant["fixed_len"] = [len(i.split(";")) for i in only_significant.index]
    only_significant = only_significant[only_significant["fixed_len"] == 7]
    only_significant["size"] = -np.log10(only_significant[0]) * 30
    only_significant["color"] = ["red" for i in only_significant.index]
    only_significant["color"][only_significant["s"] > 0] = "blue"
    only_significant["order"] = [i.split(";")[3] for i in only_significant.index]

    # Create a mapping of unique orders to node shapes
    unique_orders = only_significant['order'].unique()
    shapes = ['o', 's', 'd', '^', 'v', '>', '<', 'p', 'H', '8', 'o']  # Shapes for unique orders

    shape_mapping = dict(zip(unique_orders, shapes))

    # Create a new "shape" column based on the mapping
    only_significant['shape'] = only_significant['order'].map(shape_mapping)
    only_significant["label"] = [n.split(";")[-2:][0] + ";" + n.split(";")[-2:][-1] for n in
                                 list(only_significant.index)]

    # Get values from img arrays
    bact_for_corr = pd.DataFrame(columns=only_significant.index)
    for pixel in only_significant.index:
        ROW = only_significant.loc[pixel]["len"]
        INDEX = only_significant.loc[pixel]["index"]
        bact_for_corr[pixel] = img_array[:, ROW, INDEX]

    # Build correlation matrix
    inter_corr = bact_for_corr.corr(method="spearman")
    # Zero all the corrs under a certain threshold
    inter_corr[abs(inter_corr) < THRESHOLD] = 0.0

    # Visualize the interaction network
    # Create a graph
    G = nx.Graph()

    # Add nodes to the graph
    num_nodes = len(inter_corr)
    nodes = list(range(num_nodes))
    G.add_nodes_from(nodes)

    # Add weighted edges based on correlations
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            r = list(inter_corr.index)[i]
            c = list(inter_corr.columns)[j]
            weight = inter_corr[c][r]
            if abs(weight) > 0.5:  # Adjust the threshold as needed
                G.add_edge(i, j, weight=weight)

    # Create a dictionary to map numerical node names to original names
    numerical_to_original_names = {i: name for i, name in enumerate(
        [n.split(";")[-2:][0] + ";" + n.split(";")[-2:][-1] for n in list(inter_corr.index)])}

    # Assign original names to the nodes
    for i, node in G.nodes(data=True):
        node['name'] = numerical_to_original_names[i]

    node_colors = {i: color for i, color in enumerate(only_significant['color'])}
    node_sizes = {i: size for i, size in enumerate(only_significant['size'])}
    node_shapes = {i: shape for i, shape in enumerate(only_significant['shape'])}

    # Create a layout for the graph (e.g., spring_layout)
    pos = nx.circular_layout(G)

    # Define edge colors and widths based on weight sign and absolute value
    edge_colors = ['blue' if weight >= 0 else 'red' for i, j, weight in G.edges(data='weight')]
    edge_widths = [-np.log10(abs(data['weight'])) * 9 for _, _, data in
                   G.edges(data=True)]  # Adjust the scaling factor as needed

    # Draw nodes
    fig, ax = plt.subplots(figsize=(10, 10))
    # Draw nodes with colors from the "color" column
    for i in G.nodes():
        nx.draw_networkx_nodes(G, pos, nodelist=[i], node_size=node_sizes[i], node_color=node_colors[i],
                               node_shape=node_shapes[i])

    # Draw edges with colors and widths
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.7)

    # Add labels to nodes
    # Calculate positions for node labels on the periphery
    node_labels = {i: node['name'] for i, node in G.nodes(data=True)}
    # Calculate positions for node labels (vertical orientation)
    label_positions = {node: (pos[node][0], pos[node][1] + 0.01) for node in G.nodes()}
    description = nx.draw_networkx_labels(G, label_positions, labels=node_labels, font_size=10, font_color='black')

    r = fig.canvas.get_renderer()
    trans = plt.gca().transData.inverted()
    for i, (node, t) in enumerate(description.items()):
        theta = 2.0 * np.pi * i / len(description)
        bb = t.get_window_extent(renderer=r)
        bbdata = bb.transformed(trans)
        radius = 1.2 + bbdata.width / 2.
        position = (radius * np.cos(theta), radius * np.sin(theta))
        t.set_position(position)
        t.set_rotation(theta * 360.0 / (2.0 * np.pi))
        t.set_clip_on(False)

    # Display the graph
    plt.axis('off')
    fig.tight_layout()
    fig.savefig(f"{save}/interaction.png")
    plt.show(block=False)


def get_row_and_col(bact_df, taxon):
    """
    Get the row and column of the iMic image that consists a certain taxon.
    :param bact_df: Dataframe with bact names according to the image order (dataframe).
    :param taxon: Taxonomy level (int).
    :return: col_index (int), row_index (int).
    """
    row_index, col_index = bact_df.index[bact_df.eq(taxon).any(axis=1)][0], bact_df.columns[bact_df.eq(taxon).any()]
    return col_index, row_index


def calc_unique_corr(bact_df, taxon, imgs, tag, eval="corr"):
    """
    Apply post hoc test with tag to a specific taxon
    :param bact_df: Dataframe with bact names according to the image order (dataframe).
    :param taxon: Taxonomy level (int).
    :param imgs: List of loaded iMic images (list).
    :param tag: Tag dataframe with a column named "Tag" (dataframe)
    :param eval: Evaluation method if the tag is binary - "man", if the tag is categorial -
    "category", if the tag is continuous - "corr".
    :return: col_index(int), row_index (int), scc = post hoc score (float), p = p-value of post hoc test (float).
    """
    col_index, row_index = get_row_and_col(bact_df, taxon)
    first_col_index = list(col_index)[0]
    result = [imgs[i, row_index, first_col_index] for i in range(imgs.shape[0])]
    # calc corr
    if eval == "corr":
        scc, p = spearmanr(result, tag)

    elif eval == "category":
        grouped_data = []
        group_labels = []

        # Replace 'Group1', 'Group2', etc., with the actual group names
        for group_name in tag.values.unique():
            group_data_indexes = np.where(tag == group_name)[0]
            grouped_data.append(group_data_indexes)
            group_labels.append(group_name)

        # Perform the Kruskal-Wallis test
        scc, p = kruskal(*grouped_data)


    elif eval == "man":
        zero_indexes = np.where(tag == 0)[0]
        one_indexes = np.where(tag == 1)[0]
        otu0 = [result[i] for i in zero_indexes.tolist()]
        otu1 = [result[i] for i in one_indexes.tolist()]

        # # Combine the samples and assign group labels
        combined_data = np.concatenate([otu0, otu1])
        group_labels = ['Sample 1'] * len(otu0) + ['Sample 2'] * len(otu1)

        # Create a DataFrame to store the combined data and group labels
        df___ = pd.DataFrame({'Data': combined_data, 'Group': group_labels})

        # Sort the combined dataset in ascending order
        df_sorted = df___.sort_values(by='Data')

        # Assign ranks to the observations
        df_sorted['Rank'] = df_sorted['Data'].rank()

        # Calculate the sum of the ranks for each group
        R1 = df_sorted[df_sorted['Group'] == 'Sample 1']['Rank'].sum()
        U1 = len(otu0) * len(otu1) + len(otu0) * (len(otu0) + 1) / 2 - R1
        try:
            scc, p = mannwhitneyu(otu0, otu1)
        except:
            scc, p = 0, 0
        scc = U1 - (len(otu0) * len(otu1)) / 2 + 0.5

    return col_index, row_index, scc, p


def calculate_all_imgs_tag_corr(folder, tag, start_i, eval="corr", sis=None, correct_first=False,
                                mode="test", shuffle=False):
    """
    Calculate the post hoc test to all the taxa over all images and build a df of scores and p-values.
    :param folder:  Folder where the images are saved (str).
    :param tag: Tag dataframe with a column named "Tag" (dataframe).
    :param start_i: Starting taxonomy for the post hoc test (int)
    :param eval: Evaluation method if the tag is binary - "man", if the tag is categorical -
    "category", if the tag is continuous - "corr" (str).
    :param sis: Determines whether to apply sister correction. One of "Bonferroni" or "No".
    :param correct_first: Determines whether to apply FDR correction to the starting taxonomy (Bullian).
    :param mode: Mode of the miMic test - "test" or "plot" (str).
    :param shuffle: Determines whether to shuffle the tag (Bullian).
    :return: Dataframe of corrs (dataframe).
    """
    img_arrays, names = load_img(folder, tag)
    tag = tag.loc[names]
    tag = tag.reset_index()
    if shuffle:
        np.random.shuffle(tag["Tag"])
    bact_names = np.load(f'{folder}/bact_names.npy', allow_pickle=True)
    bact_names_df = pd.DataFrame(bact_names)

    dict_corrs = dict()
    dict_ps = dict()
    all_ps = dict()
    all_stat = dict()

    different_tax_in_level = list(set(bact_names_df.iloc[start_i]))  # tax 1
    different_tax_in_level = [string for string in different_tax_in_level if string.strip()]

    def binary_rec_by_pval(different_tax_in_level, eval="corr", sis=None):
        """
        Apply post hoc test along the cladogram trajectories.
        :param different_tax_in_level: List of unique tax in a certain taxonomy level in the iMic image (list).
        :param eval: Evaluation method if the tag is binary - "man", if the tag is categorical -
        "category", if the tag is continuous - "corr" (str).
        :param sis: Determines whether to apply sister correction. One of "Bonferroni" or "No".
        :return: None
        """
        for tax in different_tax_in_level:
            # Stop condition
            if tax == '' or tax == "0.0" or tax is None:
                # Leaf in the middle of the tree
                return "leaf"

            col_index, row_index, scc, p = calc_unique_corr(bact_names_df, tax, img_arrays, tag, eval)
            all_ps[tax] = p
            all_stat[tax] = scc
            if p >= 0.05:
                # Not significant - stop
                continue
            if (row_index + 1) >= bact_names_df.shape[0]:
                # Leaf in the end of the tree
                dict_corrs[tax] = scc
                dict_ps[tax] = p
                continue

            all_sons = set(bact_names_df[col_index].loc[row_index + 1])
            ret = binary_rec_by_pval(all_sons, eval, sis)
            if ret == "leaf":
                # This is the leaf:
                dict_corrs[tax] = scc
                dict_ps[tax] = p

            if sis == "bonferroni" and len(all_sons) > 1 and len(all_sons.intersection(dict_ps.keys())) > 0:
                sons_pv = {k: all_ps[k] for k in all_sons}
                min_son = min(sons_pv, key=sons_pv.get)
                del sons_pv[min_son]

                rejected_r, corrected_p_values_r, _, _ = smt.multipletests(list(sons_pv.values()),
                                                                           method="bonferroni")
                for e, son in enumerate(sons_pv):
                    if corrected_p_values_r[e] >= 0.05:
                        for bact in [k for k in dict_ps.keys() if son in k]:
                            del dict_ps[bact]
                    else:
                        dict_ps[son] = corrected_p_values_r[e]

    def rec_all_leafs(different_tax_in_level, eval, correct_first):
        """
        Calculate post hoc test along the cladogram trajectories recursively.
        :param different_tax_in_level: List of unique tax in a certain taxonomy level in the iMic image (list).
        :param eval: Evaluation method if the tag is binary - "man", if the tag is categorical -
        "category", if the tag is continuous - "corr" (str).
        :param correct_first: Determines whether to apply FDR correction to the starting taxonomy (Bullian).
        :return: Dataframe of corrs (dataframe).
        """
        for tax in different_tax_in_level:
            # Stop condition
            if tax == '':
                # Leaf in the middle of the tree
                return "leaf"

            col_index, row_index, scc, p = calc_unique_corr(bact_names_df, tax, img_arrays, tag, eval)
            if (row_index + 1) >= bact_names_df.shape[0]:
                # Leaf in the end of the tree
                dict_corrs[tax] = scc
                dict_ps[tax] = p
                continue

            all_sons = set(bact_names_df[col_index].loc[row_index + 1])
            ret = rec_all_leafs(all_sons, eval, correct_first)
            if ret == "leaf":
                # This is the leaf:
                dict_corrs[tax] = scc
                dict_ps[tax] = p

    binary_rec_by_pval(different_tax_in_level, eval, sis)

    if correct_first:
        all_ps_df = pd.Series(all_ps)
        all_ps_df = all_ps_df.to_frame()
        all_ps_df["len"] = [len(i.split(";")) for i in all_ps_df.index]
        all_ps_df["s"] = pd.Series(all_stat)
        to_test = all_ps_df[all_ps_df["len"] == start_i]
        if len(to_test.index) > 1:
            rejected_r, corrected_p_values_r, _, _ = smt.multipletests(list(to_test[0].values),
                                                                       method="bonferroni")
            to_test[0] = corrected_p_values_r
            to_throw = to_test[to_test[0] > 0.05]
            all_ps_df.loc[to_test.index, 0] = corrected_p_values_r.tolist()

    if mode == "plot":
        if shuffle == False:
            # Open folder for plots saving
            directory = "plots"
            # Check if the directory already exists
            if not os.path.exists(directory):
                # If it doesn't exist, create the directory
                os.makedirs(directory)

            # Plot histograms of all different taxonomy levels (1)
            mpl.rc('font', family='Times New Roman')
            SIZE = 15
            df_hists = pd.DataFrame(columns=[1, 2, 3, 4, 5, 6, 7])
            column_labels = [1, 2, 3, 4, 5, 6, 7]
            line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
            line_widths = [1, 1.5, 2, 1, 1.5, 2, 1]
            fig, ax = plt.subplots(figsize=(12, 4))
            df_hists[1] = img_arrays[:, 1, :].flatten()
            df_hists[2] = img_arrays[:, 2, :].flatten()
            df_hists[3] = img_arrays[:, 3, :].flatten()
            df_hists[4] = img_arrays[:, 4, :].flatten()
            df_hists[5] = img_arrays[:, 5, :].flatten()
            df_hists[6] = img_arrays[:, 6, :].flatten()
            df_hists[7] = img_arrays[:, 7, :].flatten()
            stds = df_hists.std()
            means = df_hists.mean()
            for i, col_label in enumerate(column_labels):
                df_hists[col_label].plot.density(color="k", alpha=0.9, label=str(col_label), linestyle=line_styles[i],
                                                 linewidth=line_widths[i],
                                                 ax=ax)
                print(f"Taxonomy {col_label}")
                print(stats.kstest(df_hists[col_label], 'norm', args=(means.iloc[i], stds.iloc[i])))

            plt.legend(fontsize=15)
            plt.ylabel("Frequency", fontsize=SIZE)
            plt.xlabel("Log abundances", fontsize=SIZE)
            plt.xticks(fontsize=SIZE)
            plt.yticks(fontsize=SIZE)
            plt.xlim([-3, 3])
            plt.tight_layout()
            plt.savefig(f"{directory}/hist.png")
            plt.show(block=False)

            # Plot interacrions plot (2)
            build_interactions(all_ps_df, bact_names_df, img_arrays, directory)

            # Plot correlations within family (3)
            imgs_p, imgs_s = build_img_from_table(all_ps_df, 0, "s", bact_names_df)

            # check family test
            mpl.rc('font', family='Times New Roman')
            list_of_families = bact_names[5]
            uniques = list(set(bact_names[5]))
            dict_pos = dict()
            dict_neg = dict()
            for f in uniques:
                f_indexes = [i for i, value in enumerate(list_of_families) if value == f]
                all_children = imgs_s[6:, f_indexes]
                non_zero = all_children.sum().sum()
                if non_zero != 0.0:
                    negatives = (all_children < 0.0).sum()
                    positives = (all_children > 0.0).sum()
                    dict_pos[f] = positives
                    dict_neg[f] = negatives

            df_to_plot = pd.DataFrame(index=list(dict_pos.keys()), columns=['Positives', 'Negatives'])
            df_to_plot['Positives'] = list(dict_pos.values())
            df_to_plot['Negatives'] = list(dict_neg.values())
            df_to_plot.index = [i.split(";")[-1] for i in df_to_plot.index]
            df_to_plot.plot(kind="barh", color=["blue", "red"], figsize=(4, 4))
            plt.xlabel("Number", fontsize=SIZE)
            plt.xticks(fontsize=SIZE)
            plt.yticks(fontsize=SIZE)
            plt.tight_layout()
            plt.savefig(f"{directory}/corrs_within_family.png")
            plt.show(block=False)

            # Plot correlations on tree
            creare_tree_view(bact_names, imgs_p, imgs_s, directory)

    series_corrs = pd.Series(dict_corrs).to_frame("scc")
    series_ps = pd.Series(dict_ps).to_frame("p")
    df_corrs = pd.concat([series_corrs, series_ps], axis=1)
    df_corrs.index = [i.split("_")[0] for i in df_corrs.index]
    df_corrs = df_corrs.groupby(df_corrs.index).mean()
    if correct_first and start_i > 1:
        df_filtered = df_corrs[~df_corrs.index.str.contains('|'.join(to_throw.index))]
        return df_filtered
    else:
        return df_corrs

def calculate_p_value(img_arrays,taxon,tag):
    """
    Calculate p-value of nested GLM.
    :param img_arrays: Ndarray of iMic images (ndarray)
    :param taxon: Taxonomy level (int).
    :param tag: Tag dataframe with a column named "Tag" (dataframe).
    :return: nested GLM p-value (float)
    """
    num_samples = img_arrays.shape[0]
    features = img_arrays[:,:taxon+1,:].reshape(num_samples, -1)
    if taxon == 0:
        model = sm.OLS(tag, features).fit()
        return model.f_pvalue
    else:
        full_model = sm.OLS(tag, features).fit()
        upper_features = img_arrays[:,:taxon,:].reshape(num_samples, -1)
        upper_model = sm.OLS(tag, upper_features).fit()
        K1 = len(list(upper_model.params.index))
        ALL = len(list(full_model.params.index))
        K2 = ALL - K1
        S_A = upper_model.ssr
        S_B = full_model.ssr
        Z = (S_A/(K1 - 1))/S_B/(num_samples - K2)
        p_value = 1 - stats.f.cdf(Z, K1 - 1, num_samples - K2)
        return p_value





def apply_nested_anova(folder, tag, mode="test", eval="man"):
    """
    Apply apriori nested test (ANOVA- for binary and categorical tags and GLM for continuous).
    :param folder: Folder where the images are saved (str).
    :param tag: Tag dataframe with a column named "Tag" (dataframe).
    :param mode: Mode of the miMic test - "test" or "plot" (str).
    :param eval: Evaluation method if the tag is binary - "man", if the tag is categorical -
        "category", if the tag is continuous - "corr" (str).
    :return: In "test" mode returns the p-value, in "plot" mode returns a dataframe with the nested p-values
    of each taxonomy level.
    """
    img_arrays, bact_names, tag = load_from_folder(folder, tag)
    p_vals_df = pd.DataFrame(index=[1, 2, 3, 4, 5, 6, 7], columns=["nested-p"])
    if eval != "corr":
        for taxon_lavel in range(img_arrays.shape[1]):
            layer = bact_names[taxon_lavel, :]
            bbb = img_arrays[:, :taxon_lavel + 1, :]

            unique_values, indices = np.unique(layer, return_index=True)

            aaa = bbb[:, :, indices]
            num_of_bacts = aaa.shape[-1]
            tag_f = tag.to_numpy().repeat(num_of_bacts, axis=1).reshape(-1, 1, num_of_bacts)
            d_with_tag = np.hstack([aaa, tag_f])
            base_data = d_with_tag.swapaxes(1, 2).reshape(aaa.shape[0] * num_of_bacts, -1)

            df = pd.DataFrame(base_data).add_prefix("level")
            df = df.rename(columns={f"level{taxon_lavel + 1}": "Tag"})

            ll = "/".join(df.columns[:-1])
            model = ols(f'Tag ~ {ll}', data=df.astype(float)).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            # print(anova_table)
            P = anova_table.iloc[taxon_lavel]["PR(>F)"]
            p_vals_df["nested-p"][taxon_lavel + 1] = P
            if P < 0.05:
                print(f"Test succeeded in level {taxon_lavel + 1}:P = {P}")
                if mode == "test":
                    break
    else:
        # Apply regression on all the flattened tree
        for taxon_lavel in range(img_arrays.shape[1]):
            P = calculate_p_value(img_arrays,taxon_lavel,tag)
            p_vals_df["nested-p"][taxon_lavel + 1] = P
            if P < 0.05:
                print(f"Test succeeded in level {taxon_lavel + 1}:P = {P}")
                if mode == "test":
                    break

    if mode == "test":
        return P
    else:
        return p_vals_df


def plot_rp_sp_anova_p(df, save):
    """
    Plot RP vs SP over the different taxonomy levels and the p-values of the apriori test as function of taxonomy.
    :param df: RP and SP dataframe of the post hoc test applied (dataframe).
    :param save: Name of folder to save the plot (str).
    :return: None. Display the RP vs SP vs apriori p-values as function of taxonomy.
    """
    TAX = 1
    SIZE = 15
    mpl.rc('font', family='Times New Roman')
    real_sh = df[['RP', 'SP']]

    # Create the first y-axis (bar plot)
    ax1 = real_sh.plot(kind="bar", color=["blue", "red"], figsize=(4, 4), rot=0)
    ax1.set_ylabel("Number of significants", fontsize=SIZE)
    ax1.set_xlabel("Starting Taxonomy", fontsize=SIZE)
    ax1.tick_params(axis="x", labelsize=SIZE)
    ax1.tick_params(axis="y", labelsize=SIZE)
    plt.legend(loc="upper center")

    # Create a second y-axis (scatter plot)
    ax2 = ax1.twinx()
    # Set the y-ticks for ax2 to match df["nested ANOVA"] values
    ax2.set_yticks([0, 5, 10, 15, 20, 25, 30])
    ax2.set_ylabel("-log10(p-value)", fontsize=SIZE, color="green")  # Customize the label as needed
    ax2.tick_params(axis="y", labelsize=SIZE, color="green")  # Adjust tick label font size
    logged = df["nested-p"].apply(lambda x: -np.log10(x))
    ax2.plot(ax1.get_xticks(), logged.values, linestyle="-", marker="o", markersize=5, color="green",
             label="Line Plot")
    logged_ = logged.values[TAX:]

    ax2.plot(ax1.get_xticks()[TAX:], logged_, linestyle="-", marker="o",
             markersize=5, color="grey",
             )

    plt.tight_layout()
    plt.savefig(f"{save}/tax_vs_rp_sp_anova_p.png")
    plt.show(block=False)


def calculate_rsp(df, tax, save):
    """
    Calculate RSP score for different betas and create the appropriate plot.
    :param df: RP and SP dataframe of the post hoc test applied (dataframe).
    :param tax: Starting taxonomy selected in the post hoc test (int).
    :param save: Name of folder to save the plot (str).
    :return: None. Display the RSP score as a function of beta.
    """
    list_beta = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
                 0.95, 1]
    to_plot = pd.DataFrame(index=list_beta, columns=["RSP"])
    for beta in list_beta:
        RP = df["RP"][tax]
        SP = df["SP"][tax]
        to_plot["RSP"][beta] = (beta * RP - SP) / (beta * RP + SP)

    # plot RSP vs beta
    SIZE = 15
    mpl.rc('font', family='Times New Roman')
    to_plot.plot(legend=False, figsize=(4, 4), color="grey")
    plt.ylim([0.5, 1.1])  # to_plot.min().values[0]-0.05
    plt.xlabel(r'$\beta$', fontsize=SIZE)
    plt.ylabel(r'RSP($\beta$)', fontsize=SIZE)
    plt.xticks(fontsize=SIZE)
    plt.yticks(fontsize=SIZE)
    plt.tight_layout()
    plt.savefig(f"{save}/rsp_vs_beta.png")
    plt.show(block=False)


def apply_mimic(folder, tag, eval="man", sis="bonferroni", correct_first=True, mode="test", save=False, tax=None):
    """
    Apply the apriori ANOVA test and the post hoc test of miMic.
    :param folder: Folder path of the iMic images (str).
    :param tag: Tag dataframe with a column named "Tag" (dataframe).
    :param eval: Evaluation method if the tag is binary - "man", if the tag is categorical -
        "category", if the tag is continuous - "corr" (str).
    :param sis: Determines whether to apply sister correction. One of "Bonferroni" or "No".
    :param correct_first: Determines whether to apply FDR correction to the starting taxonomy (Bullian).
    :param mode: Mode of the miMic test - "test" or "plot" (str).
    :param save: Determines whether to save the final corrS_df of the miMic test (Bullian).
    :param tax: Starting taxonomy selected in the post hoc test (int).
    :return: If the apriori test is not significant, prints that and does not continue to the next step. If the
    apriori test is significant prints that and continues to the post hoc test. Prints the number of RPs found in each
    taxonomy level. At last if the the save variable is True it saves the df_corrs. It returns the selected starting taxonomy in the test mode.
    If the function is in "plot" mode it returns 6 plots.
    """
    if mode == "test":
        # Apply apriori nested ANOVA test
        p = apply_nested_anova(folder, tag, mode=mode, eval=eval)
        if p > 0.05:
            print(f"Apriori nested ANOVA test is not significant, getting P = {p}.")

        else:
            print(f"Apriori nested ANOVA test is significant.\nTrying postHC miMic test.")

            # Apply post HOC miMic test
            for t in [1, 2, 3]:
                print(f"Taxonomy is {t}")
                df_corrs = calculate_all_imgs_tag_corr(folder, tag, t, eval=eval,
                                                       sis=sis, correct_first=correct_first, mode=mode, shuffle=False)
                df_corrs = df_corrs.dropna()
                n_significant = (df_corrs["p"] < 0.05).sum()
                print(f"Number of RP: {n_significant}")
                if n_significant > 0:
                    break

                if t == 3:
                    if n_significant == 0:
                        print("Cannot find significant taxa by starting at 1 of the first 3 taxonomy levels :(")

            # save statistic and p-values df
            if save:
                df_corrs.to_csv(f"{folder}/df_corrs.csv")
            return t


    elif mode == "plot":
        # Apply apriori nested ANOVA test
        p_vals_df = apply_nested_anova(folder, tag, mode=mode)

        # Apply post HOC miMic test
        num_s_df = pd.DataFrame(index=[1, 2, 3, 4, 5, 6, 7], columns=["RP", "SP"])
        for t in [1, 2, 3, 4, 5, 6, 7]:
            print(f"Taxonomy is {t}")
            df_corrs_real = calculate_all_imgs_tag_corr(folder, tag, t, eval=eval,
                                                        sis=sis, correct_first=correct_first, mode="test",
                                                        shuffle=False)

            df_corrs_real = df_corrs_real.dropna()
            n_significant = (df_corrs_real["p"] < 0.05).sum()
            num_s_df["RP"][t] = n_significant
            print(f"Number of RP: {n_significant}")

        for t in [1, 2, 3, 4, 5, 6, 7]:
            print(f"Taxonomy is {t}")
            df_corrs_shuffled = calculate_all_imgs_tag_corr(folder, tag, t, eval=eval,
                                                            sis=sis, correct_first=correct_first, mode="test",
                                                            shuffle=True)
            df_corrs_shuffled = df_corrs_shuffled.dropna()
            n_significant_s = (df_corrs_shuffled["p"] < 0.05).sum()
            num_s_df["SP"][t] = n_significant_s
            print(f"Number of SP: {n_significant_s}")

        # Build a common table for RP, SP, and nested ANOVA p-values
        results_to_plot = pd.concat([num_s_df, p_vals_df], axis=1)
        if not os.path.exists("plots"):
            # If it doesn't exist, create the directory
            os.makedirs("plots")
        # Plot RP, SP, ANOVA vs taxonomy
        plot_rp_sp_anova_p(results_to_plot, "plots")
        # Plot RSP(beta) vs beta
        calculate_rsp(num_s_df, tax, "plots")
        # Plot inside plots on the taxonomy selected
        calculate_all_imgs_tag_corr(folder, tag, tax, eval=eval,
                                    sis=sis, correct_first=correct_first, mode="plot", shuffle=False)

        c = 0
