import warnings
import MIPMLP
import samba
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from scipy.stats import spearmanr
from statsmodels.formula.api import ols
import matplotlib as mpl
import matplotlib
import matplotlib.cm as cm

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal
import networkx as nx
import ete3
from .tax_tree_create import create_tax_tree
from ete3 import NodeStyle, TextFace, add_face_to_node, TreeStyle
from copy import deepcopy
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
            # Check if the file name (without the extension) is in the index of a variable 'tag'
            if file_path.split("\\")[-1].replace(".npy", "") in [str(i) for i in tag.index]:
                arrays.append(np.load(file_path, allow_pickle=True, mmap_mode='r'))
                names.append(file_path.split("\\")[-1].replace(".npy", ""))
            # depends on your computer system
            elif file_path.split("/")[-1].replace(".npy", "") in [str(i) for i in tag.index]:
                arrays.append(np.load(file_path, allow_pickle=True, mmap_mode='r'))
                names.append(file_path.split("/")[-1].replace(".npy", ""))

    final_array = np.stack(arrays, axis=0)
    return final_array, names


def load_from_folder(samba_output, folder, tag):
    """
    Load all images
    :param samba_output: Samba outputs, if you already have them- miMic will read it from the folder you specified,
    else miMic will apply samba and set `samba_output` to None.
    :param folder: Folder where the images are saved (str)
    :param tag: Tag dataframe with a column named "Tag" (dataframe)
    :return: img_arrays-ndarray of images (ndarray), bact_names-ndarray of taxa names (ndarray), tag (dataframe)
    """
    if samba_output is None:
        bact_names = np.load(f'{folder}/bact_names.npy', allow_pickle=True)
        img_arrays, names = load_img(folder, tag)
    elif samba_output is not None:
        img_arrays, bact_names, ordered_df = samba_output
        names = ordered_df.index
    tag.index = [str(i) for i in tag.index]
    names= [str(i) for i in names]
    tag = tag.loc[names]
    index_name = tag.index.name
    if index_name == None:
        index_name = "index"
    tag = tag.reset_index()
    del tag[index_name]

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
            updated = "k__" + i.split(";")[0] + ";" + "p__" + i.split(";")[1] + ";" + "c__" + \
                      i.split(";")[2]
        elif j == 4:
            updated = "k__" + i.split(";")[0] + ";" + "p__" + i.split(";")[1] + ";" + "c__" + i.split(";")[
                2] + ";" + "o__" + i.split(";")[3]

        elif j == 5:
            updated = "k__" + i.split(";")[0] + ";" + "p__" + i.split(";")[1] + ";" + "c__" + i.split(";")[
                2] + ";" + "o__" + i.split(";")[3] + ";" + "f__" + i.split(";")[4]

        elif j == 6:
            updated = "k__" + i.split(";")[0] + ";" + "p__" + i.split(";")[1] + ";" + "c__" + i.split(";")[
                2] + ";" + "o__" + i.split(";")[3] + ";" + "f__" + i.split(";")[4] + ";" + "g__" + \
                      i.split(";")[5]

        elif j == 7:
            updated = "k__" + i.split(";")[0] + ";" + "p__" + i.split(";")[1] + ";" + "c__" + i.split(";")[
                2] + ";" + "o__" + i.split(";")[3] + ";" + "f__" + i.split(";")[4] + ";" + "g__" + i.split(";")[
                          5] + ";" + "s__" + i.split(";")[6]

        otu_train_cols.append(updated)
    return otu_train_cols


def rgba_to_hex(rgba):
    """
    Convert rgba to hex.
    :param rgba: rgba color (tuple).
    :return: hex color (str).
    """
    return matplotlib.colors.rgb2hex(rgba)


def creare_tree_view(names, mean_0, mean_1, directory, threshold_p=0.05, family_colors=None):
    """
    Create correlation cladogram, such that tha size of each node is according to the -log(p-value), the color of
    each node represents the sign of the post hoc test, the shape of the node (circle, square,sphere) is based on
    miMic, Utest, or both results accordingly, and if `colorful` is set to True, the background color of the node will be colored based on the family color.
    :param names:  List of sample names (list) :param mean_0: 2D ndarray of the images filled with the post hoc p-values (ndarray).
    :param mean_1:  2D ndarray of the images filled with the post hoc scores (ndarray). :param directory: Folder to
    save the correlation cladogram (str) :param family_colors: Dictionary of family colors (dict) :return: None
    """
    T = ete3.PhyloTree()
    u_test = pd.read_pickle("u_test_without_mimic.pkl")
    flag_check_exist = False
    try:
        mimic_and_utest = pd.read_pickle("miMic&Utest.pkl")
        flag_check_exist = True
    except:
        mimic_and_utest = None
    # Get the number of rows and columns in the ndarray
    num_rows, num_cols = names.shape
    first_non_empty_cells = [None] * num_cols

    # first_non_empty_cells will store all the leaves
    for col_idx in range(num_cols):
        flag = False
        for row_idx in reversed(range(num_rows)):
            # if the kingdom is unassigned, we will set the first non empty cell to 0.0
            if 'unassigned' in names[1, col_idx].lower():
                first_non_empty_cells[col_idx] = '0.0'
                break
            # Check if the current cell is empty (None or nan or 0.0) we can check it if it contains at least one letter
            contains_letter = any(char.isalpha() for char in names[row_idx, col_idx])
            if not contains_letter:
                continue

            # If the current cell is not empty, update the first non-empty cell for this column
            if first_non_empty_cells[col_idx] is None:
                if mean_0[row_idx, col_idx] < threshold_p and mean_0[row_idx, col_idx] != 0.0:
                    flag = True
                    first_non_empty_cells[col_idx] = names[row_idx, col_idx]
                    names[row_idx, col_idx] = names[row_idx, col_idx]
                if flag == False:
                    first_non_empty_cells[col_idx] = '0.0'
                break

    # removing all the leaves that are not significant, those labeled with 0.0
    first_non_empty_cells = [cell for cell in first_non_empty_cells if cell != '0.0']

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

            # for u test without mimic results the name is fixed to the correct version of the taxonomy
            # for the mimic results the name is the actual name
            u_test_name = create_list_of_names([(';'.join(s[0]))])[0]

            actual_name = ";".join(s[0])

            if actual_name == 'k__Bacteria;p__Proteobacteria;c__Betaproteobacteria;o__Burkholderiales;f__Oxalobacteraceae_0' or u_test_name == "k__Bacteria;p__Proteobacteria;c__Betaproteobacteria;o__Burkholderiales;f__Oxalobacteraceae_0":
                c = 0

            if s[0][-1] not in T or not any([anc.species == a for anc, a in
                                             zip(T.search_nodes(name=s[0][-1])[0].get_ancestors()[:-1],
                                                 reversed(s[0]))]):
                t = T
                if len(s[0]) != 1:
                    t = T.search_nodes(full_name=s[0][:-1])[0]

                if flag_check_exist:
                    mimic_and_utest_index = mimic_and_utest.index[mimic_and_utest.index.isin([u_test_name])]
                    if mimic_and_utest_index != None:
                        t = t.add_child(name=s[0][-1])
                        t.species = s[0][-1]
                        t.add_feature("full_name", s[0])
                        t.add_feature("max_0_grad",
                                      -np.log10(mimic_and_utest.loc[mimic_and_utest_index]['p'].iloc[0] + epsilon))
                        t.add_feature("max_1_grad", mimic_and_utest.loc[mimic_and_utest_index]['scc'].iloc[0])
                        t.add_feature("max_2_grad", mimic_and_utest.loc[mimic_and_utest_index]['p'].iloc[0])
                        t.add_feature("shape", "sphere")

                        # if the name is including family level, we will set the family color
                        if family_colors != None:
                            split_name = str(mimic_and_utest_index[0]).split(';')
                            if len(split_name) >= 5:
                                family_name = split_name[4].split('__')[1].split('_')[0]
                                family_color = family_colors.get(family_name, "nocolor")
                            else:
                                # if the name is not including family level, we will not set a family color
                                family_color = "nocolor"
                            t.add_feature("family_color", family_color)

                        continue

                # taking the node index, if the node is in u_test (without mimic results)
                in_utest_index = u_test.index[u_test.index.isin([u_test_name])]
                if in_utest_index != None:

                    t = t.add_child(name=s[0][-1])
                    t.species = s[0][-1]
                    t.add_feature("full_name", s[0])
                    t.add_feature("max_0_grad", -np.log10(u_test.loc[in_utest_index]['p'].iloc[0] + epsilon))
                    t.add_feature("max_1_grad", u_test.loc[in_utest_index]['scc'].iloc[0])
                    t.add_feature("max_2_grad", u_test.loc[in_utest_index]['p'].iloc[0])
                    t.add_feature("shape", "square")

                    # if the name is including family level, we will set the family color
                    if family_colors != None:
                        split_name = str(in_utest_index[0]).split(';')
                        if len(split_name) >= 5:
                            family_name = split_name[4].split('__')[1].split('_')[0]
                            family_color = family_colors.get(family_name, "nocolor")
                        else:
                            # if the name is not including family level, we will not set a family color
                            family_color = "nocolor"
                        t.add_feature("family_color", family_color)

                    continue

                # nodes in mimic results without u-test

                t = t.add_child(name=s[0][-1])
                t.species = s[0][-1]
                t.add_feature("full_name", s[0])
                t.add_feature("max_0_grad", -np.log10(mean_0[names == actual_name].mean() + epsilon))
                t.add_feature("max_1_grad", mean_1[names == actual_name].mean())
                t.add_feature("max_2_grad", mean_0[names == actual_name].mean())
                t.add_feature("shape", "circle")

                if family_colors != None:
                    # setting the family color
                    split_name = actual_name.split(';')
                    if len(split_name) >= 5:
                        family_color = family_colors.get(actual_name.split(';')[4].split('_')[0], "nocolor")
                    else:
                        family_color = "nocolor"
                    t.add_feature("family_color", family_color)

    T0 = T.copy("deepcopy")
    bound_0 = 0
    for t in T0.get_descendants():
        nstyle = NodeStyle()
        nstyle["size"] = 30
        nstyle["fgcolor"] = "gray"

        name = ";".join(t.full_name)

        if (t.max_1_grad > bound_0) and (t.max_2_grad < threshold_p):
            nstyle["fgcolor"] = "blue"
            nstyle["size"] = t.max_0_grad * 17

            if t.shape == "square":
                nstyle["shape"] = "square"

            if t.shape == "sphere":
                nstyle["shape"] = "sphere"
            if t.shape == "circle":
                nstyle["shape"] = "circle"

            if family_colors != None:
                if t.family_color != "nocolor":
                    hex_color = rgba_to_hex(t.family_color)
                    nstyle['bgcolor'] = hex_color

        elif (t.max_1_grad < bound_0) and (t.max_2_grad < threshold_p):

            nstyle["fgcolor"] = "red"
            nstyle["size"] = t.max_0_grad * 17

            if t.shape == "square":
                nstyle["shape"] = "square"
            if t.shape == "sphere":
                nstyle["shape"] = "sphere"
            if t.shape == "circle":
                nstyle["shape"] = "circle"

            if family_colors != None:
                if t.family_color != "nocolor":
                    hex_color = rgba_to_hex(t.family_color)
                    nstyle['bgcolor'] = hex_color

        # if the node is not significant we will still color it by its family color
        if family_colors != None:
            if t.family_color != "nocolor":
                hex_color = rgba_to_hex(t.family_color)
                nstyle['bgcolor'] = hex_color

        elif not t.is_root():
            # if the node is not significant, we will detach it
            if not any([anc.max_0_grad > bound_0 for anc in t.get_ancestors()[:-1]]) and not any(
                    [dec.max_0_grad > bound_0 for dec in t.get_descendants()]):
                t.detach()
        t.set_style(nstyle)

    for node in T0.get_descendants():
        if node.is_leaf():
            # checking if the name is ending with _{digit} if so i will remove it
            if node.name[-1].isdigit() and node.name.endswith(f'_{node.name[-1]}'):
                node.name = node.name[:-1]
            name = node.name.replace('_', ' ').capitalize()
            if name == "":
                name = node.get_ancestors()[0].replace("_", " ").capitalize()
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
        #control branch width
        node.img_style["hz_line_width"] = 18
        node.img_style["vt_line_width"] = 18

        if node.is_leaf():
            tax = D[len(node.full_name)]
            if len(node.full_name) == 7:
                name = node.up.name.replace("[", "").replace("]", "") + " " + node.name.lower()
            else:
                name = node.name

            F = TextFace(f"{name} {tax} ", fsize=100, ftype="Arial")  # {tax}
            add_face_to_node(F, node, column=0, position="branch-right")

    ts.layout_fn = my_layout
    T0.show(tree_style=(ts))
    T0.render(f"{directory}/correlations_tree.svg", tree_style=deepcopy(ts))


def convert_original(name):
    components = name.split(";")

    # Extract the names after the prefixes and join them with semicolons
    formatted_string = ";".join(component.split("__")[1] for component in components if component)

    return formatted_string


def build_interactions(bact_names, img_array, save, family_colors, threshold_p=0.05, THRESHOLD=0.5):
    """
    Plot interaction network between the significant taxa founded by miMic, such that each node color
    is according to the sigh of the post hoc test with the tag, its shape is according to its order, and
    its edge width is according to the correlation between the pair. There are edges only between pirs with
    correlation above the threshold.
    :param bact_names: Dataframe with bact names according to the image order (dataframe)
    :param img_array: List of loaded iMic images (list).
    :param save: Name of folder to save the plot (str).
    :param family_colors: Dictionary of family colors (dict).
    :param threshold_p: The threshold for significant value (float).
    :param THRESHOLD: The threshold for having an edge (float).
    :return: None. It creates a plot.
    """
    all_ps = pd.read_pickle("df_corrs.pkl")
    all_ps['len'] = [len(i.split(";")) for i in all_ps.index]
    all_ps.rename(columns={'p': '0'}, inplace=True)
    all_ps.rename(columns={'scc': 's'}, inplace=True)

    only_significant = all_ps[all_ps['0'] < threshold_p]
    # Initialize a DataFrame to store the first index
    df_index = pd.DataFrame(index=only_significant.index, columns=["index"])

    for pixel in only_significant.index:
        row = all_ps.loc[pixel]["len"]
        pixel_o = convert_original(pixel)
        indexes = min(
            [i for i, value in enumerate(bact_names.loc[row]) if pixel_o == value])
        df_index["index"][pixel] = indexes
    only_significant["index"] = df_index["index"]
    # only species
    only_significant = only_significant[only_significant["len"] == 7]

    only_significant["fixed_len"] = [len(i.split(";")) for i in only_significant.index]
    only_significant = only_significant[only_significant["fixed_len"] == 7]
    only_significant["size"] = -np.log10(only_significant['0']) * 30
    only_significant["color"] = ["red" for i in only_significant.index]
    only_significant["color"][only_significant["s"] > 0] = "blue"
    only_significant["order"] = [i.split(";")[3] for i in only_significant.index]

    # Create a mapping of unique orders to node shapes
    unique_orders = only_significant['order'].unique()
    shapes = ['8', '4', '3', '2', '1', 'x', '+', 'D', 'h', 'p', 's', 'o', 'v', '^', '<', '>', 'd', '8', 'P', '*', 'X',
              '_', '|']  # Shapes for unique orders

    shape_mapping = {}
    for i, order in enumerate(unique_orders):
        shape_mapping[order] = shapes[i % len(shapes)]  # Cycle through shapes list using modulo operator

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
            if abs(weight) > THRESHOLD:  # Adjust the threshold as needed
                G.add_edge(i, j, weight=weight)

    # Create a dictionary to map numerical node names to original names
    numerical_to_original_names = {i: name for i, name in enumerate(
        [n.split(";")[-2:][0] + ";" + n.split(";")[-2:][-1] for n in list(inter_corr.index)])}

    # Assign original names to the nodes
    for i, node in G.nodes(data=True):
        # numerical_to_original_names[i] is in the format of "g__X;s__Y"
        parts = numerical_to_original_names[i].split(';')
        g_part = parts[0].split('g__')[1] if 'g__' in parts[0] else ''
        s_part = parts[1].split('s__')[1] if 's__' in parts[1] else ''
        g_part = g_part.rsplit("_", maxsplit=1)[0]
        s_part = s_part.rsplit("_", maxsplit=1)[0]
        node['name'] = g_part + ";" + s_part

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

        if family_colors:
            node_name = create_list_of_names([("".join(node_labels[node]))])[0]
            node_name = node_name.replace("k__", "g__").replace("p__", "s__")
            family_name = [i.split(';')[4] for i in inter_corr.index if
                           node_name in i]
            font_color = family_colors.get(family_name[0].split('__')[1], "black")  # Default font color is black
            t.set_color(font_color)

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
    # result = [imgs[i,row_index,first_col_index] for i in range (imgs.shape[1],-1,-1) if imgs[i,row_index,first_col_index] != 0.0]

    # calc corr
    if eval == "corr":
        scc, p = spearmanr(result, tag['Tag'])

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


def calculate_all_imgs_tag_corr(samba_output, folder, tag, start_i, eval="corr", sis='fdr_bh', correct_first=False,
                                mode="test", threshold_p=0.05, THRESHOLD_edge=0.5, shuffle=False, colorful=None):
    """
    Calculate the post hoc test to all the taxa over all images and build a df of scores and p-values.
    :param samba_output: Samba outputs, if you already have them- miMic will read it from the folder you specified,
    else miMic will apply samba and set `samba_output` to None.
    :param folder:  Folder where the images are saved (str).
    :param tag: Tag dataframe with a column named "Tag" (dataframe).
    :param start_i: Starting taxonomy for the post hoc test (int)
    :param eval: Evaluation method if the tag is binary - "man", if the tag is categorical -
    "category", if the tag is continuous - "corr" (str).
    :param sis: Determines whether to apply sister correction. One of ['fdr_bh', 'bonferroni', 'No']. Default is "fdr_bh" (str).
    :param correct_first: Determines whether to apply FDR correction to the starting taxonomy (Boolean).
    :param mode: Mode of the miMic test - "test" or "plot" (str).
    :param threshold_p: The threshold for significant value (float).
    :param THRESHOLD_edge: The threshold for having an edge in "interaction" plot (float).
    :param shuffle: Determines whether to shuffle the tag (Boolean).
    :param colorful: Determines whether to color the nodes by their family (Boolean).
    :return: Dataframe of corrs (dataframe).
    """
    # If you already have samba outputs, we will read them from the folder you specified
    if samba_output is None:
        img_arrays, names = load_img(folder, tag)

    # If we apply samba on your data
    elif samba_output is not None:
        img_arrays, bact_names, ordered_df = samba_output
        names = ordered_df.index

    tag = tag.loc[names]
    tag = tag.reset_index()
    if shuffle:
        np.random.shuffle(tag["Tag"])

    if samba_output is None:
        bact_names = np.load(f'{folder}/bact_names.npy', allow_pickle=True)
    bact_names_df = pd.DataFrame(bact_names)

    if mode == 'leaves' or start_i == 'noAnova':
        start_i = 0
        df = bact_names_df.replace('0.0', pd.NA)
        df = df.apply(lambda col: col.dropna().iloc[-1])
        # now df contains only the last leaf of each branch
        bact_names_df_leaf = df.to_frame().T

    dict_corrs = dict()
    dict_ps = dict()
    all_ps = dict()
    all_stat = dict()

    if mode == 'leaves' or start_i == 'noAnova':
        # Getting the unique leaves taxa based on 'bac_names_df_leaf', start_i=0 because we extracted all the leaves to one row
        different_tax_in_level = list(set(bact_names_df_leaf.iloc[start_i]))
    else:
        different_tax_in_level = list(set(bact_names_df.iloc[start_i]))

    different_tax_in_level = [string for string in different_tax_in_level if string.strip()]
    different_tax_in_level = sorted(different_tax_in_level, reverse=True)

    def binary_rec_by_pval(different_tax_in_level, eval="corr", sis='fdr_bh'):
        """
        Apply post hoc test along the cladogram trajectories.
        :param different_tax_in_level: List of unique tax in a certain taxonomy level in the iMic image (list).
        :param eval: Evaluation method if the tag is binary - "man", if the tag is categorical -
        "category", if the tag is continuous - "corr" (str).
        :param sis: Determines whether to apply sister correction. One of ['fdr_bh', 'bonferroni', 'No']. Default is "fdr_bh" (str).
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
            if p >= threshold_p:
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

            if sis != "No" and len(all_sons) > 1 and len(all_sons.intersection(dict_ps.keys())) > 0:
                sorted_sons = sorted(list(all_sons))
                sons_pv = {k: all_ps[k] for k in sorted_sons}
                min_son = min(sons_pv, key=sons_pv.get)
                del sons_pv[min_son]

                rejected_r, corrected_p_values_r, _, _ = smt.multipletests(list(sons_pv.values()),
                                                                           method=sis)
                for e, son in enumerate(sons_pv):
                    if corrected_p_values_r[e] >= threshold_p:
                        for bact in [k for k in dict_ps.keys() if son in k]:
                            # the son is not significant so we are giving it a p-value of 0.06
                            all_ps[bact] = threshold_p + 0.01
                            del dict_ps[bact]
                    else:
                        dict_ps[son] = corrected_p_values_r[e]
                        all_ps[son] = corrected_p_values_r[e]

        if sis != "No" and mode == 'leaves':
            p_val_u_test = [bac_p for bac_p in all_ps.values()]
            rejected_r, corrected_p_values_r, _, _ = smt.multipletests(list(p_val_u_test), method=sis)
            for e, tax in enumerate(all_ps.keys()):
                if corrected_p_values_r[e] < threshold_p:
                    dict_ps[tax] = corrected_p_values_r[e]
                    all_ps[tax] = corrected_p_values_r[e]
                else:
                    try:
                        # the leaf is not significant we will label it with a p-value that bigger than the threshold
                        all_ps[tax] = threshold_p + 0.01
                        del dict_ps[tax]
                    except:
                        pass

    binary_rec_by_pval(different_tax_in_level, eval, sis)

    if correct_first:

        all_ps_df = pd.Series(all_ps)
        all_ps_df = all_ps_df.to_frame()

        all_ps_df["len"] = [len(i.split(";")) for i in all_ps_df.index]
        all_ps_df["s"] = pd.Series(all_stat)

        to_test = all_ps_df[all_ps_df["len"] == start_i]
        flag_throw = False
        if len(to_test.index) > 1:
            flag_throw = True
            rejected_r, corrected_p_values_r, _, _ = smt.multipletests(list(to_test[0].values),
                                                                       method=sis)
            to_test[0] = corrected_p_values_r
            to_throw = to_test[to_test[0] > threshold_p]
            all_ps_df.loc[to_test.index, 0] = corrected_p_values_r.tolist()

    if mode == "plot":
        if shuffle == False:
            # Open folder for plots saving
            directory = "plots"
            # Check if the directory already exists
            if not os.path.exists(directory):
                # If it doesn't exist, create the directory
                os.makedirs(directory)

            # Histograms of all different taxonomy levels plot (3)
            mpl.rc('font', family='Times New Roman')
            SIZE = 15
            column_labels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']

            df_hists = pd.DataFrame(columns=column_labels)
            line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
            line_widths = [1, 1.5, 2, 1, 1.5, 2, 1]
            fig, ax = plt.subplots(figsize=(12, 4))
            df_hists['Kingdom'] = img_arrays[:, 1, :].flatten()
            df_hists['Phylum'] = img_arrays[:, 2, :].flatten()
            df_hists['Class'] = img_arrays[:, 3, :].flatten()
            df_hists['Order'] = img_arrays[:, 4, :].flatten()
            df_hists['Family'] = img_arrays[:, 5, :].flatten()
            df_hists['Genus'] = img_arrays[:, 6, :].flatten()
            df_hists['Species'] = img_arrays[:, 7, :].flatten()
            stds = df_hists.std()
            means = df_hists.mean()
            for i, col_label in enumerate(column_labels):
                df_hists[col_label].plot.density(color="k", alpha=0.9, label=str(col_label), linestyle=line_styles[i],
                                                 linewidth=line_widths[i],
                                                 ax=ax)
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

            # Plot correlations within family (4)

            if correct_first:
                if not all_ps_df.empty and flag_throw:
                    to_throw_indexes = all_ps_df.index.str.contains('|'.join(to_throw.index))
                    all_ps_df.loc[to_throw_indexes, 0] = threshold_p + 0.01

            imgs_p, imgs_s = build_img_from_table(all_ps_df, 0, "s", bact_names_df)

            df_corss = pd.read_pickle("df_corrs.pkl")

            for df_corr_name in enumerate(df_corss.index):
                df_corr_name = df_corr_name[1]
                cell_col = []
                len_name_dfcorr = len(df_corr_name.split(";"))
                cell_row = bact_names_df.iloc[len_name_dfcorr]
                for col_index, value in enumerate(cell_row):
                    df_corr_name_o = convert_original(df_corr_name)
                    if df_corr_name_o in value:
                        # checking that the cell is a leaf- doesnt have any other children
                        if (len_name_dfcorr == 7) or (
                                len_name_dfcorr < 7 and bact_names[len_name_dfcorr + 1, col_index] == '0.0'):
                            cell_col = col_index
                            break
                        else:
                            continue

                imgs_p[len_name_dfcorr, cell_col] = df_corss.loc[df_corr_name, "p"]
                imgs_s[len_name_dfcorr, cell_col] = df_corss.loc[df_corr_name, "scc"]

            # check family test
            mpl.rc('font', family='Times New Roman')

            all_leaves_in_df_corss = df_corss.index
            # taking all the families, if the length of the name is 5 = family
            list_of_families = [';'.join(i.split(';')[:5]) for i in df_corss.index if len(i.split(';')) >= 5]
            uniques = list(set(list_of_families))
            dict_pos = dict()
            dict_neg = dict()
            for f in uniques:

                pos_count = 0
                neg_count = 0

                for leaf in all_leaves_in_df_corss:
                    if f in leaf:
                        p_leaf = df_corss.loc[leaf, "p"]
                        if p_leaf < threshold_p:
                            score = df_corss.loc[leaf, 'scc']
                            if score > 0:
                                pos_count += 1
                            if score < 0:
                                neg_count += 1
                if pos_count == 0 and neg_count == 0:
                    continue
                f = f.split(";")[-1].split("__")[1]
                f = f.rsplit("_", maxsplit=1)[0]
                dict_pos[f] = pos_count
                dict_neg[f] = neg_count

        flag = False
        if not dict_pos and not dict_neg:
            flag = True
            print("miMic did not find significant families.")
        else:

            df_to_plot = pd.DataFrame(index=list(dict_pos.keys()), columns=['Positives', 'Negatives'])
            df_to_plot['Positives'] = list(dict_pos.values())
            df_to_plot['Negatives'] = list(dict_neg.values())
            df_to_plot.index = [i.split(";")[-1] for i in df_to_plot.index]
            cmap_set2 = cm.get_cmap('Set2')

            colors_tab10 = [cmap_set2(i) for i in range(cmap_set2.N)]

            # Function to darken a color
            def darken_color(color, factor=0.9):
                return tuple(min(max(comp * factor, 0), 1) for comp in color)

            # Darken colors from both colormaps
            darkened_colors_tab10 = [darken_color(color) for color in colors_tab10]


            extended_colors = darkened_colors_tab10
            # Create a dictionary to store the color for each family
            family_colors = {}

            # Iterate over unique families and assign colors
            unique_families = df_to_plot.index
            for i, family in enumerate(unique_families):
                # Use modulo to cycle through the extended color list
                rgba_color = extended_colors[i % len(extended_colors)]
                # Convert RGBA color to tuple and store it in the dictionary
                family_colors[family] = tuple(rgba_color)

            # Plot the DataFrame
            ax = df_to_plot.plot(kind="barh", color=['blue', 'red'], figsize=(4, 4))

            # Customize the plot
            num_families = len(df_to_plot.index)
            fig = plt.gcf()
            fig_height = max(5, num_families * 0.5)  # Minimum height of 5 inches, adjust multiplier as needed

            if flag == True or colorful != True:
                family_colors = None
            # Set the size of the current figure
            fig.set_size_inches(8, fig_height)
            plt.xlabel("Number", fontsize=SIZE)
            plt.xticks(fontsize=SIZE)
            plt.yticks(range(len(df_to_plot.index)), df_to_plot.index, fontsize=SIZE)
            if family_colors:
                for label in ax.get_yticklabels():
                    label.set_color(family_colors[label.get_text()])  # Set color based on family name
            plt.tight_layout()
            plt.savefig(f"{directory}/corrs_within_family.png")
            plt.show(block=False)

            # Interactions plot (5)
            build_interactions(bact_names_df, img_arrays, directory, family_colors, threshold_p=threshold_p,
                               THRESHOLD=THRESHOLD_edge)

            # Plot correlations on tree
            if flag == True or colorful != True:
                family_colors = None
            # Correlations on the tree Plot (6)
            creare_tree_view(bact_names, imgs_p, imgs_s, directory, threshold_p, family_colors)

    series_corrs = pd.Series(dict_corrs).to_frame("scc")
    series_ps = pd.Series(dict_ps).to_frame("p")
    df_corrs = pd.concat([series_corrs, series_ps], axis=1)
    df_corrs = df_corrs.groupby(df_corrs.index).mean()
    if correct_first and start_i > 1:
        try:
            df_filtered = df_corrs[~df_corrs.index.str.contains('|'.join(to_throw.index))]
        except:
            df_filtered = df_corrs
        return df_filtered
    else:
        return df_corrs


def calculate_p_value(img_arrays, taxon, tag):
    """
    Calculate p-value of nested GLM.
    :param img_arrays: Ndarray of iMic images (ndarray)
    :param taxon: Taxonomy level (int).
    :param tag: Tag dataframe with a column named "Tag" (dataframe).
    :return: nested GLM p-value (float)
    """
    num_samples = img_arrays.shape[0]
    features = img_arrays[:, :taxon + 1, :].reshape(num_samples, -1)
    if taxon == 0:
        model = sm.OLS(tag, features).fit()
        return model.f_pvalue
    else:
        full_model = sm.OLS(tag, features).fit()
        upper_features = img_arrays[:, :taxon, :].reshape(num_samples, -1)
        upper_model = sm.OLS(tag, upper_features).fit()
        K1 = len(list(upper_model.params.index))
        ALL = len(list(full_model.params.index))
        K2 = ALL - K1
        S_A = upper_model.ssr
        S_B = full_model.ssr
        Z = (S_A / (K1 - 1)) / S_B / (num_samples - K2)
        p_value = 1 - stats.f.cdf(Z, K1 - 1, num_samples - K2)
        return p_value


def apply_nested_anova(samba_output, folder, tag, mode="test", eval="man", threshold_p=0.05):
    """
    Apply apriori nested test (ANOVA- for binary and categorical tags and GLM for continuous).
    :param samba_output: Samba outputs, if you already have them- miMic will read it from the folder you specified,
    else miMic will apply samba and set `samba_output` to None.
    :param folder: Folder where the images are saved (str).
    :param tag: Tag dataframe with a column named "Tag" (dataframe).
    :param mode: Mode of the miMic test - "test" or "plot" (str).
    :param eval: Evaluation method if the tag is binary - "man", if the tag is categorical -
        "category", if the tag is continuous - "corr" (str).
    :return: In "test" mode returns the p-value, in "plot" mode returns a dataframe with the nested p-values
    of each taxonomy level.
    """
    taxonomy_level = ['Anaerobic', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']

    img_arrays, bact_names, tag = load_from_folder(samba_output, folder, tag)
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
            if P < threshold_p:
                taxonomy = taxonomy_level[taxon_lavel]
                print(f"Test succeeded in level {taxonomy}: P = {P}")
                if mode == "test":
                    break
    else:
        # Apply regression on all the flattened tree
        for taxon_lavel in range(img_arrays.shape[1]):
            P = calculate_p_value(img_arrays, taxon_lavel, tag)
            p_vals_df["nested-p"][taxon_lavel + 1] = P
            if P < threshold_p:
                print(f"Test succeeded in level {taxon_lavel + 1}:P = {P}")
                if mode == "test":
                    break

    if mode == 'test':
        return P
    else:
        return p_vals_df


def plot_rp_sp_anova_p(df, mixed, save):
    """
     Plot RP vs SP over the different taxonomy levels and color the background of the plot till the selected taxonomy, based on miMic test.
    :param df: RP and SP dataframe of the post hoc test applied (dataframe).
    :param save: Name of folder to save the plot (str).
    :return: None. Display the RP vs SP .
    """
    SIZE = 15
    mpl.rc('font', family='Times New Roman')
    taxonomy_level = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species', mixed]
    tax_level_chosen = mixed.split("&")[0].strip()
    tax_level_chosen_index = taxonomy_level.index(tax_level_chosen)

    real_sh = df[['RP', 'SP']]

    # Create the first y-axis (horizontal bar plot)
    fig, ax1 = plt.subplots(figsize=(6, 4))  # Adjust the figure size as needed

    real_sh.plot(kind="barh", color=["blue", "red"], ax=ax1)
    ax1.set_xlabel("Number of significants leaves", fontsize=SIZE)  # Adjust the x-axis label
    ax1.set_yticks(np.arange(len(taxonomy_level)))  # Set the y-ticks to match the custom labels
    ax1.set_yticklabels(taxonomy_level, fontsize=SIZE)
    ax1.set_ylabel("Starting Taxonomy", fontsize=SIZE)
    ax1.tick_params(axis="both", labelsize=SIZE)
    ax1.set_xscale('log')

    # Adding background color behind the bars ( from kingdom till the selected taxonomy)
    ymin = -0.5
    ymax = tax_level_chosen_index + 0.5
    ax1.axhspan(ymin, ymax, facecolor='lightgray', zorder=-1)

    # Tax vs RP SP ANOVA p plot (1)
    plt.tight_layout()
    plt.savefig(f"{save}/tax_vs_rp_sp_anova_p.png")
    plt.show(block=False)


def calculate_rsp(df, save):
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
    RP = df["RP"][8]
    SP = df["SP"][8]
    if RP == 0 and SP == 0:
        print("miMic did not find RP on your Dataframe, please check the RP and SP terminal's outputs.")
        return

    for beta in list_beta:
        to_plot["RSP"][beta] = (beta * RP - SP) / (beta * RP + SP)

    # RSP vs beta plot (2)
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


def apply_mimic(folder, tag, eval="man", sis="fdr_bh", correct_first=True, mode="test", save=True, tax=None,
                colorful=True, threshold_p=0.05, THRESHOLD_edge=0.5, rawData=None, taxnomy_group="sub PCA",
                preprocess='True', processed=None, apply_samba=True, samba_output=None):
    """
    Apply the apriori ANOVA test and the post hoc test of miMic.
    :param folder: Folder path of the iMic images (str).
    :param tag: Tag dataframe with a column named "Tag" (dataframe).
    :param eval: Evaluation method if the tag is binary - "man", if the tag is categorical -
        "category", if the tag is continuous - "corr" (str).
    :param sis: Determines whether to apply sister correction. One of ['fdr_bh', 'bonferroni', 'No']. Default is "fdr_bh" (str).
    :param correct_first: Determines whether to apply FDR correction to the starting taxonomy (Boolean).
    :param mode: Mode of the miMic test - "test" or "plot" (str).
    :param save: Determines whether to save the final corrS_df of the miMic test (Boolean).
    :param colorful: Determines whether to apply colorful mode to the miMic test (Boolean).
    :param tax: Starting taxonomy selected in the post hoc test (int).
    :param threshold_p: The threshold for significant value (float).
    :param THRESHOLD: The threshold for having an edge in "interaction" plot (float).
    :param rawData: Dataframe with the raw data (dataframe).
    :param taxnomy_group: The group of the taxonomy (str).["sub PCA", "mean", "sum"], default is "sub PCA".
    :param preprocess: Determines whether to preprocess the data (Boolean). Default is True.
    :param processed: Processed data (dataframe).
    :param apply_samba: whether to apply samba or no. Default is True (Boolean).
    :param samba_output: Samba outputs, if you already have them- miMic will read it from the folder you specified,
    else miMic will apply samba and set `samba_output` to None.
    :return: If the apriori test is not significant, prints that and does not continue to the next step. If the
    apriori test is significant prints that and continues to the post hoc test. Prints the number of RPs found in each
    taxonomy level. At last if the the save variable is True it saves the df_corrs. It returns the selected starting taxonomy in the test mode.
    If the function is in "plot" mode it returns 6 plots.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if mode == "preprocess":
            if preprocess:
                # checking if the rowData is not provided
                if rawData is None:
                    print("Please provide a rawData in format of a csv file.")
                    return

                processed = MIPMLP.preprocess(rawData, taxnomy_group=taxnomy_group)
                print("Preprocessing is done.")
                return processed
            else:
                return

        if mode == 'test':
            if processed is None:
                print("Please provide a processed data.")
                return

            # If you do not have samba output, samba should set to True
            if apply_samba:
                array_of_imgs, bact_names, ordered_df = samba.micro2matrix(processed, folder, save=save)
                samba_output = (array_of_imgs, bact_names, ordered_df)
            else:
                # If you already have samba outputs, we will read them from the folder you specified and set our variable of `samba_output` to None.
                samba_output = None

            print("\nApply nested Anova test")
            p = apply_nested_anova(samba_output, folder, tag, mode=mode, eval=eval, threshold_p=threshold_p)
            if p > threshold_p:
                print(f"Apriori nested ANOVA test is not significant, getting P = {p}.")
                t1 = 'noAnova'

            else:
                print(f"Apriori nested ANOVA test is significant.\n\nTrying postHC miMic test.")
                # Apply post HOC miMic test
                print("Checking on the path starting from 1/2/3 taxonomy levels")
                for t1 in [1, 2, 3]:
                    print(f"\nTaxonomy is {t1}")
                    df_corrs123 = calculate_all_imgs_tag_corr(samba_output, folder, tag, t1, eval=eval,
                                                              sis=sis, correct_first=correct_first, mode=mode,
                                                              threshold_p=threshold_p,
                                                              shuffle=False)
                    df_corrs123 = df_corrs123.dropna()
                    n_significant = (df_corrs123["p"] < threshold_p).sum()
                    print(f"Number of RP: {n_significant}")
                    if n_significant > 0:
                        break

                    if t1 == 3:
                        if n_significant == 0:
                            print(
                                "miMic did not find significant taxa by starting at 1 of the first 3 taxonomy levels.")
                            t1 = 'noAnova'

            print("\nTesting on the leaves:")
            df_corrs_leaves = calculate_all_imgs_tag_corr(samba_output,folder, tag, 0, eval=eval,
                                                         sis=sis, correct_first=correct_first, mode='leaves',
                                                         threshold_p=threshold_p,
                                                         shuffle=False)
            df_corrs_leaves = df_corrs_leaves.dropna()
            n_significant = (df_corrs_leaves["p"] < threshold_p).sum()
            print(f"Number of RP: {n_significant}\n")
            if n_significant == 0:
                print("miMic did not find any significant taxa on the leaves.\n")
                # if we did not find any significant taxa in the leaves and also miMic test did not find anything as well, we will stop here.
                if t1 == 'noAnova':
                    t1 = 'nosignificant'
                    return t1, samba_output

            corrected_list_names = create_list_of_names(df_corrs_leaves.index)
            original_list_names = df_corrs_leaves.index

            # dropping leaves that have sons- not relevant

            for i, corrected_name in zip(original_list_names, corrected_list_names):
                if any(name.startswith(corrected_name) for name in corrected_list_names if name != corrected_name):
                    df_corrs_leaves = df_corrs_leaves.drop(i)

            df_corrs_leaves.index = create_list_of_names(df_corrs_leaves.index)

            if t1 != 'noAnova':
                df_corrs123.index = create_list_of_names(df_corrs123.index)
                df_corrs = pd.concat([df_corrs123, df_corrs_leaves])
                df_corrs = df_corrs[~df_corrs.index.duplicated(keep='first')]
                common_rows = pd.merge(df_corrs_leaves, df_corrs123, how='inner', left_index=True, right_index=True)

                mimic_and_utest = df_corrs[df_corrs.index.isin(common_rows.index)]
                # Subtract the common rows from df_corrs_leaves
                df_corrs_leaves_difference = df_corrs_leaves[~df_corrs_leaves.index.isin(common_rows.index)]
                df_corrs_leaves_difference.to_pickle("u_test_without_mimic.pkl")
                df_corrs.to_pickle("df_corrs.pkl")
                mimic_and_utest.to_pickle("miMic&Utest.pkl")

                # save statistic and p-values df
                if save:
                    df_corrs.to_csv(f"{folder}/df_corrs.csv")
                    df_corrs_leaves_difference.to_csv(f"{folder}/u_test_without_mimic.csv")
                    df_corrs123.to_csv(f"{folder}/just_mimic.csv")
                    mimic_and_utest.to_csv(f"{folder}/miMic&Utest.csv")

            else:
                df_corrs_leaves.to_pickle("u_test_without_mimic.pkl")
                df_corrs_leaves.to_pickle("df_corrs.pkl")

                if save:
                    df_corrs_leaves.to_csv(f"{folder}/u_test_without_mimic.csv")

            return t1, samba_output


        elif mode == "plot":
            # tax= the taxonomy level that miMic test choose to start from
            if tax != 'nosignificant':
                taxonomy_level = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
                if tax != 'noAnova':
                    mixed = taxonomy_level[tax - 1] + ' & leaves'
                else:
                    mixed = 'leaves'

                # Apply apriori nested ANOVA test
                p_vals_df = apply_nested_anova(samba_output,folder, tag, mode=mode, threshold_p=threshold_p)
                num_s_df = pd.DataFrame(index=[1, 2, 3, 4, 5, 6, 7, 8], columns=["RP", "SP"])

                # RP
                bact_name_level = {}
                non_common_count_RP = 0

                # Apply post HOC miMic test
                print("\nRP - post HOC mimic on original and addition \n")
                for t in [1, 2, 3, 4, 5, 6, 7, 8]:

                    if t == 8:
                        print("\nleaves")
                        # check leaves test
                        df_corrs_real = calculate_all_imgs_tag_corr(samba_output,folder, tag, 0, eval=eval,
                                                                    sis=sis, correct_first=correct_first, mode="leaves",
                                                                    threshold_p=threshold_p,
                                                                    shuffle=False)
                        df_corrs_real = df_corrs_real.dropna()
                        n_significant = (df_corrs_real["p"] < threshold_p).sum()
                        num_s_df["RP"][t] = n_significant
                        print(f"Number of RP on leaves: {n_significant}\n")

                        name_real_leaves= {index for index, row in df_corrs_real.iterrows() if row['p'] < threshold_p}

                        # if anova did not find any significant level
                        if tax == 'noAnova':
                            non_common_count_RP = len(set(name_real_leaves))

                        else:
                            non_common_count_RP = len(set(name_real_leaves) | set(bact_name_level[tax]))

                        continue

                    print(f"\nTaxonomy is {t}")
                    df_corrs_real = calculate_all_imgs_tag_corr(samba_output,folder, tag, t, eval=eval,
                                                                sis=sis, correct_first=correct_first, mode="test",
                                                                threshold_p=threshold_p,
                                                                shuffle=False)
                    df_corrs_real = df_corrs_real.dropna()
                    n_significant = (df_corrs_real["p"] < threshold_p).sum()
                    num_s_df["RP"][t] = n_significant
                    print(f"Number of RP: {n_significant}")

                    if t == tax:
                        names_original_real = {index for index, row in df_corrs_real.iterrows() if
                                               row['p'] < threshold_p}
                        bact_name_level[t] = names_original_real

                # SP
                bact_name_level_shuffle = {}
                non_common_count_SP = 0
                print("\nSP - post HOC mimic on original and addition \n")

                for t in [1, 2, 3, 4, 5, 6, 7, 8]:

                    if t == 8:
                        print("\nleaves")
                        # check leaves test
                        df_corrs_shuffled = calculate_all_imgs_tag_corr(samba_output,folder, tag, 0, eval=eval,
                                                                        sis=sis, correct_first=correct_first,
                                                                        mode="leaves", threshold_p=threshold_p,
                                                                        shuffle=True)
                        df_corrs_shuffled = df_corrs_shuffled.dropna()
                        n_significant = (df_corrs_shuffled["p"] < threshold_p).sum()
                        num_s_df["SP"][t] = n_significant
                        print(f"Number of SP on leaves: {n_significant}\n")

                        name_shuffle_leaves = {index for index, row in df_corrs_shuffled.iterrows() if
                                              row['p'] < threshold_p}
                        if tax == 'noAnova':
                            non_common_count_SP = len(set(name_shuffle_leaves))
                        else:
                            non_common_count_SP = len(set(name_shuffle_leaves) | set(bact_name_level_shuffle[tax]))

                        continue

                    print(f"\nTaxonomy is {t}")
                    df_corrs_shuffled = calculate_all_imgs_tag_corr(samba_output,folder, tag, t, eval=eval,
                                                                    sis=sis, correct_first=correct_first, mode="test",
                                                                    threshold_p=threshold_p,
                                                                    shuffle=True)
                    df_corrs_shuffled = df_corrs_shuffled.dropna()
                    n_significant = (df_corrs_shuffled["p"] < threshold_p).sum()
                    num_s_df["SP"][t] = n_significant
                    print(f"Number of SP: {n_significant}")

                    if t == tax:
                        names_addition_shuffle = {index for index, row in df_corrs_shuffled.iterrows() if
                                                  row['p'] < threshold_p}
                        bact_name_level_shuffle[t] = names_addition_shuffle

                num_s_df['RP'][8] = non_common_count_RP
                num_s_df['SP'][8] = non_common_count_SP
                print(f"{mixed}:\nRP: {num_s_df['RP'][8]}\nSP: {num_s_df['SP'][8]}\n")

                # Build a common table for RP, SP, and nested ANOVA p-values
                results_to_plot = pd.concat([num_s_df, p_vals_df], axis=1)
                if not os.path.exists("plots"):
                    # If it doesn't exist, create the directory
                    os.makedirs("plots")
                # Plot RP, SP, ANOVA vs taxonomy
                plot_rp_sp_anova_p(results_to_plot, mixed, save="plots")
                # Plot RSP(beta) vs beta
                calculate_rsp(num_s_df, "plots")
                # Plot inside plots on the taxonomy selected
                calculate_all_imgs_tag_corr(samba_output,folder, tag, tax, eval=eval,
                                            sis=sis, correct_first=correct_first, mode="plot", threshold_p=threshold_p,
                                            THRESHOLD_edge=THRESHOLD_edge,
                                            shuffle=False,
                                            colorful=colorful)
