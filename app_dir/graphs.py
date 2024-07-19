"""### graphs"""

# pip install pyvis

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

import networkx as nx
from networkx import Graph

from pyvis.network import Network

from app_dir.models.emb_model import get_most_similar_sentences__version_pl_1
from app_dir.models.emb_model import get_most_similar_sentences__version_3d_1

import enum


class GraphMode(enum.Enum):
    mode_3d = 2
    mode_planarn_1 = 1


def get_nx_graph_and_cmps_sv(sentences_to_comp, rate, type_of_graph: GraphMode):
    graph = nx.Graph()
    graph.add_nodes_from(sentences_to_comp)

    if type_of_graph == GraphMode.mode_planarn_1:
        edges = get_most_similar_sentences__version_pl_1(sentences_to_comp, rate)
    if type_of_graph == GraphMode.mode_3d:
        edges = get_most_similar_sentences__version_3d_1(sentences_to_comp, rate)

    graph.add_edges_from(edges)

    dict_res = {
        "comps_sv": list(nx.connected_components(graph)),
        "graph_nx": graph
    }

    # dict_res['graph_pyvis'] = net

    return dict_res


def get_pyvis_graph_from_nx_graph(nx_graph):
    net = Network(notebook=True,
                  cdn_resources='remote',
                  height="750px",
                  width="100%",
                  # bgcolor="#222222",
                  # font_color="white",
                  select_menu=True,
                  filter_menu=True)
    net.from_nx(nx_graph)
    return net


# pip install community
# pip install python-louvain

import community.community_louvain as lo


def get_clusters_and_colorized_graph__version_with_colors(graph_nx: Graph):
    partition = lo.best_partition(graph_nx)

    amount_of_colors = len(set(partition.values()))

    def get_n_colors(n):
        cmap = plt.cm.get_cmap('Oranges')
        return [mcolors.to_hex(cmap(i)) for i in np.linspace(0, 1, n)]

    colors_range = get_n_colors(amount_of_colors)

    cl_id__color__dict = dict()

    color_map = []

    clusters = []
    color_count = 0
    for cluster_id in set(partition.values()):
        cl_id__color__dict[cluster_id] = colors_range[color_count]
        color_count += 1

        nodes = [nodes for nodes in partition.keys() if partition[nodes] == cluster_id]
        # min amount of nodes in cluster is 2
        if len(nodes) > 1:
            clusters.append(nodes)

    for node in graph_nx:
        if partition[node] in cl_id__color__dict:
            # cluster_id = partition[node]
            # color = cl_id__color__dict[cluster_id]
            # color_map.append(color)
            graph_nx.nodes[node]['color'] = cl_id__color__dict[partition[node]]
        else:
            # color_map.append("fff5eb")
            graph_nx.nodes[node]['color'] = "#ffffff"

    # nx.draw(graph_nx, node_color=color_map, with_labels=True)
    return clusters, graph_nx


def get_clusters_and_colorized_graph__version_with_groups(graph_nx: Graph):
    partition = lo.best_partition(graph_nx)
    cl_id__group_number__dict = dict()

    clusters = []
    group_count = 0
    for cluster_id in set(partition.values()):
        cl_id__group_number__dict[cluster_id] = group_count
        group_count += 1

        nodes = [nodes for nodes in partition.keys() if partition[nodes] == cluster_id]
        # min amount of nodes in cluster is 2
        if len(nodes) > 1:
            clusters.append(nodes)

    for node in graph_nx:
        neighbours = graph_nx.neighbors(node)
        # if partition[node] in cl_id__group_number__dict:
        #     l = cl_id__group_number__dict[partition[node]]
        # else:
        #     l = -1
        l = cl_id__group_number__dict[partition[node]]
        neighbours2 = [n for n in neighbours]
        if len(neighbours2) == 0:
            l = -1

        graph_nx.nodes[node]['group'] = l
        graph_nx.nodes[node]['title'] = f"group: {l}\n"
        s = '\n'.join(map(str, neighbours2))
        graph_nx.nodes[node]['title'] += f"neighbours: {s}\n"

    return clusters, graph_nx
