import io
import json
from io import StringIO
from typing import Callable

import pandas as pd

from app_dir.back.helper import get__matrix__hist_of_matrix_nums
from app_dir.data import data_gen
from app_dir.graphs import get_nx_graph_and_cmps_sv, get_clusters_and_colorized_graph__version_with_groups, GraphMode
from app_dir.logger import get_logger

log = get_logger("app_logic")


def get_matrix_and_hist_plot(channel_id, amount_of_videos):
    videos_to_comp = data_gen.get_titles_of_videos_data(channel_id=channel_id,
                                                        amount=amount_of_videos)
    return get__matrix__hist_of_matrix_nums(videos_to_comp)


"""### wrappers for models"""

"""
:returns dict{
    graph_nx: networkx graph,
    # graph_pyvis: pyvis graph,
    clusters: clusters
}
"""


def imp_func(youtube_chanel_id, rate=0.75, amount_of_max_videos=500, get_all=False):
    # videos = data_gen.get_all_videos_from_youtube_chanel_that_is_on_native_lang(channel_id=youtube_chanel_id)
    videos_to_comp = data_gen.get_titles_of_videos_data(channel_id=youtube_chanel_id, amount=amount_of_max_videos,
                                                        get_all=get_all)

    nx_graph_and_cmps_sv = get_nx_graph_and_cmps_sv(videos_to_comp, rate, type_of_graph=GraphMode.mode_3d)
    (clusters, colorized_graph) = get_clusters_and_colorized_graph__version_with_groups(
        nx_graph_and_cmps_sv['graph_nx'])
    # nx_graph_and_cmps_sv['graph_nx'] = colorized_graph

    # print(clusters)

    # log.debug(f"printing colorized graph")
    # nx.draw(colorized_graph)
    # log.debug(f"nodes of col_gr: {colorized_graph.nodes}")

    # import gc
    # del model_sent_embedding
    # gc.collect()

    dict_res = {}
    dict_res['graph_nx'] = colorized_graph
    # dict_res['graph_pyvis'] = nx_graph_and_cmps_sv['graph_pyvis']
    dict_res['clusters'] = clusters

    def get_beaut_str_2d_arr(arr):
        str_res = ""
        count = 1
        for i in arr:
            str_res += f"{count} = {i}\n"
            count += 1
        return str_res

    # logging.debug(f"\nimp_func( {youtube_chanel_id},{rate},{amount_of_max_videos},{get_all} )")
    # log.debug(f"\ndict_res:{dict_res}\n")
    log.info(f"dict_res:"
             # f"graph_pyvis: {dict_res['graph_pyvis'].__repr__()}\n"
             f"clusters: {get_beaut_str_2d_arr(dict_res['clusters'])}")

    return dict_res


def get_pipeline_prediction(channel_id, rate: float, amount_of_videos):
    res_of_app = imp_func(youtube_chanel_id=channel_id, amount_of_max_videos=amount_of_videos, rate=rate,
                          get_all=False)

    res = dict()

    from random import randint

    colors = []
    n = len(res_of_app['clusters'])

    for i in range(n):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    # #b
    # for node in res_of_app['graph_pyvis'].get_nodes():
    #   print(node)
    # #/b

    count_for_dict = 1
    clusters_in_series_form = dict()
    for i in res_of_app['clusters']:
        example_node = i[0]
        group_name = res_of_app['graph_nx'].nodes[example_node]['group']
        # res.append(i)
        # todo
        name = "just_name_"
        name = name + str(group_name)
        clusters_in_series_form[name] = pd.Series(i)
        count_for_dict += 1
        # res[count_for_dict] = i
        # res.append(["______"])
    if len(clusters_in_series_form) != 0:
        df_clusters = pd.concat(clusters_in_series_form, axis=1)
    else:
        df_clusters = pd.DataFrame()
    df_clusters = df_clusters.fillna("ꚙ")

    # res_json = json.dumps(res)

    def save_json(json_text, output_filename):
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            json.dump(json_text, outfile, ensure_ascii=False)

    # save_json(df_clusters, "clusters.json")

    # todo delete pyvis vars
    from pyvis.network import Network
    # net = Network()
    #
    # net.from_nx(res_of_app['graph_nx'])

    from app_dir.graphs import get_pyvis_graph_from_nx_graph

    pyvis_graph = get_pyvis_graph_from_nx_graph(res_of_app['graph_nx'])

    # node_colors = nx.get_node_attributes(res_of_app['graph_pyvis'], 'color')
    # log.debug(f"node_colors: {node_colors}")

    # Draw the graph with node colors
    # pos = nx.spring_layout(res_of_app['graph_pyvis'])
    # nx.draw_networkx(res_of_app['graph_pyvis'], pos, node_color=list(node_colors.values()), with_labels=True)
    # plt.show()

    pyvis_graph.show_buttons(filter_=['physics', 'nodes'])
    #     options = """
    # const options = {
    #   "physics": {
    #     "barnesHut": {
    #       "gravitationalConstant": -27600
    #     },
    #     "minVelocity": 0.75
    #   }
    # }
    #     """
    #     pyvis_graph.set_options(options)

    # pyvis_graph.save_graph("networkx-pyvis.html")

    name_of_file_with_pyvis_graph = "networkx-pyvis.html"
    html_str = pyvis_graph.generate_html()
    with open(name_of_file_with_pyvis_graph, "w", encoding="utf-8") as f:
        f.write(html_str)

    # log.debug(f"html: {html}")
    #
    # f = open("networkx-pyvis.html")

    # # colorizing graph
    # node_degrees = dict(gr_res['graph_pyvis'].degrees())

    # for node_id, degree in node_degrees.items():
    #   node_degrees[node_id] = get_node_color(degree)

    # for node_id, attrs in gr_res['graph_pyvis'].nodes(data=True):
    #   attrs['color'] = node_degrees[node_id]
    # # /colorizing graph

    # gr_nx = gr_res['graph_nx']
    # import plotly.express as px

    # df = px.data.gapminder().query("country=='Canada'")
    # fig = px.line(df, x="year", y="lifeExp", title='Life expectancy in Canada')
    # pl_html = plotly.io.to_html(fig)
    # with open("pl.html") as f2:
    #   f2.write(pl_html)

    # logging.debug(f"\n{res}\n")

    # f = open("networkx-pyvis.html")
    # todo it was changed from gr.df
    return df_clusters, name_of_file_with_pyvis_graph
