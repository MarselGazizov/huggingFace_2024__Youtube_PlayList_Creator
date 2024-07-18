import json

import gradio as gr
import pandas as pd
import logging

"""logger"""
from app_dir.logger import get_logger
logging.basicConfig()
log = get_logger("app")

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
# client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")


# def respond(
#     message,
#     history: list[tuple[str, str]],
#     system_message,
#     max_tokens,
#     temperature,
#     top_p,
# ):
#     messages = [{"role": "system", "content": system_message}]

#     for val in history:
#         if val[0]:
#             messages.append({"role": "user", "content": val[0]})
#         if val[1]:
#             messages.append({"role": "assistant", "content": val[1]})

#     messages.append({"role": "user", "content": message})

#     response = ""

#     for message in client.chat_completion(
#         messages,
#         max_tokens=max_tokens,
#         stream=True,
#         temperature=temperature,
#         top_p=top_p,
#     ):
#         token = message.choices[0].delta.content

#         response += token
#         yield response

# """
# For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
# """
# demo = gr.ChatInterface(
#     respond,
#     additional_inputs=[
#         gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
#         gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
#         gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
#         gr.Slider(
#             minimum=0.1,
#             maximum=1.0,
#             value=0.95,
#             step=0.05,
#             label="Top-p (nucleus sampling)",
#         ),
#     ],
# )


# if __name__ == "__main__":
#     demo.launch()


#######################


### libraries
# transformers gradio
# google-api-python-client
# thread6
# sentence-transformers
# pyvis
# community
# python-louvain
# plotly==5.22.0


# !pip install -q -U transformers gradio
from transformers.utils import logging as tr_logging

tr_logging.set_verbosity_error()

# !pip install --upgrade google-api-python-client
# from google.colab import userdata


"""### vars"""
# HF_TOKEN_F = userdata.get('HF_TOKEN_F')
# API_KEY = userdata.get('api_key')


"""### data"""
from app_dir.data import data_gen

"""
### models
"""

"""
#### translator
"""

"""#### model( embedding )"""

# pip install thread6
# pip install sentence-transformers


"""#### summarization"""

"""### graphs"""
from app_dir.graphs import get_graphs_and_cmp_sv
from app_dir.graphs import get_clusters

"""### wrappers for models"""


def imp_func(youtube_chanel_id, rate=0.75, amount_of_max_videos=500, get_all=False):
    # videos = data_gen.get_all_videos_from_youtube_chanel_that_is_on_native_lang(channel_id=youtube_chanel_id)
    videos_to_comp = data_gen.get_titles_of_videos_data(channel_id=youtube_chanel_id, amount=amount_of_max_videos,
                                                        get_all=get_all)

    gr_res = get_graphs_and_cmp_sv(videos_to_comp, rate)
    clusters = get_clusters(gr_res['graph_nx'])
    # print(clusters)

    # import gc
    # del model_sent_embedding
    # gc.collect()

    dict_res = {}
    dict_res['graph_pyvis'] = gr_res['graph_pyvis']
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
             f"graph_pyvis: {dict_res['graph_pyvis'].__repr__()}\n"
             f"clusters: {get_beaut_str_2d_arr(dict_res['clusters'])}")

    return dict_res


"""# gradio"""


# !pip install plotly==5.22.0


# import matplotlib.cm as cm
# import numpy as np

# def get_node_color(val, cmap='RdYlGn'):
#     norm = cm.Normalize()
#     rgba = cm.get_cmap(cmap)(norm(val))
#     r, g, b, _ = rgba
#     color_hex = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
#     return color_hex

def get_pipeline_prediction(channel_id, rate: float, amount_of_videos):
    pipeline_output = imp_func(youtube_chanel_id=channel_id, amount_of_max_videos=amount_of_videos, rate=rate,
                               get_all=False)

    res = dict()

    from random import randint

    colors = []
    n = len(pipeline_output['clusters'])

    for i in range(n):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    # #b
    # for node in pipeline_output['graph_pyvis'].get_nodes():
    #   print(node)
    # #/b

    count_for_dict = 1
    clusters_in_series_form = dict()
    for i in pipeline_output['clusters']:
        # res.append(i)
        # todo
        name = "just_name_"
        clusters_in_series_form[name + str(count_for_dict)] = pd.Series(i)
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

    pipeline_output['graph_pyvis'].show_buttons(filter_=['physics'])
    options = """
    const options = {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -28050
        },
        "minVelocity": 0.75
      }
    }
    """
    pipeline_output['graph_pyvis'].set_options(options)
    pipeline_output['graph_pyvis'].save_graph("networkx-pyvis.html")
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
    return (gr.DataFrame(df_clusters), "networkx-pyvis.html")

    # HTML(filename="networkx-pyvis.html")

    # return res


demo = gr.Interface(
    fn=get_pipeline_prediction,
    inputs=[gr.Text(label="channel id", type="text"), gr.Slider(0, 1, step=0.05, value=0.8), gr.Number()],
    outputs=[gr.DataFrame(), gr.File(file_types=[".html"])]
)

# demo = gr.Blocks()
# with demo:
#     inp = gr.Textbox(placeholder="Enter text.")
#     scroll_btn = gr.Button("Scroll")
#     no_scroll_btn = gr.Button("No Scroll")
#     big_block = gr.HTML("""
#     <div style='height: 800px; width: 100px; background-color: pink;'></div>
#     """)
#     out = gr.Textbox()

#     scroll_btn.click(lambda x: x,
#                inputs=inp,
#                outputs=out,
#                 scroll_to_output=True)
#     no_scroll_btn.click(lambda x: x,
#                inputs=inp,
#                outputs=out)

# UCdxesVp6Fs7wLpnp1XKkvZg
# UCuXYmUOJSbEH1x88WUV1aMg
# UCuXYmUOJSbEH1x88WUV1aMg

# demo.launch(share=True,debug=True)
if __name__ == "__main__":
    demo.launch(debug=True)
