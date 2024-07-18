import json

import gradio as gr

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
from transformers.utils import logging

logging.set_verbosity_error()

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

    print(f"___DEBUG___/ imp_func( {youtube_chanel_id},{rate},{amount_of_max_videos},{get_all} )/ dict_res:{dict_res}")

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
    for i in pipeline_output['clusters']:
        # res.append(i)
        res[count_for_dict] = i
        # res.append(["______"])
    res_json = json.dumps(res)

    pipeline_output['graph_pyvis'].show_buttons(filter_=['physics'])
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

    print(f"___DEBUG___/ get_pipeline_prediction( {channel_id},{rate},{amount_of_videos} )/ RES:{res}")

    # f = open("networkx-pyvis.html")
    return (res_json, "networkx-pyvis.html")

    # HTML(filename="networkx-pyvis.html")

    # return res


demo = gr.Interface(
    fn=get_pipeline_prediction,
    inputs=[gr.Text(label="channel id", type="text"), gr.Slider(0, 1, step=0.05, value=0.8), gr.Number()],
    outputs=[gr.Text(type="text"), gr.File(file_types=[".html"])]
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
    demo.launch()
