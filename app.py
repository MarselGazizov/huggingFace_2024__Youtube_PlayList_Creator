import json

import gradio as gr
import networkx as nx
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
from app_dir.graphs import get_nx_graph_and_cmps_sv, GraphMode
from app_dir.graphs import get_clusters_and_colorized_graph__version_with_groups




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



    # HTML(filename="networkx-pyvis.html")

    # return res


from app_dir.gradio_front import demo

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
