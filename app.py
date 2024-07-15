import gradio as gr
from huggingface_hub import InferenceClient

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
from transformers import pipeline
from transformers.utils import logging
logging.set_verbosity_error()


# !pip install --upgrade google-api-python-client
# from google.colab import userdata






"""### vars"""
# HF_TOKEN_F = userdata.get('HF_TOKEN_F')
# API_KEY = userdata.get('api_key')
import os

HF_TOKEN_F = os.getenv('HF_TOKEN_F')
API_KEY = os.getenv('api_key')








"""### data"""
from apiclient.discovery import build

youtube = build('youtube', 'v3', developerKey=API_KEY)

class Data_gen:

  videos = []

  _youtube = None

  def __init__(self, youtube):
    self._youtube = youtube

  def get_all_videos_from_youtube_chanel_that_is_on_native_lang(self, channel_id):

    res = self._youtube.channels().list(id=channel_id,
                                  part='contentDetails').execute()

    playlist_id = res['items'][0]['contentDetails']['relatedPlaylists']['uploads']

    next_page_token = None

    while True:
        res = self._youtube.playlistItems().list(playlistId=playlist_id,
                                           part='snippet',
                                           maxResults=50,
                                           pageToken=next_page_token).execute()
        self.videos += res['items']
        next_page_token = res.get('nextPageToken')

        if next_page_token is None:
            break

    return self.videos

  def get_titles_of_videos_data(self, amount=500, get_all=False):
    if(get_all):
      r = self.videos.copy()
    else:
      r = self.videos.copy()[:amount]
    for i in range(len(r)):
      r[i] = self.videos[i]['snippet']['title']
    return r

data_gen = Data_gen(
    youtube = youtube
    )





"""
### models

#### translator
"""

pipe_tr = pipeline("translation", model="utrobinmv/t5_translate_en_ru_zh_base_200")

from transformers import AutoModelForSeq2SeqLM

class Translator_to_eng:

  _tr = None

  def __init__(self, tr):
    self._tr = tr

  def translate(self, str_in: str):
    return self._tr(f"translate to en: {str_in}", max_length=100)

translator_to_eng = Translator_to_eng(pipe_tr)



"""#### translating"""



# def get_all_videos_from_youtube_chanel_and_turn_to_eng_lang(channel_id):
#   return list(map(translator_to_eng.translate, get_all_videos_from_youtube_chanel_that_is_on_native_lang(channel_id)))

# def get_titles_of_videos_data_and_turn_to_eng_lang(videos, amount, get_all=False):
#   return list(map(translator_to_eng.translate, get_titles_of_videos_data(videos, amount, get_all=False)))

# videos_both_native_and_eng = []

# videos_both_native_and_eng.append(all_videos_from_yout_chanel_that_is_on_native_lang)

# all_videos_from_yout_chanel_that_is_on_eng_lang = []
# for i in all_videos_from_yout_chanel_that_is_on_native_lang:
#   all_videos_from_yout_chanel_that_is_on_eng_lang.append(translator_to_eng.translate(i))

# videos_both_native_and_eng.append(all_videos_from_yout_chanel_that_is_on_eng_lang)



# title_of_videos_both_native_and_eng = dict()
# title_of_videos_both_native_and_eng['native'] = get_titles_of_videos_data(
#     videos_both_native_and_eng['native']
# )
# title_of_videos_both_native_and_eng['eng'] = get_titles_of_videos_data(
#     videos_both_native_and_eng['eng']
# )



"""#### model( embedding )"""

# pip install thread6
# pip install sentence-transformers


from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import random

from threading import Semaphore, Thread

model_sent_embedding = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings_from_sentences(sentences):
  embeddings = model_sent_embedding.encode(sentences, convert_to_tensor=True)
  return embeddings

def how_similar_sentences(sentences):
  r = get_embeddings_from_sentences(sentences)
  return util.cos_sim(r,r)

def get_most_similar_sentences(sentences, rate_in_mtrx=0.7):
  mtrx = how_similar_sentences(sentences)
  arr_res = []
  # todo debug sentences

  s = Semaphore(1)

  def fill_j(i):

    # #first version
    # arr = []
    # # rand = random.random()
    # for j in range(i+1, len(sentences)):
    #   # print(f"potok with r={rand}, max_eq={max_eq}")
    #   if(i==j):
    #     print("ERROR")
    #   if(mtrx[i][j]>=rate_in_mtrx):
    #     # arr.append([sentences[i],sentences[j]])
    #     arr.append([sentences[i], sentences[j]])
    # # arr.append((sentences[max_index[0]],sentences[max_index[1]]))
    # # arr.sort()
    # # s.acquire()
    # # # arr_res.extend(arr)
    # # arr_res.add(arr)
    # # print(f"potok ext arr_res with: {arr}\n")
    # # s.release()

    # # todo fix array indexes
    # s.acquire()
    # arr_res.extend(arr)
    # print(f"potok ext arr_res with: {arr}\n")
    # s.release()

    #second version
    # arr =
    max_eq = 0
    max_index = [-1, -1]
    # rand = random.random()
    for j in range(i+1, len(sentences)):
      # print(f"potok with r={rand}, max_eq={max_eq}")
      if(i==j):
        print("ERROR")
      if(mtrx[i][j]>=rate_in_mtrx and mtrx[i][j]>max_eq):
        # arr.append([sentences[i],sentences[j]])
        max_eq = mtrx[i][j]
        max_index = [i, j]
    # arr.append((sentences[max_index[0]],sentences[max_index[1]]))
    # arr.sort()
    # s.acquire()
    # # arr_res.extend(arr)
    # arr_res.add(arr)
    # print(f"potok ext arr_res with: {arr}\n")
    # s.release()
    if(max_index[0] != -1 and max_index[1] != -1):
      if(max_index[0]==max_index[1]):
        print("ERROR_2")
      t = [sentences[max_index[0]],sentences[max_index[1]]]
      if(sentences[max_index[0]]==sentences[max_index[1]]):
        print(f"ERROR_3: {sentences[max_index[0]]==sentences[max_index[1]]}, i={max_index[0]}, j={max_index[1]}")
      else:
        s.acquire()
        arr_res.append(t)
        print(f"potok ext arr_res with: {t}\n")
        s.release()

  threads = []
  for i in range(len(sentences)):
    t1 = Thread(target=fill_j, args=(i,), daemon=True)
    threads.append(t1)
    t1.start()

  for t in threads:
    t.join()

  print(f"\n{len(threads)}\n")

  for t in threads:
    print(t.is_alive())

  return arr_res






"""#### summarization"""

# chatbot = pipeline("question-answering",
#                    model="facebook/blenderbot-400M-distill")

chatbot = pipeline("text2text-generation", model="google/flan-t5-large")

def make_one_title(array_of_videos):
  res = []
  for rr in array_of_videos:
    translated = list(map(translator_to_eng.translate, rr))
    for i in range(len(translated)):
      translated[i] = translated[i][0]['translation_text']
    translated = "; ".join(translated)

    ask = f"create title for these words and sentences: {translated}"
    k = chatbot(ask)[0]['generated_text']
    r = []
    r.append(f"name of list: {k}")
    r.extend(rr)
    res.append(r)
  return res



"""### graphs"""

# pip install pyvis

import matplotlib.pyplot as plt
import networkx as nx

from pyvis.network import Network
from IPython.display import display, HTML

def get_graphs_and_cmp_sv(sentences_to_comp, rate):
  graph = nx.Graph()
  graph.add_nodes_from(sentences_to_comp)
  edjes = get_most_similar_sentences(sentences_to_comp, rate)
  graph.add_edges_from(edjes)

  dict_res = {
      "comp_sv": list(nx.connected_components(graph)),
      "amount_of_comp_sv": nx.number_connected_components(graph),
      "graph_nx": graph
  }

  net = Network(notebook=True, cdn_resources='remote')
  net.add_nodes(sentences_to_comp)
  net.add_edges(edjes)

  dict_res['graph_pyvis'] = net

  return dict_res



# pip install community
# pip install python-louvain

import community.community_louvain as lo

def get_clusters(graph_nx):
  partition = lo.best_partition(graph_nx)

  clusters = []


  for cluster_id in set(partition.values()):
      nodes = [nodes for nodes in partition.keys() if partition[nodes] == cluster_id]
      if(len(nodes)>2):
        clusters.append(nodes)
  return clusters



"""### wrappers for models"""

def imp_func(youtube_chanel_id, rate=0.75, amount_of_max_videos=500, get_all=False):
  videos = data_gen.get_all_videos_from_youtube_chanel_that_is_on_native_lang(channel_id=youtube_chanel_id)
  videos_to_comp = data_gen.get_titles_of_videos_data(amount=amount_of_max_videos, get_all=get_all)

  gr_res = get_graphs_and_cmp_sv(videos_to_comp, rate)
  clusters = get_clusters(gr_res['graph_nx'])
  print(clusters)

  # import gc
  # del model_sent_embedding
  # gc.collect()

  dict_res = {}
  dict_res['graph_pyvis'] = gr_res['graph_pyvis']
  dict_res['clusters'] = clusters

  return dict_res








"""# gradio"""

# !pip install plotly==5.22.0

import os
import gradio as gr

# import matplotlib.cm as cm
# import numpy as np

# def get_node_color(val, cmap='RdYlGn'):
#     norm = cm.Normalize()
#     rgba = cm.get_cmap(cmap)(norm(val))
#     r, g, b, _ = rgba
#     color_hex = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
#     return color_hex

def get_pipeline_prediction(channel_id, rate: float, amount_of_videos):

    pipeline_output = imp_func(youtube_chanel_id=channel_id, amount_of_max_videos=amount_of_videos, rate=rate, get_all=False)

    res = []

    from random import randint

    colors = []
    n = len(pipeline_output['clusters'])

    for i in range(n):
      colors.append('#%06X' % randint(0, 0xFFFFFF))

    # #b
    # for node in pipeline_output['graph_pyvis'].get_nodes():
    #   print(node)
    # #/b


    for i in pipeline_output['clusters']:
      res.append(i)
      res.append(["______"])



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


    # f = open("networkx-pyvis.html")
    return (res, "networkx-pyvis.html")



    # HTML(filename="networkx-pyvis.html")

    # return res

demo = gr.Interface(
  fn=get_pipeline_prediction,
  inputs = [gr.Text(label="channel id", type="text"), gr.Slider(0, 1, step=0.05, value=0.8), gr.Number()],
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

#UCdxesVp6Fs7wLpnp1XKkvZg
#UCuXYmUOJSbEH1x88WUV1aMg
#UCuXYmUOJSbEH1x88WUV1aMg

# demo.launch(share=True,debug=True)
if __name__ == "__main__":
    demo.launch()
