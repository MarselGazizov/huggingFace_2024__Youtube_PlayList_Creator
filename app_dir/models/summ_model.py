from transformers import pipeline

# chatbot = pipeline("question-answering",
#                    model="facebook/blenderbot-400M-distill")

chatbot = pipeline("text2text-generation", model="google/flan-t5-large")

# def make_one_title(array_of_videos):
#   res = []
#   for rr in array_of_videos:
#     translated = list(map(translator_to_eng.translate, rr))
#     for i in range(len(translated)):
#       translated[i] = translated[i][0]['translation_text']
#     translated = "; ".join(translated)

#     ask = f"create title for these words and sentences: {translated}"
#     k = chatbot(ask)[0]['generated_text']
#     r = []
#     r.append(f"name of list: {k}")
#     r.extend(rr)
#     res.append(r)
#   return res
