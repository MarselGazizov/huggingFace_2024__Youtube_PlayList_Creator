from concurrent.futures import ProcessPoolExecutor

from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import random
from threading import Semaphore, Thread
import logging

from transformers import pipeline



"""logger"""
from app_dir.logger import get_logger

log = get_logger("emb_model")

########################################

# model 1:

model_sent_embedding = SentenceTransformer("all-MiniLM-L6-v2")


def get_embeddings_from_sentences(sentences):
    embeddings = model_sent_embedding.encode(sentences, convert_to_tensor=True)
    return embeddings


# model 2:

# model_sent_embedding = SentenceTransformer("all-MiniLM-L6-v2")

# def get_embeddings_from_sentences(sentences):
#   embeddings = model_sent_embedding.encode(sentences, convert_to_tensor=True)
#   return embeddings

# model 3:

# model_sent_embedding = pipeline("feature-extraction", model="google/canine-c")
# from transformers import CanineTokenizer, CanineModel
# model = CanineModel.from_pretrained('google/canine-c')
# tokenizer = CanineTokenizer.from_pretrained('google/canine-c')

# def get_embeddings_from_sentences(sentences):
#  encoding = tokenizer(sentences, padding="longest", truncation=True, return_tensors="pt")

#  outputs = model(**encoding) # forward pass

#  pooled_output = outputs.pooler_output
#  sequence_output = outputs.last_hidden_state
#  return pooled_output


########################################


def how_similar_sentences(sentences):
    r = get_embeddings_from_sentences(sentences)
    # return model.similarity(r,r)
    mtrx = util.cos_sim(r, r)
    # log.info(f"\n mtrx: {mtrx} \n")
    return mtrx


