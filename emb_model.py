from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import random
from threading import Semaphore, Thread




#model 1:

# model_sent_embedding = SentenceTransformer("all-MiniLM-L6-v2")

# def get_embeddings_from_sentences(sentences):
#   embeddings = model_sent_embedding.encode(sentences, convert_to_tensor=True)
#   return embeddings

#model 2:

# model_sent_embedding = SentenceTransformer("all-MiniLM-L6-v2")

# def get_embeddings_from_sentences(sentences):
#   embeddings = model_sent_embedding.encode(sentences, convert_to_tensor=True)
#   return embeddings

#model 3:

model_sent_embedding = pipeline("feature-extraction", model="google/canine-c")

def get_embeddings_from_sentences(sentences):
  embeddings = model_sent_embedding.encode(sentences, convert_to_tensor=True)
  return embeddings


def how_similar_sentences(sentences):
  r = get_embeddings_from_sentences(sentences)
  # return model.similarity(r,r)
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
