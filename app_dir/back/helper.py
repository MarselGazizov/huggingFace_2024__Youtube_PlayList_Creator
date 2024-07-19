from matplotlib import pyplot as plt

from app_dir.models.emb_model import how_similar_sentences
from threading import Semaphore, Thread


def get__matrix__hist_of_matrix_nums(sentences):
    mtrx = how_similar_sentences(sentences)

    numbers_in_matrix = []

    s = Semaphore(1)

    def fill_j(i1):

        arr = []
        for j in range(i1 + 1, len(sentences)):
            arr.append([mtrx[i1], mtrx[j]])

        s.acquire()
        numbers_in_matrix.extend(arr)
        s.release()

    threads = []
    for i in range(len(sentences)):
        t1 = Thread(target=fill_j, args=(i,), daemon=True)
        threads.append(t1)
        t1.start()

    for t in threads:
        t.join()

    hist = plt.hist(numbers_in_matrix)

    return mtrx, hist
