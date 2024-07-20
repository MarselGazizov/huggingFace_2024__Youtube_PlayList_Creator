from matplotlib import pyplot as plt

from app_dir.models.emb_model import how_similar_sentences
from threading import Semaphore, Thread

from concurrent.futures import ProcessPoolExecutor

from app_dir.logger import get_logger

log = get_logger("helper")


def get__matrix__hist_of_matrix_nums(sentences):
    mtrx = how_similar_sentences(sentences)

    numbers_in_matrix = []

    s = Semaphore(1)

    def fill_j(i1):

        arr = []
        for j in range(i1 + 1, len(sentences)):
            arr.append(float(mtrx[i1][j]))

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

    fig = plt.figure()
    plt.hist(numbers_in_matrix)

    return mtrx, fig


def get__matrix__hist_of_matrix_nums__in_parallel_mode(sentences):
    mtrx = how_similar_sentences(sentences)

    numbers_in_matrix = []

    s = Semaphore(1)

    def fill_j(i1):
        arr = []
        for j in range(i1 + 1, len(sentences)):
            arr.append(mtrx[i1][j])

        s.acquire()
        numbers_in_matrix.extend(arr)
        s.release()

    amount_of_workers = _get_am_of_workers(len(sentences))
    with ProcessPoolExecutor(max_workers=amount_of_workers) as executor:
        for i in range(len(sentences)):
            executor.submit(fill_j(i))

    fig = plt.figure()
    plt.hist(numbers_in_matrix)

    return mtrx, fig


DEF_RATE_IN_MTRX = 0.7


def get_most_similar_sentences__version_pl_1(sentences, rate_in_mtrx=DEF_RATE_IN_MTRX):
    mtrx = how_similar_sentences(sentences)
    arr_res = []
    # todo debug sentences

    s = Semaphore(1)

    def fill_j(i1):

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

        # second version
        # arr =
        max_eq = 0
        max_index = [-1, -1]
        # rand = random.random()
        for j in range(i1 + 1, len(sentences)):
            # print(f"potok with r={rand}, max_eq={max_eq}")
            if i1 == j:
                log.error("Error_1")
            if mtrx[i1][j] >= rate_in_mtrx and mtrx[i1][j] > max_eq:
                # arr.append([sentences[i],sentences[j]])
                max_eq = mtrx[i1][j]
                max_index = [i1, j]
        # arr.append((sentences[max_index[0]],sentences[max_index[1]]))
        # arr.sort()
        # s.acquire()
        # # arr_res.extend(arr)
        # arr_res.add(arr)
        # print(f"potok ext arr_res with: {arr}\n")
        # s.release()
        if max_index[0] != -1 and max_index[1] != -1:
            if max_index[0] == max_index[1]:
                log.error("Error_2")
            t = [sentences[max_index[0]], sentences[max_index[1]]]
            if sentences[max_index[0]] == sentences[max_index[1]]:
                log.error(
                    f"ERROR_3: {sentences[max_index[0]] == sentences[max_index[1]]}, i={max_index[0]}, j={max_index[1]}")
            else:
                s.acquire()
                arr_res.append(t)
                log.info(f"potok ext arr_res with: {t} \n")
                s.release()

    threads = []
    for i in range(len(sentences)):
        t1 = Thread(target=fill_j, args=(i,), daemon=True)
        threads.append(t1)
        t1.start()

    for t in threads:
        t.join()

    log.info(f"\nlen(threads): {len(threads)}\n")

    # for t in threads:
    #     log.info(f"t.is_alive(): {t.is_alive()}")

    return arr_res


def get_most_similar_sentences__version_3d_1(sentences, rate_in_mtrx=DEF_RATE_IN_MTRX):
    mtrx = how_similar_sentences(sentences)
    arr_res = []
    # todo debug sentences

    s = Semaphore(1)

    def fill_j(i1):

        # first version
        arr = []
        # rand = random.random()
        for j in range(i1 + 1, len(sentences)):
            # print(f"potok with r={rand}, max_eq={max_eq}")
            if i1 == j:
                log.error("Error_1")
            if mtrx[i1][j] >= rate_in_mtrx:
                # arr.append([sentences[i],sentences[j]])
                arr.append([sentences[i1], sentences[j]])
        # arr.append((sentences[max_index[0]],sentences[max_index[1]]))
        # arr.sort()
        # s.acquire()
        # # arr_res.extend(arr)
        # arr_res.add(arr)
        # print(f"potok ext arr_res with: {arr}\n")
        # s.release()

        # todo fix array indexes
        s.acquire()
        arr_res.extend(arr)
        log.debug(f"potok ext arr_res with: {arr}\n")
        s.release()

        # second version
        # # arr =
        # max_eq = 0
        # max_index = [-1, -1]
        # # rand = random.random()
        # for j in range(i1 + 1, len(sentences)):
        #     # print(f"potok with r={rand}, max_eq={max_eq}")
        #     if i1 == j:
        #         log.error("Error_1")
        #     if mtrx[i1][j] >= rate_in_mtrx and mtrx[i1][j] > max_eq:
        #         # arr.append([sentences[i],sentences[j]])
        #         max_eq = mtrx[i1][j]
        #         max_index = [i1, j]
        # # arr.append((sentences[max_index[0]],sentences[max_index[1]]))
        # # arr.sort()
        # # s.acquire()
        # # # arr_res.extend(arr)
        # # arr_res.add(arr)
        # # print(f"potok ext arr_res with: {arr}\n")
        # # s.release()
        # if max_index[0] != -1 and max_index[1] != -1:
        #     if max_index[0] == max_index[1]:
        #         log.error("Error_2")
        #     t = [sentences[max_index[0]], sentences[max_index[1]]]
        #     if sentences[max_index[0]] == sentences[max_index[1]]:
        #         log.error(
        #             f"ERROR_3: {sentences[max_index[0]] == sentences[max_index[1]]}, i={max_index[0]}, j={max_index[1]}")
        #     else:
        #         s.acquire()
        #         arr_res.append(t)
        #         log.info(f"potok ext arr_res with: {t} \n")
        #         s.release()

    threads = []
    for i in range(len(sentences)):
        t1 = Thread(target=fill_j, args=(i,), daemon=True)
        threads.append(t1)
        t1.start()

    for t in threads:
        t.join()

    log.info(f"\nlen(threads): {len(threads)}\n")

    # for t in threads:
    #     log.info(f"t.is_alive(): {t.is_alive()}")

    return arr_res


def _get_am_of_workers(len_sent):
    amount_of_workers = -1
    if len_sent <= 100:
        amount_of_workers = 1
    elif 100 < len_sent <= 400:
        amount_of_workers = 2
    elif 400 < len_sent <= 1000:
        amount_of_workers = 4
    elif 1000 < len_sent:
        amount_of_workers = 6
    return amount_of_workers


def get_most_similar_sentences__version_3d_2_in_parallel(sentences,
                                                         rate_in_mtrx=DEF_RATE_IN_MTRX,
                                                         amount_of_workers=-1):
    len_sent = len(sentences)
    amount_of_workers = _get_am_of_workers(len_sent)

    # fixme edit code
    mtrx = how_similar_sentences(sentences)
    arr_res = []
    # todo debug sentences

    s = Semaphore(1)

    def fill_j(i1):

        arr = []
        for j in range(i1 + 1, len(sentences)):
            if mtrx[i1][j] >= rate_in_mtrx:
                arr.append([sentences[i1], sentences[j]])

        s.acquire()
        arr_res.extend(arr)
        s.release()

    with ProcessPoolExecutor(max_workers=amount_of_workers) as executor:
        for i in range(len(sentences)):
            executor.submit(fill_j(i))

    return arr_res
