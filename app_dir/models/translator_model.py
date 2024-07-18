# pipe_tr = pipeline("translation", model="utrobinmv/t5_translate_en_ru_zh_base_200")

# from transformers import AutoModelForSeq2SeqLM

# class Translator_to_eng:

#   _tr = None

#   def __init__(self, tr):
#     self._tr = tr

#   def translate(self, str_in: str):
#     return self._tr(f"translate to en: {str_in}", max_length=100)

# translator_to_eng = Translator_to_eng(pipe_tr)


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
