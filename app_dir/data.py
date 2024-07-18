from apiclient.discovery import build
import os
import logging

# vars

from app_dir.api_vars import API_KEY

#
"""logger"""
from app_dir.logger import get_logger

log = get_logger("data", logging.INFO)


class DataGen:
    _youtube = None

    def __init__(self, youtube):
        self._youtube = youtube

    def get_all_videos_from_youtube_chanel_that_is_on_native_lang(self, channel_id):
        log.info("start of func")

        videos = []

        res = self._youtube.channels().list(id=channel_id,
                                            part='contentDetails').execute()

        playlist_id = res['items'][0]['contentDetails']['relatedPlaylists']['uploads']

        next_page_token = None

        while True:
            res = self._youtube.playlistItems().list(playlistId=playlist_id,
                                                     part='snippet',
                                                     maxResults=50,
                                                     pageToken=next_page_token).execute()
            videos += res['items']
            next_page_token = res.get('nextPageToken')

            if next_page_token is None:
                break

        return videos

    def get_titles_of_videos_data(self, channel_id, amount=500, get_all=False):
        log.info("start of func")
        if get_all:
            r = self.get_all_videos_from_youtube_chanel_that_is_on_native_lang(channel_id)
        else:
            r = self.get_all_videos_from_youtube_chanel_that_is_on_native_lang(channel_id)[:amount]
        for i in range(len(r)):
            r[i] = r[i]['snippet']['title']
        log.info(f"len(res) = {len(r)}")
        return r


youtube = build('youtube', 'v3', developerKey=API_KEY)

data_gen = DataGen(
    youtube=youtube
)
