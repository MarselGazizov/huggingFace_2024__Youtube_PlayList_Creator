from apiclient.discovery import build
import os
import logging

# vars

from vars import API_KEY


#


class DataGen:
    videos = []

    _youtube = None

    def __init__(self, youtube):
        self._youtube = youtube

    def get_all_videos_from_youtube_chanel_that_is_on_native_lang(self, channel_id):
        logging.info("start of func")

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
        logging.info("start of func")
        if (get_all):
            r = self.videos.copy()
        else:
            r = self.videos.copy()[:amount]
        for i in range(len(r)):
            r[i] = self.videos[i]['snippet']['title']
        logging.info(f"len(res) = {len(r)}")
        return r


youtube = build('youtube', 'v3', developerKey=API_KEY)

data_gen = DataGen(
    youtube=youtube
)
