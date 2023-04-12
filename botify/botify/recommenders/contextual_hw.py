from .toppop import TopPop
from .recommender import Recommender
import random


class ContextualHW(Recommender):
    """
    Recommend tracks closest to the previous one.
    Fall back to the random recommender if no
    recommendations found for the track.
    """

    def __init__(self, tracks_redis, catalog, top_tracks, users_cache, users_tracks_for_recs, rec_n):
        self.tracks_redis = tracks_redis
        self.fallback = TopPop(tracks_redis, top_tracks)
        self.catalog = catalog
        self.top_tracks = top_tracks
        self.users_cache = users_cache
        self.users_tracks_for_recs = users_tracks_for_recs
        self.rec_n = rec_n

    def track_cache(self, user_cache: list) -> list:
        return [item[0] for item in user_cache]

    def new_track_for_rec(self, user_cache, user_tracks_for_recs):
        return max(user_cache, key=lambda item: item[-1] if item[0] not in user_tracks_for_recs else 0)

    # TODO Seminar 5 step 1: Implement contextual recommender based on NN predictions
    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        previous_track = self.tracks_redis.get(prev_track)
        if previous_track is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        if not self.users_tracks_for_recs[user]:
            first_track = prev_track
            self.users_tracks_for_recs[user].append(first_track)
        else:
            first_track = self.users_tracks_for_recs[user][-1]

        first_track = self.tracks_redis.get(first_track)

        self.users_cache[user].append((prev_track, prev_track_time))

        first_track = self.catalog.from_bytes(first_track)
        recommendations = first_track.recommendations
        if recommendations is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        available_recommendations = [item for item in list(recommendations[:self.rec_n])
                                     if item not in self.track_cache(self.users_cache[user])]

        if not available_recommendations:
            new_track_for_rec, time = self.new_track_for_rec(self.users_cache[user], self.users_tracks_for_recs[user])
            if time > 0.5:
                self.users_tracks_for_recs[user].append(new_track_for_rec)
                return self.recommend_next(user, prev_track, prev_track_time)
            else:
                return self.fallback.recommend_next(user, prev_track, prev_track_time)

        else:
            recommended_track = random.choice(available_recommendations)
            return recommended_track

