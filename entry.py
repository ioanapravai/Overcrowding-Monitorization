import datetime


class Entry:
    def __init__(self, camera_time, people_count):

        self.camera_time = camera_time
        self.weekday = camera_time.weekday()
        self.people_count = people_count

