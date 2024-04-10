import sys

sys.path.append("..")
from datetime import datetime
from processing import make_dirs


def get_time_title(station):
    today = datetime.now()
    today_date = today.strftime("%Y%m%d")
    today_date_hr = today.strftime("%Y%m%d_%H:%M")
    make_dirs.make_dirs(today_date, station)

    return today_date, today_date_hr
