import requests
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import chain
from utils import convert_wind_to_events
import io


def get_air_quality_data():
    # download multivarite air quality time series data and process it into hetergeneous squences

    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv"
    r = requests.get(data_url)
    data_pd = pd.read_csv(io.StringIO(r.content.decode('utf-8')))[24:].interpolate(axis=0)
    years = np.array(data_pd['year'])
    months = np.array(data_pd['month'])
    days = np.array(data_pd['day'])
    hours = np.array(data_pd['hour'])
    timestamps = np.array([datetime(y, m, d, h) for y, m, d, h in zip(years, months, days, hours)])
    data_pd = data_pd.set_index(timestamps)
    # up sampling and interpolate for event extraction
    wind_speed = data_pd['Iws']
    up_index = pd.date_range(start=wind_speed.index[0], end=wind_speed.index[-1], freq='5T')
    wind_speed_up = wind_speed.reindex(up_index).interpolate()
    rain = data_pd['Ir']
    up_index = pd.date_range(start=rain.index[0], end=rain.index[-1], freq='5T')
    rain_up = rain.reindex(up_index).interpolate()
    # event extraction
    wind_start, wind_drop = convert_wind_to_events(wind_speed_up, top_C=50, start_C=10)
    wind_start_events = [('wind_start', wind_speed_up.index[i]) for i in wind_start]
    wind_drop_events = [('wind_drop', wind_speed_up.index[i]) for i in wind_drop]
    rain_start, rain_drop = convert_wind_to_events(rain_up, top_C=1, start_C=0.5)
    rain_start_events = [('rain_start', rain_up.index[i]) for i in rain_start]
    # sorting event based on time
    events = sorted(chain(wind_start_events, wind_drop_events,
                          rain_start_events), key=lambda x: x[1])
    data_dict = {}
    time_series = data_pd[['pm2.5', 'DEWP', 'TEMP', 'PRES']]
    data_dict['time_series'] = time_series
    data_dict['event_sequence'] = events
    num_event_types = 3
    return data_dict, num_event_types
