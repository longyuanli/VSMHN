def convert_wind_to_events(event_series, top_C, start_C):
    flag = False
    event_start = []
    for i in range(len(event_series)):
        if not flag:
            if event_series[i] > top_C:
                event_start.append(i)
                flag = True
        else:
            if event_series[i] < top_C:
                flag = False
    event_top = []
    for i in range(len(event_start)):
        while (event_start[i]-1 >= 0) and (event_series[event_start[i]-1] < event_series[event_start[i]]) and (event_series[event_start[i]] > start_C):
            event_start[i] = event_start[i]-1
    event_top = event_start[:]
    for i in range(len(event_top)):
        while (event_top[i]+1 < len(event_series)) and (event_series[event_top[i]+1] > event_series[event_top[i]]):
            event_top[i] = event_top[i]+1
    return event_start, event_top
