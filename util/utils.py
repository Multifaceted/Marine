def round_time(x):
    minute = int(round(x.minute / 30)*30)
    if minute == 60:
        if x.hour == 23:
            try:
                x = x.replace(minute=0, hour=0, day=x.day+1)
            except:
                try:
                    x = x.replace(minute=0, hour=0, day=1, month=x.month+1)
                except:
                    x = x.replace(minute=0, hour=0, day=1, month=1, year=x.year+1)
                
        else:
            x = x.replace(minute=0, hour=x.hour+1)
    else:
        x = x.replace(minute=minute)
    x = x.replace(second=0)
    return x