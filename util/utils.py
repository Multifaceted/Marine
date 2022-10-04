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

def interpolate(df, name_ls, method, **kwargs):
    assert method in ["linear", "slinear", "cubic", "spline", "cubicspline", "polynomial"]

    if method == "polynomial":
        assert type(kwargs["order"]) == type(0)
    for name in name_ls:
        df[name] = df[name].interpolate(method=method, **kwargs)

    return df
