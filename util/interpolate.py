def interpolate(df, name_ls, method, **kwargs):
    assert method in ["linear", "slinear", "cubic", "spline", "cubicspline", "polynomial"]

    if method == "polynomial":
        assert type(kwargs["order"]) == type(0)
    for name in name_ls:
        df[name] = df[name].interpolate(method=method, **kwargs)

    return df
