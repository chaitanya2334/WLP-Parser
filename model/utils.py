def to_scalar(var):
    # returns a python float
    f = var.view(-1).data.tolist()[0]

    return f