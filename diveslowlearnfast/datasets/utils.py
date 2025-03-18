
def get_sample(dataset):
    result = next(iter(dataset))
    return result[0], result[1]