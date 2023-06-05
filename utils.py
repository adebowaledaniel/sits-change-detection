import rasterio
from rasterio import windows

######### Read Train, Validation, and Evaluation ids #########
def read_ids():
    """
    Read grid split ids from file
    """
    with open("data/gridsplit.txt", "r") as f:
        lines = f.readlines()
        Train_ids = eval(lines[0].split(":")[1])
        Val_ids = eval(lines[1].split(":")[1])
        test_ids = eval(lines[2].split(":")[1])
    return Train_ids, Val_ids, test_ids

def loaddata(sits, yraster, idraster, bound):
    with rasterio.open(sits) as src:
        X = src.read(window=windows.from_bounds(*bound, transform=src.transform)).astype("uint16")
    y = rasterio.open(yraster).read(1, window=windows.from_bounds(*bound, transform=rasterio.open(yraster).transform)).astype("uint8")
    ids = rasterio.open(idraster).read(1, window=windows.from_bounds(*bound, transform=rasterio.open(idraster).transform)).astype("uint32")
    return X, y, ids

class standardize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        return (sample - self.mean) / self.std