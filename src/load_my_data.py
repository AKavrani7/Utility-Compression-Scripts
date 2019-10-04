import scipy.io as sio

def load_mat_file(file):
    data = sio.loadmat(file)
    return data