import scipy.io as spio
import os

# loading the data file 
def retrieve_data(file_name, directory):
    os.chdir(directory)
    Data = spio.loadmat(file_name)

    return Data

