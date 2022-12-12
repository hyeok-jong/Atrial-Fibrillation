import torch
import numpy as np
import pickle
import torchvision.transforms as transforms

from scipy.io import loadmat


class Randomcrop():
    def __init__(self, length : int):
        self.length = length

    def __call__(self, array : torch.tensor or np.array):
        total_len = array.shape[0]
        start = np.random.randint(low = 0, high = total_len - self.length)
        
        end = start + self.length
        return array[start:end]

class Tensor_and_Norm():
    def __call__(self, array : np.array):
        tensor = torch.FloatTensor(array)
        return torch.nn.functional.normalize(tensor, dim = 0)


class ECG_dataset(torch.utils.data.Dataset):
    def __init__(self, length, pickle_dir):
        self.pickle = pickle.load(open(pickle_dir, 'rb'))
        self.mat_dir = list(self.pickle.keys())
        self.transform = transforms.Compose([
                                            Tensor_and_Norm(),
                                            Randomcrop(length)
                                            ])
    def __len__(self):
        return len(self.pickle.keys())
    
    def __getitem__(self, idx):
        '''
        ecg.shape : batch*length
        '''
        ecg_dir = self.mat_dir[idx]
        ecg = loadmat(ecg_dir)['val'].squeeze()
        ecg = self.transform(ecg)
        cls = torch.FloatTensor([self.pickle[ecg_dir]])

        return ecg, cls
        


# https://discuss.pytorch.org/t/how-to-handle-last-batch-in-lstm-hidden-state/40858