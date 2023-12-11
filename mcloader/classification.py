import torch
from torch.utils.data import Dataset
import mc
import io 

from PIL import Image


class ClassificationDataset(Dataset):
    """Dataset for classification.
    """

    def __init__(self, split='train', pipeline=None):
        self.root = f'/mnt/petrelfs/share/images/{split}'
        list_file = f'/mnt/petrelfs/share/images/meta/{split}.txt'

        self.metas = []
        with open(list_file) as f:
            for line in f.readlines():
                line = line.strip().split(' ')
                assert len(line) == 2
                self.metas.append(line)

        self.initialized = False
        self.pipeline = pipeline

    def __len__(self):
        return len(self.metas)

    def _init_tcs_client(self):
        if not self.initialized:
            client_config_file = "/mnt/petrelfs/shenxuyang/mc.conf"
            self.tcsclient = mc.TcsClient.GetInstance('/mnt/lustre/share/tcs/pcs_server_list.conf', client_config_file)
            self.initialized = True

    def __getitem__(self, index):
        filename = self.root + '/' + self.metas[index][0]
        cls = int(self.metas[index][1])

        # tcs_server
        self._init_tcs_client()
        value = mc.pyvector()
        self.tcsclient.Get(filename, value)
        value_buf = mc.ConvertBuffer(value)
        buff = io.BytesIO(value_buf)
        with Image.open(buff) as img:
            img = img.convert('RGB')
        
        if self.pipeline is not None:
            img = self.pipeline(img)

        return img, cls