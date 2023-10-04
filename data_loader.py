import torch
import folders


class DataLoader(object):
    """Dataset class for IQA databases"""

    def __init__(self, dataset, img_indx, batch_size=1, istrain=True):

        self.batch_size = batch_size
        self.istrain = istrain


        if dataset == 'live':
            self.data = folders.LIVEFolder(index=img_indx, train=self.istrain)
        elif dataset == 'csiq':
            self.data = folders.CSIQFolder(index=img_indx, train=self.istrain)
        else:
            print("dataloader.py: dataset not find")
            exit()

    def get_data(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size,
                                                     shuffle=True, pin_memory=True, num_workers=0)
        else:
            dataloader = torch.utils.data.DataLoader(self.data, batch_size=1, shuffle=False)
        return dataloader