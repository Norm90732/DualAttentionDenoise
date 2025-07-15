import torchxrayvision as xrv
import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import GaussianNoise
import torchvision.transforms.v2 as transforms_v2

dataset = xrv.datasets.NIH_Dataset(
    imgpath="/blue/uf-dsi/normansmith/projects/TVAttention-Autoencoder/data/images",
    csvpath="/blue/uf-dsi/normansmith/projects/TVAttention-Autoencoder/data/Data_Entry_2017.csv",
    unique_patients=True,
    views=['PA']
)

transformStruct = transforms_v2.Compose([
                transforms_v2.ToImage(),
                transforms_v2.Resize((256,256))
            ])

class DataLoaderNoisyClean(Dataset):
    def __init__(self, dataset,transform=None,noiseParam:tuple[float,float,bool] = (0.0,0.05,True)): #0.05 noise in first trial, #0.1 noise in second trial, #0.15 noise in third trial
        self.dataset = dataset
        mean,sigma,clip = noiseParam
        self.noiseStructure = GaussianNoise(mean=mean,sigma=sigma,clip=clip)
        #Update transforms if overfitting
        self.transform = transform
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        cleanImg = self.dataset[idx]['img'] #For each image, get clean image
        cleanImg = torch.from_numpy(cleanImg).float() #Convert to  Torch Tensor
        #Normalize
        if cleanImg.min() < 0 or cleanImg.max() > 1:
            cleanImg = (cleanImg - cleanImg.min()) / (cleanImg.max() - cleanImg.min())
        # add Gaussian Noise to Noise Image
        if self.transform is not None:
            cleanImg = self.transform(cleanImg)
        noiseImg = self.noiseStructure(cleanImg)
        #Return pair of images
        return noiseImg, cleanImg


#processedDataSet = DataLoaderNoisyClean(dataset, transform=transformStruct)
