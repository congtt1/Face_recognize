import os
from random import shuffle
import torch
import cv2
import numpy as np
from backbones import get_model
from tqdm import tqdm
from PIL import Image
import torchvision
from torchvision import transforms
import torch.utils.data as data
import torchvision.transforms.functional as F
class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, max_wh - w -hp, max_wh - h - vp)
		return F.pad(image, padding, 255, 'constant')

class imagesfromList(data.Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform
    
    def __getitem__(self,index):
        img_path = self.images[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)

class ImagefromData(data.Dataset):
    def __init__(self, images, transform = None):
        self.images = images
        self.transform = transform

    def __getitem__(self, index):
        img = self.images[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img
    def __len__(self):
        return len(self.images)
        
class ArcFace(object):
    def __init__(self, network, weight, device='cpu'):
        self.network = network
        self.model = get_model(network, fp16=False)
        self.model.load_state_dict(torch.load(weight, map_location = torch.device(device)))
        self.model.eval()
        self.transform = transforms.Compose([
            SquarePad(),
            transforms.Resize(112),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    def extract_feature(self, imgs, batchsize=1):
        img_dataset = ImagefromData(imgs, self.transform)
        img_loader = data.DataLoader(img_dataset, batch_size = batchsize, shuffle=False)
        # feat = self.model()
        features = torch.zeros((len(imgs), 512))
        with torch.no_grad():
            for i, img_batch in enumerate(img_loader):
                feat = self.model(img_batch)
                features[i*batchsize: (i+1)*batchsize, :] = feat
        features = np.array(features)
        return features

if __name__ == '__main__':
    img_folder = 'add_image'
    paths = []
    for root, folders, files in os.walk(img_folder):
        for file in files:
            paths.append(os.path.join(root, file))
    num = 0
    # paths = paths[:500]
    # with open('add_image.txt','w') as f:
    #     for path in paths:
    #         print(path, file=f)
    # exit()
    transform = transforms.Compose([
        transforms.Resize(112,112),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    dataset = imagesfromList(paths,transform)
    batchsize = 8
    data_loader = data.DataLoader(dataset,batch_size=batchsize, shuffle=False)
    # print(len(data_loader.dataset))
    # exit()
    model = get_model('r50', fp16=False)
    # print(model)
    # exit()
    checkpoint = 'ms1mv3_arcface_r50_fp16/backbone.pth'
    # checkpoint = 'ms1mv3_arcface_r50_fp16/rank_0_softmax_weight_mom.pt'
    # check_point = torch.load(checkpoint,map_location=torch.device('cpu'))
    model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
    model.eval()
    features = torch.zeros((len(paths), 512))
    with torch.no_grad():
        for i, imgs in tqdm(enumerate(data_loader)):
            feat = model(imgs)
            features[i*batchsize : (i+1)*batchsize] = feat
    print(features)
    features = np.array(features)
    # print(features.shape)
    np.save('add_db.npy',features)
