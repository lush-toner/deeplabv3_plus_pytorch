from PIL import Image
import numpy as np

import torch
import cv2
import torchvision.transforms as tr

from modeling.deeplab import DeepLab
from dataloaders.utils import decode_segmap

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

cp_file = "./2022-04-19-cp-pascal-deeplab-resnet/model_best.pth.tar"

checkpoint = torch.load(cp_file)

model = DeepLab(num_classes=21,
                backbone='resnet',
                output_stride=16,
                sync_bn=True,
                freeze_bn=False)

model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.to(device)

def transform(image):
    return tr.Compose([
        # tr.Resize(513),
        # tr.CenterCrop(513),
        tr.ToTensor(),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])(image)

torch.set_grad_enabled(False)

image = Image.open('2007_000836.jpg')

# image = Image.open('001.bmp')
# image = image.convert('RGB')

inputs = transform(image).to(device)
output = model(inputs.unsqueeze(0)).squeeze().cpu().numpy()
pred = np.argmax(output, axis=0)
# pred.save('output.jpg')
# Then visualize it:
output = decode_segmap(pred, dataset="pascal", plot=False)

cv2.imwrite("./output.jpg", output)
