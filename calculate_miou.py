import numpy as np
from tqdm import tqdm
import easydict
import os

from dataloaders import make_data_loader
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

from modeling.deeplab import *

from utils.loss import SegmentationLosses
from utils.lr_scheduler import LR_Scheduler

from modeling.sync_batchnorm.replicate import patch_replication_callback

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # Set the GPUs 2 and 3 to use

def calculate_miou(args):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    weight = None
    loss_func = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
    train_loader, val_loader, test_loader, nclass = make_data_loader(args, **kwargs) 
    evaluator = Evaluator(nclass)


    if args.dataset == "pascal":
        checkpoint = torch.load(args.checkpoint_dir)
        model = DeepLab(num_classes=21,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=False,
                        freeze_bn=False)

    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    evaluator.reset()

    tbar = tqdm(val_loader, desc='\r')
    test_loss = 0.0
    d = 1
    for i, sample in enumerate(tbar):
        image, target = sample['image'], sample['label']
        if args.cuda:
            image, target = image.cuda(), target.cuda()
        with torch.no_grad():
            output = model(image)
        loss = loss_func(output, target)
        test_loss += loss.item()
        tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
        pred = output.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        evaluator.add_batch(target, pred)


    mIoU = evaluator.Mean_Intersection_over_Union()
    print("\n", d, mIoU)


args = easydict.EasyDict({
    "checkpoint_dir" : "2022-04-19-cp-pascal-deeplab-resnet/model_best.pth.tar",
    "dataset" : "pascal",
    "use_sbd" : False,
    "crop_size" : 513,
    "batch_size" : 4,
    "backbone" : "resnet",
    "out_stride" : 16,
    "workers" : 4,
    "loss_type" : "ce",
    "cuda" : True,
})

if __name__ == '__main__':
    calculate_miou(args)
