import os
import datetime
import re
import torch

from untils import train_one_epoch,evaluate,create_logger,count_params
from my_dataset import VOCSegmentation
import transforms as T
import torch.optim.lr_scheduler as lr_scheduler

from model.networks import FoodSwinV2
from thop import profile, clever_format

def poly_rate(optimizer, epoch, num_epochs, base_lr, power=0.9):
    lr = max(base_lr * (1-epoch/num_epochs)**power,2e-7)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

class SegmentationPresetTrain:
    def __init__(self, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * crop_size)
        max_size = int(2 * crop_size)

        trans = [T.RandomResize2(min_size, max_size)]  
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self,base_size=768, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.Resize([base_size, base_size]),
            # T.RandomCrop(base_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train):
    base_size = 256
    crop_size = 256

    return SegmentationPresetTrain(crop_size) if train else SegmentationPresetEval(base_size)


def create_model(num_classes, pretrain=False):
    model = FoodSwinV2(num_classes=num_classes)
    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> train.txt
    train_dataset = VOCSegmentation(args.data_path,
                                    transforms=get_transform(train=True),
                                    txt_name="train.txt")

    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
    test_dataset = VOCSegmentation(args.data_path,
                                   transforms=get_transform(train=False),
                                   txt_name="test.txt")
    '''
    unlable_dataset = VOCSegmentation(args.data_path,
                                    transforms=get_transform(train=True),
                                    txt_name="unlable.txt")
    
    trainu_loader = torch.utils.data.DataLoader(unlable_dataset,
                                            batch_size=4,
                                            num_workers=num_workers,
                                            pin_memory=True)
    '''
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              collate_fn=test_dataset.collate_fn)
    
    
    model = create_model(num_classes=num_classes)

    #w = torch.load('save_weights/model.pth')['model']
    #model.load_state_dict(w)
    model.to(device)
    print(count_params(model))
    
    params_to_optimize = [
        {"params": [p for p in model.parameters() if p.requires_grad]},
    ]
    

    # optimizer = SGD(model.parameters(),lr=args.lr,momentum=0.9, weight_decay=1e-4)
    #optimizer = SGD(params_to_optimize,lr=args.lr,momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.lr, weight_decay=1e-8)
    
    StepLR = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=15,gamma=0.75)


    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "backbone.se" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))


    logger = create_logger('log')
    logger.info('------Begin Training Model------')
    best = 0

    for epoch in range(0, args.epochs):
        # 将下面两行训练代码注释掉，引入对应的权重即可直接跑测试集        
        loss, lr = train_one_epoch(model, optimizer, train_loader ,device=device)
        #StepLR.step()
        lr = poly_rate(optimizer,epoch,args.epochs,args.lr)
        if epoch >= 0:
            confmat = evaluate(model, test_loader, device=device, num_classes=num_classes,epoch=epoch)
            val_info = str(confmat)

            match = re.search(r'mean IoU:\s*([0-9.]+)', val_info)
            if match:
                mean_iou = float(match.group(1))
                if mean_iou > best:
                    # 只有在 mean IoU 大于 x 时才保存这一轮的权重
                    best = mean_iou
                    if mean_iou > 44:
                        save_file = {"model": model.state_dict(),
                                    #"optimizer": optimizer.state_dict(),
                                   # "epoch": epoch,
                                    "args": args}

                        torch.save(save_file, "save_weights/model_{}.pth".format(epoch))


            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {loss:.4f}\n" \
                         f"lr1: {lr:.8f}\n" \
                         f"best: {best:.2f} \n"

                         
            logger.info(train_info+val_info)
            #logger.info(val_info)
            # write into txt
            

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn training")
    # 在my_dataset.py中切换数据集
    parser.add_argument("--data-path", default="../../dataset")
    # 类别数 FoodSeg103设置为103，UEC设置为102
    parser.add_argument("--num-classes", default=103, type=int)
    parser.add_argument("--aux", default=False, type=bool, help="auxilier loss")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=2, type=int)
    parser.add_argument("--epochs", default=150, type=int, metavar="N",
                        help="number of total epochs to train")
    # 初始学习率 103是0.000012，UEC是0.000006
    parser.add_argument('--lr', default=0.00001, type=float, help='initial learning rate')
    parser.add_argument('--lr2', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01)

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=15, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=True, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    # 训练时要导入STDC网络的预训练权重
    parser.add_argument('--weights', type=str, default='../dataset/weights/pretrain/STDC.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    # args = parse_args()
    
    # if not os.path.exists("./save_weights"):
    #     os.mkdir("./save_weights")
    # main(args)
    

    model = create_model(num_classes=103)
    model.cuda()
    model.eval()
    input = torch.randn(1,3,256,256).cuda()
    output = model(input)
    print(output.shape)
    print('Model %s created, param count: %d' %
                     ('EMCAD decoder: ', sum([m.numel() for m in model.parameters()])))

    
    

    

