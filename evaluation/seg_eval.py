from fileinput import filename
import cv2
import os
import numpy as np
from tqdm import tqdm
import argparse

COLORIZE_YYX = {0:[0,0,0],10:[11,134,184],20:[0,0,255],30:[0,255,255],40:[255,0,0],50:[34,139,34],60:[255,191,0]}
COLORIZE_WHU = {0:[0,0,0],10:[11,134,184],20:[0,0,255],30:[0,255,255],40:[255,0,0],50:[34,139,34],60:[255,191,0],70:[128,0,128]}
MAP = {'whu': COLORIZE_WHU, 'yyx': COLORIZE_YYX}
NUM_CLASSES = {'whu': 8, 'yyx': 7}


def segdataprocess(datapath, savepath, dataset):
    savepath_view = os.path.join(savepath,'results_view/')   # 彩色标签
    savepath_label = os.path.join(savepath,'results_label/') # 数据集标签类型

    os.makedirs(savepath_view, exist_ok=True)
    os.makedirs(savepath_label, exist_ok=True)
    names = os.listdir(datapath)
    
    color_dict = MAP[dataset]
    
    # map label from (0, 255) to (0, 10 * num_classes)
    for name in tqdm(names):
        if name.split('_')[-2] != 'pred':
            continue

        base = name.split('.')[0] if dataset == 'yyx' else name.split('_')[3]
        
        img = cv2.imread(os.path.join(datapath, name), cv2.IMREAD_COLOR)
        b = img[:, :, 0].astype(np.float32)
        class_map = np.rint(b * 100 / 255).astype(np.uint8)

        color_view = np.zeros((*class_map.shape, 3), dtype=np.uint8)

        for cid, bgr in color_dict.items():
            color_view[class_map == cid] = bgr
        
        ii = np.stack([class_map] * 3, axis=-1)

        cv2.imwrite(os.path.join(savepath_view, base + '_view.png'), color_view)
        cv2.imwrite(os.path.join(savepath_label, base + '.png'), ii)

    print('Processing has been completed:', datapath)

def segevaluation(label_gt_path, pred_label_path, dataset):

    num_classes = NUM_CLASSES[dataset]

    # 生成混淆矩阵
    def fast_hist(pred, true, n):
        pred = (pred[:,:,0])//10
        true = (true[:,:,0])//10 
        k = (pred >= 0) & (pred < n)
        return np.bincount(pred[k].astype(int) + n * true[k].astype(int), minlength=n**2).reshape(n,n)

    def per_class_iou(hist):
        np.seterr(divide='ignore', invalid='ignore')
        iou = np.diag(hist) / np.maximum((hist.sum(1)+hist.sum(0)-np.diag(hist)),1)
        np.seterr(divide="warn", invalid="warn")
        iou[np.isnan(iou)] = 0
        return iou

    def per_class_pa(hist):
        return np.diag(hist)/np.maximum(hist.sum(1),1)

    def oa(hist):
        return np.diag(hist).sum()/hist.sum()

    names = os.listdir(label_gt_path)  # 获取真实标签文件夹内所有文件名

    hists = np.zeros((num_classes,num_classes)) 
    oas = []
    for i in range(len(names)):
        name = names[i]
        true = cv2.imread(os.path.join(label_gt_path, name) ,1)
        n = name.split('.')[0] + '_pred_label.png'
        pred = cv2.imread(os.path.join(pred_label_path, n), 1)
        hist = fast_hist(pred, true, num_classes)
        hists += hist
        oas.append(oa(hist))

    class_iou = per_class_iou(hists)
    class_pa = per_class_pa(hists)
    all_miou = np.nanmean(class_iou)
    all_mpa = np.nanmean(class_pa)
    all_oa = sum(oas)/len(names)
    sum_iou = 0
    sum_pa = 0
    for i in range(1, len(class_iou)):
        sum_iou += class_iou[i]
        sum_pa += class_pa[i]
    avg_iou = sum_iou / (len(class_iou) - 1)
    avg_pa = sum_pa / (len(class_iou) - 1)
    

    print('----------------------------')
    print('filename:' + pred_label_path)
    print('number of images:', len(names))
    print('results of oa:', all_oa)
    print('results of mIoU:', all_miou * 100, avg_iou * 100)
    print('results of mPA:', all_mpa * 100, avg_pa * 100)
    print('+------' + '+-----------' + '+-------------------' + '+-------------------+')
    print('|------' + '|   class   ' + '|        IoU        ' + '|        mPA        |')
    print('+------' + '+-----------' + '+-------------------' + '+-------------------+')
    print('------> ' + 'background:' + '|' + str(round(class_iou[0] * 100, 4)).ljust(18), '|' + str(round(class_pa[0] * 100, 4)))
    print('------> ' + 'BareGround:' + '|' + str(round(class_iou[1] * 100, 4)).ljust(18), '|' + str(round(class_pa[1] * 100, 4)))
    print('------> ' + 'Low-Veg:   ' + '|' + str(round(class_iou[2] * 100, 4)).ljust(18), '|' + str(round(class_pa[2] * 100, 4)))
    print('------> ' + 'Trees:     ' + '|' + str(round(class_iou[3] * 100, 4)).ljust(18), '|' + str(round(class_pa[3] * 100, 4)))
    print('------> ' + 'Houses:    ' + '|' + str(round(class_iou[4] * 100, 4)).ljust(18), '|' + str(round(class_pa[4] * 100, 4)))
    print('------> ' + 'Roads:     ' + '|' + str(round(class_iou[5] * 100, 4)).ljust(18), '|' + str(round(class_pa[5] * 100, 4)))
    print('------> ' + 'Others:    ' + '|' + str(round(class_iou[6] * 100, 4)).ljust(18), '|' + str(round(class_pa[6] * 100, 4)))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Procession for results.')
    parser.add_argument('--result_path', type=str, default='', help='path to the results of images')
    parser.add_argument('--dataset', type=str, default='yyx', help='dataset name')
    parser.add_argument('--label_path',type=str, default='', help='path to the true labels of images')
    args = parser.parse_args()

    datapath , savepath, dataset = args.result_path, args.result_path, args.dataset
    lbl_gt_path = args.label_path
    pred_label_path = os.path.join(savepath, 'results_label')


    segdataprocess(datapath, savepath, dataset)
    segevaluation(lbl_gt_path, pred_label_path, dataset)
      