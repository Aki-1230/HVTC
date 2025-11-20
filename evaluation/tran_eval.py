import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as getpsnr
import argparse
from pathlib import Path
import shutil

def getssim(im1, im2):
    data = ssim(im1, im2, channel_axis=-1)
    return data

def output(index, images, data_names, datas):
    print('The results of ' + index + '_' + images + ':')
    print('max_' + images + ':' + str('  ') + str(max(datas)) + str('     ') +
          data_names[datas.index(max(datas))].split('f')[0] + '.png')
    print('min_' + images + ':' + str('  ') + str(min(datas)) + str('     ') +
          data_names[datas.index(min(datas))].split('f')[0] + '.png')
    print('mean_' + images + ':' + str('  ') + str(np.mean(datas)))

def compute_metrics(filepath, choose_index, choose_model):
    files = os.listdir(filepath)
    filenames = [item for item in files if os.path.isfile(os.path.join(filepath, item))]
    nums = len(filenames)
    fake_A_names, fake_B_names = [], []

    for i in range(nums):
        name = filenames[i]
        x = name.split('.')[0]
        y = x.split('_')
        ### x: xx_xx_..._fake_A/B or xx_xx_..._real_A/B or xx_xx_..._rec_A/B
        if y[-2] == 'fake':
        # if y[-2] == 'fake' and y[-3] != 'gray' and y[-3] !='edge':
            if y[-1] == 'B':
                fake_B_names.append(name)
            elif y[-1] == 'A':
                fake_A_names.append(name)

    num = len(fake_B_names)
    print('++++++++++++++++++++++++++ evaluation ++++++++++++++++++++++++++')
    print('The number of test images:' + str('  ') + str(num))
    print('Start the evaluation ! ! ! !')

    datas_ssim_A = []
    datas_ssim_B = []
    datas_psnr_A = []
    datas_psnr_B = []

    if choose_model == 'Single':
        for i in range(num):    
            fake_B = cv2.imread(os.path.join(filepath, fake_B_names[i]), 1)                     
            name_B = fake_B_names[i].split('f')[0] + 'real_B.png'         
            real_B = cv2.imread(os.path.join(filepath, name_B), 1)  
            ssim_B = getssim(real_B,fake_B)
            datas_ssim_B.append(ssim_B)
            psnr_B = getpsnr(real_B,fake_B)
            datas_psnr_B.append(psnr_B)        
    else:
        for i in range(num):
            fake_A = cv2.imread(os.path.join(filepath, fake_A_names[i]), 1)
            fake_B = cv2.imread(os.path.join(filepath, fake_B_names[i]), 1)
            name_A = fake_A_names[i].split('f')[0] + 'real_A.png'
            name_B = fake_B_names[i].split('f')[0] + 'real_B.png'
            real_A = cv2.imread(os.path.join(filepath, name_A))
            real_B = cv2.imread(os.path.join(filepath, name_B))
            ssim_A = getssim(real_A,fake_A)
            datas_ssim_A.append(ssim_A)
            ssim_B = getssim(real_B,fake_B)
            datas_ssim_B.append(ssim_B)
            psnr_A = getpsnr(real_A,fake_A)
            datas_psnr_A.append(psnr_A)
            psnr_B = getpsnr(real_B,fake_B)
            datas_psnr_B.append(psnr_B)

    if choose_model == 'Single':
    
        if choose_index == 'ssim':
            print('-----------------------------------------------------------------')
            output('ssim', 'B', fake_B_names, datas_ssim_B)
            print('-----------------------------------------------------------------')
        elif choose_index == 'psnr':
            print('-----------------------------------------------------------------')
            output('psnr', 'B', fake_B_names, datas_psnr_B)
            print('-----------------------------------------------------------------')
        else:
            print('-----------------------------------------------------------------')
            output('ssim', 'B', fake_B_names, datas_ssim_B)
            print('-----------------------------------------------------------------')
            output('psnr', 'B', fake_B_names, datas_psnr_B)
            print('-----------------------------------------------------------------')    
    
    else:   
        if choose_index == 'ssim':
            print('-----------------------------------------------------------------')
            output('ssim', 'A', fake_A_names, datas_ssim_A)
            print('-------------------------------')
            output('ssim', 'B', fake_B_names, datas_ssim_B)
            print('-----------------------------------------------------------------')
        elif choose_index == 'psnr':
            print('-----------------------------------------------------------------')
            output('psnr', 'A', fake_A_names, datas_psnr_A)
            print('-------------------------------')
            output('psnr', 'B', fake_B_names, datas_psnr_B)
            print('-----------------------------------------------------------------')
        else:
            print('-----------------------------------------------------------------')
            output('ssim', 'A', fake_A_names, datas_ssim_A)
            print('-------------------------------')
            output('ssim', 'B', fake_B_names, datas_ssim_B)
            print('-----------------------------------------------------------------')
            output('psnr', 'A', fake_A_names, datas_psnr_A)
            print('-------------------------------')
            output('psnr', 'B', fake_B_names, datas_psnr_B)
            print('-----------------------------------------------------------------')
    print('End the evaluation ! ! ! !')

def find_and_copy_fake_b_images(src_directory, dst_directory):
    src_path = Path(src_directory)
    dst_path = Path(dst_directory)
    
    dst_path.mkdir(parents=True, exist_ok=True)
    
    fake_b_images = list(src_path.rglob('*fake_B.png'))
    if not fake_b_images:
        print("No 'fake_B.png' files found.")
        return

    failed = []

    for img_file in fake_b_images:
        new_file_path = dst_path / img_file.name
        try:
            shutil.copy2(img_file, new_file_path)
        except Exception as e:
            failed.append((img_file, str(e)))
    
    if failed:
        print(f"Copy finished with {len(failed)} errors")
    else:
        print("Copy finished! All files copied successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation for results.')
    parser.add_argument('--result_path', type=str, default='', help='path to the results of images')
    parser.add_argument('--index', type=str, default='ssim and psnr', help='which index to calculate, the choices are [ssim , psnr,or ssim and psnr]')
    parser.add_argument('--model', type=str, default='Single', help='which index to calculate, the choices are [Single,Cycle]')
    parser.add_argument('--gt_path', type=str, default='', help='path to the ground truth images')

    args= parser.parse_args()
    filepath, choose_index, choose_model = args.result_path, args.index, args.model
    gt_path = args.gt_path
    fake_img_path = os.path.join(filepath, 'fake_B')

    compute_metrics(filepath, choose_index, choose_model)
    find_and_copy_fake_b_images(filepath, fake_img_path)
