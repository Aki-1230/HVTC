# 推理





    脚本命名：
        test_{下游分割模型}_{预训练数据集}.sh

    推理脚本：
        | ./scripts   
            | test_FCN_whu900.sh        在whu-opt-sar数据集上推理
            | test_FCN_yyx1340.sh       在yyx-opt-sar数据集上推理
            | test_FCN_whu-selected.sh  对whu-opt-sar中展示的图片进行推理
            | test_FCN_yyx-selected.sh  对yyx-opt-sar中展示的图片进行推理

    评估脚本：
        | ./scripts 
            | SSIM_PSNR_eval.sh         ssim, psnr
            | FID_KID_eval.sh           fid, kid
            | seg_eval.sh               miou, macc
    


