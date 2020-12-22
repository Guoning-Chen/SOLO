from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv
import os
from tqdm import tqdm

if __name__ == '__main__':
    config_file = '../configs/solo/decoupled_solo_r50_fpn_8gpu_3x.py'
    # download the checkpoint from model zoo and put it in `checkpoints/`
    checkpoint_file = 'D:/temp/trained/epoch_36_loss-0.1036.pth'

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    # mmdet.models.detectors.solo.SOLO

    # test a single image
    img_prefix = 'D:/Dataset/Wear/XinYang/instance_segment/data/test/2'
    dst_prefix = 'D:/Dataset/Wear/XinYang/instance_segment/data/result'
    all_imgs = os.listdir(img_prefix)
    for i in tqdm(range(len(all_imgs))):
        img = all_imgs[i]
        img_path = os.path.join(img_prefix, img)
        result = inference_detector(model, img_path)
        dst_path = os.path.join(dst_prefix, img)
        show_result_ins(img_path, result, model.CLASSES, score_thr=0.25,
                        out_file=dst_path)
