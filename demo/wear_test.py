# 更改自demo_inference.py
from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv
import os
from tqdm import tqdm
import numpy as np


def process_result(img, result, class_names, score_thr=0.3):
    """统计实例分割结果中所有mask的面积（像素数）。

    Args:
        img (str): 图像的路径
        result (tuple[list] or list): 实例分割结果，[(Tensor1, Tensor2, Tensor3)]。
        class_names (list[str] or tuple[str]): 类别名称，如：[’chain‘, 'oil', 'fiber']。
        score_thr (float): mask得分的阈值。

    Returns:
        list of tuples: 每个mask对应一个tuple，如：('chain', 120)代表该mask为chain，面积为120像素。
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    h, w, _ = img.shape
    cur_result = result[0]
    seg_label = cur_result[0]  # Tensor1
    seg_label = seg_label.cpu().numpy().astype(np.uint8)  # 将bool转成0和1
    cate_label = cur_result[1]  # Tensor2
    cate_label = cate_label.cpu().numpy()
    score = cur_result[2].cpu().numpy()  # Tensor3

    vis_inds = score > score_thr
    seg_label = seg_label[vis_inds]
    num_mask = seg_label.shape[0]
    cate_label = cate_label[vis_inds]

    mask_info = []
    total_area = w * h
    if sort_by_density:
        mask_density = []
        for idx in range(num_mask):
            cur_mask = seg_label[idx, :, :]
            cur_mask = mmcv.imresize(cur_mask, (w, h))
            cur_mask = (cur_mask > 0.5).astype(np.int32)
            mask_density.append(cur_mask.sum())
        orders = np.argsort(mask_density)
        seg_label = seg_label[orders]
        cate_label = cate_label[orders]
        cate_score = cate_score[orders]
    for idx in range(num_mask):
        # 标签
        cur_cate = cate_label[idx]  # 类别序号
        mask_label = class_names[cur_cate]  # 类别名称
        # 面积
        cur_mask = seg_label[idx, :, :]
        cur_mask = mmcv.imresize(cur_mask, (w, h))
        cur_mask = (cur_mask > 0.5).astype(np.int32)
        mask_density = cur_mask.sum()
        # 浓度（所占图像的比例，单位：%）
        density_ratio = mask_density / total_area * 100

        mask_info.append((mask_label, mask_density, density_ratio))

    return mask_info


if __name__ == '__main__':
    checkpoint_name = 'epoch-36_loss-0.1216_768x576'
    config_file = '../configs/solo/decoupled_solo_r50_fpn_8gpu_3x.py'
    # download the checkpoint from model zoo and put it in `checkpoints/`
    checkpoint_file = 'D:/temp/trained/%s/epoch_36.pth' % checkpoint_name

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    # mmdet.models.detectors.solo.SOLO

    # 测试一批图像
    # for c in range(4):
    #     img_prefix = 'D:/Dataset/Wear/XinYang/instance_segment/data/test/%d' % c
    #     dst_prefix = 'D:/temp/trained/%s/result' % checkpoint_name
    #     all_imgs = os.listdir(img_prefix)
    #     for i in tqdm(range(len(all_imgs))):
    #         img = all_imgs[i]
    #         img_path = os.path.join(img_prefix, img)
    #         result = inference_detector(model, img_path)
    #         dst_path = os.path.join(dst_prefix, img)
    #         show_result_ins(img_path, result, model.CLASSES, score_thr=0.25,
    #                         out_file=dst_path)

    # 测试单张图像
    img_path = 'D:/temp/test/raw-c1-081.bmp'
    result = inference_detector(model, img_path)
    # dst_path = 'D:/temp/test/output.bmp'
    info = process_result(img_path, result, model.CLASSES, score_thr=0.25)
    print(info)




