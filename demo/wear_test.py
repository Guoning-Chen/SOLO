# 更改自demo_inference.py
from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv
import os
from tqdm import tqdm
import numpy as np
import json
import glob


def post_treatment(img_path, result, class_names, score_thr=0.3):
    """统计实例分割结果中所有mask的面积（像素数）。

    Args:
        img_path (str): 图像的路径
        result (tuple[list] or list): 实例分割结果，[(Tensor1, Tensor2, Tensor3)]。
        class_names (list[str] or tuple[str]): 类别名称，如：[’chain‘, 'oil', 'fiber']。
        score_thr (float): mask得分的阈值。

    Returns:
        list of tuples: 每个mask对应一个tuple，如：('chain', 120)代表该mask为chain，面积为120像素。
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img_path)
    h, w, _ = img.shape
    img_info = {}
    img_info['img'] = img_path
    cur_result = result[0]
    if cur_result is None:  # 没有检测到实例
        img_info['num'] = 0
        img_info['mask'] = {}
        return img_info
    seg_label = cur_result[0]  # Tensor1
    seg_label = seg_label.cpu().numpy().astype(np.uint8)  # 将bool转成0和1
    cate_label = cur_result[1]  # Tensor2
    cate_label = cate_label.cpu().numpy()
    score = cur_result[2].cpu().numpy()  # Tensor3

    vis_inds = score > score_thr
    seg_label = seg_label[vis_inds]
    num_mask = seg_label.shape[0]
    cate_label = cate_label[vis_inds]

    total_area = w * h
    img_info['num'] = num_mask
    mask_info = []
    for idx in range(num_mask):
        # 标签
        cur_cate = cate_label[idx]  # 类别序号
        mask_label = class_names[cur_cate]  # 类别名称
        # 面积
        cur_mask = seg_label[idx, :, :]
        cur_mask = mmcv.imresize(cur_mask, (w, h))
        cur_mask = (cur_mask > 0.5).astype(np.int32)
        density = cur_mask.sum()
        # 浓度（所占图像的比例，单位：%）
        density_ratio = density / total_area * 100

        mask_info.append({
            'label': mask_label,
            'density': float(density),
            'ratio': density_ratio})
    # 将mask按面积（density）降序排列
    mask_density = [mask['density'] for mask in mask_info]
    orders = np.argsort([-d for d in mask_density])  # 加负号是为了降序，默认是升序
    mask_info = [mask_info[o] for o in orders]
    img_info['mask'] = mask_info

    return img_info


if __name__ == '__main__':
    checkpoint_name = 'epoch-36_loss-0.1216_768x576'
    config_file = '../configs/solo/decoupled_solo_r50_fpn_8gpu_3x.py'
    # download the checkpoint from model zoo and put it in `checkpoints/`
    checkpoint_file = 'D:/temp/trained/%s/epoch_36.pth' % checkpoint_name

    # build the model from a config file and a checkpoint file
    # mmdet.models.detectors.solo.SOLO
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # 保存一批图像的结果
    json_prefix = 'D:/temp/results'
    img_prefix = 'D:/Dataset/Wear/XinYang/instance_segment/data/test/2'
    all_img_files = glob.glob(os.path.join(img_prefix, '*.bmp'))
    fjson = open(os.path.join(json_prefix, 'train_details.json'), 'w')
    # dst_prefix = 'D:/temp/trained/%s/result' % checkpoint_name
    num_img = len(all_img_files)
    state = {}
    state['num'] = float(num_img)
    content = []
    for i in tqdm(range(num_img)):
        img_file = all_img_files[i]
        result = inference_detector(model, img_file)
        # dst_path = os.path.join(dst_prefix, img)
        # show_result_ins(img_path, result, model.CLASSES, score_thr=0.25,
        #                 out_file=dst_path)
        # 分析mask信息
        info_dict = post_treatment(img_file, result, model.CLASSES,
                                   score_thr=0.25)
        print(info_dict)
        content.append(info_dict)
    # 保存到json文件
    state['content'] = content
    json.dump(state, fjson, indent=4)
    fjson.close()





