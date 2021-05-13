# 更改自demo_inference.py
from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv
import os
from tqdm import tqdm
import numpy as np
import json
import glob
import cv2
import matplotlib.pyplot as plt

from mine.LR import lr_infer


def post_treatment(img_path, result, class_names, score_thr=0.3):
    """分别统计实例分割结果中每个mask的面积（像素数）。

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
    label = img_path.split('\\')[1]
    h, w, _ = img.shape
    img_info = {}
    img_info['img'] = img_path
    img_info['label'] = label
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
        # cur_mask = (cur_mask > 0.5).astype(np.int32)
        density = cur_mask.sum()
        # plt.imshow(cur_mask)
        # plt.show()
        # 浓度（所占图像的比例，单位：%）
        density_ratio = density / total_area * 100
        # 周长
        # contours, hierarchy = cv2.findContours(
        #     image=cur_mask,
        #     mode=cv2.RETR_EXTERNAL,
        #     method=cv2.CHAIN_APPROX_SIMPLE)
        # if len(contours) > 1:  # 一个mask中有多个封闭区域（一个对象被分成几部分）
        #     print(img_path)
        #     plt.imshow(cur_mask)
        #     plt.imsave()
        #     print(len(contours))

        mask_info.append({
            'label': mask_label,
            'density': float(density),
            'ratio': density_ratio,
            'perimeter': None})
    # 将mask按面积（density）降序排列
    mask_density = [mask['density'] for mask in mask_info]
    orders = np.argsort([-d for d in mask_density])  # 加负号是为了降序，默认是升序
    mask_info = [mask_info[o] for o in orders]
    img_info['mask'] = mask_info

    return img_info


def solo_infer(solo_cp_path, src_folder, dst_folder, json_path):
    config_file = '../configs/solo/decoupled_solo_r50_fpn_8gpu_3x.py'

    # 加载solo模型
    model = init_detector(config_file, solo_cp_path, device='cuda:0')

    # 创建json文件
    fjson = open(json_path, 'w')
    # 读取所有输入图像的路径
    all_img_paths = glob.glob(os.path.join(src_folder, '*/*.*'))
    num_img = len(all_img_paths)
    # 开始处理
    state = {'num': float(num_img)}
    content = []  # 所有图像的信息
    for i in tqdm(range(num_img)):
        img_path = all_img_paths[i]
        img_name = img_path.split('/')[-1]
        if cv2.imread(img_path) is None:
            print(img_path)
            continue
        result = inference_detector(model, img_path)  # 单张图像通过solo
        dst_path = os.path.join(dst_folder, img_name)  # 以相同图像名称保存
        show_result_ins(img_path, result, model.CLASSES, score_thr=0.25,
                        out_file=dst_path)  # 绘制 mask并保存
        # 分析mask信息
        img_info = post_treatment(img_path, result, model.CLASSES,
                                  score_thr=0.25)
        # print(img_info)
        content.append(img_info)
    # 保存到json文件
    state['content'] = content
    json.dump(state, fjson, indent=4)
    fjson.close()


def two_tage_infer(solo_cp_path, lr_cp_path, src_folder, dst_folder, json_path):
    solo_infer(solo_cp_path, src_folder, dst_folder, json_path)
    test_acc = lr_infer(lr_cp_path, json_path)
    print("总体准确率：", test_acc)


def check_img_path():
    """检查src文件夹下是否有图像的文件名错误（无法读取），如果有则输出"""
    src = 'D:/Dataset/Wear/XinYang/ALL/Batch2/12img/split/molilian100/split/test'
    all_img_paths = glob.glob(os.path.join(src, '*/*.*'))
    num_img = len(all_img_paths)
    for i in tqdm(range(num_img)):
        img_path = all_img_paths[i]
        if cv2.imread(img_path) is None:
            print(img_path)


if __name__ == '__main__':
    # 联合测试
    work_folder = 'epoch-36_loss-0.1216_768x576'
    solo_model_path = 'D:/temp/trained/%s/epoch_36.pth' % work_folder
    lr_path = 'D:/temp/svm-7209.pipe'
    src = 'D:/Dataset/Wear/XinYang/ALL/Batch2/12img/split/molilian100/split/test'
    segment_result_folder = 'D:/temp/results/test_batch2'
    json_path = 'D:/temp/results/test_batch2.json'

    two_tage_infer(solo_cp_path=solo_model_path,
                   lr_cp_path=lr_path,
                   src_folder=src,
                   dst_folder=segment_result_folder,
                   json_path=json_path)






