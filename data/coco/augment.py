import cv2
import os
import numpy as np
import shutil
from tqdm import tqdm
import random


def mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)  # delete folders recursively
    os.mkdir(path)  # create


def random_crop(src, dst, n, scale, dst_size=(224, 224)):
    # 使用（原图尺寸*scale）大小的矩形框，对src文件夹下的图片随机裁剪 n 次并缩放到尺寸
    # dst_size后保存到 dst文件夹下(同时会保存一份直接缩放的原图，即 num_crops+1倍)
    for img_name in os.listdir(src):
        img_path = os.path.join(src, img_name)
        img = cv2.imread(img_path)
        src_h, src_w = img.shape[0], img.shape[1]
        new_h = int(src_h * scale)
        new_w = int(src_w * scale)
        # 随机裁剪 n次
        left_top_h_range = np.random.randint(0, src_h - new_h, n).tolist()
        left_top_w_range = np.random.randint(0, src_w - new_w, n).tolist()
        for i in range(n):
            left_top_h = left_top_h_range[i]
            left_top_w = left_top_w_range[i]
            crop = img[left_top_h:left_top_h + new_h,
                       left_top_w:left_top_w + new_w]
            crop = cv2.resize(crop, dst_size)
            name_split = img_name.split('.')
            new_name = name_split[0] + \
                       '+rc%d-s%d-%d.' % (n, int(scale * 10), i + 1) + \
                       name_split[1]
            cv2.imwrite(os.path.join(dst, new_name), crop)


def flip(src):
    # 对文件夹src下的每张图片都进行翻转后保存（4X增广）
    for img_name in os.listdir(src):
        img_path = os.path.join(src, img_name)
        img = cv2.imread(img_path)
        name_split = img_name.split('.')
        # 左右翻转
        new_name = name_split[0] + '+hf.' + name_split[1]
        imgH = cv2.flip(img, 1)
        cv2.imwrite(os.path.join(src, new_name), imgH)
        # 上下翻转
        imgV = cv2.flip(img, 0)
        new_name = name_split[0] + '+vf.' + name_split[1]
        cv2.imwrite(os.path.join(src, new_name), imgV)
        # 上下+左右
        imgHV = cv2.flip(img, -1)
        new_name = name_split[0] + '+hvf.' + name_split[1]
        cv2.imwrite(os.path.join(src, new_name), imgHV)


def contrast_and_brightness(src, alpha, beta):
    # 改变 src文件夹下图片的亮度和对比度，dst = alpha * img + beta * blank
    for img_name in os.listdir(src):
        img_path = os.path.join(src, img_name)
        img = cv2.imread(img_path)
        name_split = img_name.split('.')
        new_name = name_split[0] + '+cb.' + name_split[1]
        blank = np.zeros(img.shape, img.dtype)
        dst = cv2.addWeighted(img, alpha, blank, 1 - alpha, beta)
        cv2.imwrite(os.path.join(src, new_name), dst)


def hsv_jitter(src):
    # HSV颜色空间上的色彩抖动
    pass


def pca_jitter(src):
    # 利用 PCA在 RGB颜色空间进行颜色抖动
    pass


def resize_dataset(src, dst, dst_size=(224, 224)):
    for c in os.listdir(src):
        class_folder = os.path.join(src, c)
        dst_folder = os.path.join(dst, c)
        if not os.path.exists(dst_folder):
            mkdir(dst_folder)
        for img_name in os.listdir(class_folder):
            img = cv2.imread(os.path.join(class_folder, img_name))
            img = cv2.resize(img, dst_size)
            cv2.imwrite(os.path.join(dst_folder, img_name), img)


def resize(src, dst, dst_size=(224, 224)):
    img_names = os.listdir(src)
    for i in tqdm(range(len(img_names))):
        img_name = img_names[i]
        img = cv2.imread(os.path.join(src, img_name))
        img = cv2.resize(img, dst_size)
        cv2.imwrite(os.path.join(dst, img_name), img)


def split_images(src_folder, dst_folders, nums):
    '''
    把src_folder下的图像，按照 nums[i]所示的数量，分别添加到 dst_folders[i] 下
    :param src_folder: 图像所在文件夹的路径
    :param dst_folders: 包含多个文件夹绝对路径的列表
    :param nums: 数字列表，分别对应dst_folders 的文件夹中的图像数量
    '''
    # 检查输入参数
    assert len(dst_folders) == len(nums), \
        'ERROR(split_images): dst_folders和nums的长度应该相等!'
    img_names = os.listdir(src_folder)
    num_img = len(img_names)
    sum_of_nums = 0  # sum of all images to add to list
    split = [0]
    for n in nums:
        sum_of_nums += n
        split.append(sum_of_nums)
    assert len(img_names) == sum_of_nums, \
        'ERROR(split_images): nums的和应等于src_folder下的图片总数!'

    # 开始复制到dst_folders
    all_indices = np.arange(num_img)
    np.random.shuffle(all_indices)
    for i in range(len(nums)):
        dst_folder = dst_folders[i]
        indices = all_indices[split[i]: split[i + 1]]
        for index in indices:
            img = cv2.imread(os.path.join(src_folder, img_names[index]))
            save_path = os.path.join(dst_folder, img_names[index])
            cv2.imwrite(save_path, img)
    print('Done!')


def random_choose(src_folder, dst_folder, num):
    img_names = os.listdir(src_folder)
    assert len(img_names) > num, 'ERROR：选择的数量不能超过总数！'
    indices = list(range(len(img_names)))
    random.shuffle(indices)
    for i in tqdm(range(num)):
        img_name = img_names[indices[i]]
        img = cv2.imread(os.path.join(src_folder, img_name))
        cv2.imwrite(os.path.join(dst_folder, img_name), img)


def white_bg(save_path):
    # 生成一张纯白色背景图
    img = np.zeros((1944, 2592, 3), np.uint8)
    img[:] = [255, 255, 255]
    cv2.imshow('img', img)
    cv2.imwrite(save_path, img)
    cv2.waitKey(0)


def change_bg(src_folder, bg_path, dst_folder, note):
    """
    对某个文件夹下的所有图像更换背景之后保存到另一个文件夹下
    :param src_folder: 原始图像所在的文件夹
    :param bg_path: 背景图像的绝对路径
    :param dst_folder: 保存结果的文件夹
    :param note: 背景的记号
    :return: 无
    """
    bg = cv2.imread(bg_path)
    assert bg is not None, 'ERROR: 读取不到背景！'
    img_names = os.listdir(src_folder)
    for i in tqdm(range(len(img_names))):
        img_name = img_names[i]
        img_path = os.path.join(src_folder, img_name)
        img = cv2.imread(img_path)
        assert img is not None, 'ERROR: 读取不到图像！'
        dst = bg.copy()
        assert (dst.shape[0] == bg.shape[0]) & (dst.shape[1] == bg.shape[1]), \
            'ERROR：原图和背景的尺寸不一致！'
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, thresh = cv2.threshold(gray, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        for i in range(thresh.shape[0]):
            for j in range(thresh.shape[1]):
                if thresh[i][j] == 255:
                    dst[i][j] = img[i][j]
        # cv2.imshow('result', dst)
        # cv2.waitKey(0)
        splits = img_name.split('.')
        img_name = splits[0] + '-' + note + '.' + splits[1]
        save_path = os.path.join(dst_folder, img_name)
        cv2.imwrite(save_path, dst)


def history():
    """从将src文件中不存在于dst文件夹中的图像添加到others文件夹下"""
    dst = 'D:/Dataset/Wear/XinYang/instance_segment/train/2'
    src = 'D:/Dataset/Wear/XinYang/classification/raw/origin/2'
    others = 'D:/Dataset/Wear/XinYang/instance_segment/test/2'
    dst_full_names = os.listdir(dst)
    src_full_names = os.listdir(src)
    for src_full_name in src_full_names:
        if src_full_name not in dst_full_names:
            img = cv2.imread(os.path.join(src, src_full_name))
            save_path = os.path.join(others, src_full_name)
            cv2.imwrite(save_path, img)


def change_format(src, dst, dst_format):
    img_names = os.listdir(src)
    for i in tqdm(range(len(img_names))):
        img_name = img_names[i]
        img = cv2.imread(os.path.join(src, img_name))
        img_name = img_name.split('.')[0] + '.' + dst_format
        cv2.imwrite(os.path.join(dst, img_name), img)


if __name__ == '__main__':
    src = 'D:/Dataset/Wear/XinYang/instance_segment/train/bmp'
    dst = 'D:/Dataset/Wear/XinYang/instance_segment/train/jpg'
    change_format(src, dst, dst_format='jpg')
