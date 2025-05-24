import os
import numpy as np
from PIL import Image
from itertools import combinations

# ========== 参数设置 ==========
DATASET_DIR = '..\\dataSet_pic_1epoch'
# IMAGE_SIZE = (128, 128)  # 图像统一尺寸
THRESHOLDS = {}

from PIL import Image, ImageDraw

# def render_curve_to_image(y_curve, image_size=(128, 128), line_width=1):
#     """
#     将y曲线向量还原为图像：每个点为曲线上的最黑像素
#     """
#     height, width = image_size
#     img = Image.new('L', image_size, color=255)  # 白底灰度图
#     draw = ImageDraw.Draw(img)
#
#     for x in range(len(y_curve)):
#         y = int(y_curve[x])
#         # 画线或点
#         draw.line((x, y, x, y + line_width - 1), fill=0)
#
#     return img

# ========== 轨迹提取 ==========
def load_y_curve(path):
    """
    提取图像每列中最深（最黑）像素的y坐标作为波形轨迹
    """
    img = Image.open(path).convert('L')
    img_array = np.array(img)
    height, width = img_array.shape
    y_curve = [np.argmin(img_array[:, col]) for col in range(width)]
    return np.array(y_curve, dtype=np.float32)

cnt = 1
# ========== 距离计算 ==========
def euclidean_y_distance(path1, path2):
    y1 = load_y_curve(path1)
    # img1 = render_curve_to_image(y1, image_size=(128, 128))
    # global cnt
    # if cnt <= 5:
    #     img1.show()  # 预览
    #     cnt = cnt + 1
    y2 = load_y_curve(path2)
    return np.linalg.norm(y1 - y2)

# ========== 类内最大距离作为阈值 ==========
def compute_user_thresholds():
    for user_id in sorted(os.listdir(DATASET_DIR)):
        user_path = os.path.join(DATASET_DIR, user_id)
        if not os.path.isdir(user_path):
            continue

        img_paths = [
            os.path.join(user_path, file)
            for file in sorted(os.listdir(user_path)) if file.endswith('.png')
        ]

        max_dist = 0
        for img1, img2 in combinations(img_paths, 2):
            dist = euclidean_y_distance(img1, img2)
            max_dist = max(max_dist, dist)

        THRESHOLDS[user_id] = max_dist
        print(f"👤 用户 {user_id} 的最大Y轨迹距离阈值：{max_dist:.2f}")

# ========== 多数投票判断是否属于目标类 ==========
def check_majority_belongs_to_user(test_img_path, user_id):
    """
    判断测试图像是否属于目标用户类：
    多数样本未超出距离阈值 → 属于；多数超出 → 不属于
    """
    user_path = os.path.join(DATASET_DIR, user_id)
    if not os.path.isdir(user_path):
        return False

    user_imgs = [
        os.path.join(user_path, file)
        for file in sorted(os.listdir(user_path)) if file.endswith('.png')
    ]

    over_threshold = 0
    for user_img in user_imgs:
        dist = euclidean_y_distance(test_img_path, user_img)
        print(dist)
        if dist > THRESHOLDS[user_id]:
            over_threshold += 1

    majority_needed = len(user_imgs) // 2 + 1
    return over_threshold < majority_needed  # True 表示“属于”，False 表示“不属于”

# ========== 批量验证测试集是否属于指定用户类 ==========
def batch_verify_class_majority(test_folder, target_user_id):
    print(f"\n🔍 开始测试：是否属于用户类 {target_user_id}（多数判断）\n")

    for subfolder in sorted(os.listdir(test_folder)):
        subfolder_path = os.path.join(test_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        for file in sorted(os.listdir(subfolder_path)):
            if not file.endswith(".png"):
                continue

            test_path = os.path.join(subfolder_path, file)
            belongs = check_majority_belongs_to_user(test_path, target_user_id)

            result = "✅ 属于该类（多数未超）" if belongs else "❌ 不属于（多数超出）"
            print(f"📄 {subfolder}/{file} → {result}")

# ========== 主入口 ==========
if __name__ == '__main__':
    compute_user_thresholds()
    batch_verify_class_majority('..\\testSet_pic_1epoch', target_user_id='0')
