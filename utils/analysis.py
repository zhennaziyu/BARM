import os
import torch
import clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import collections

# --- 1. 设置 ---
DATA_PATH = "/mnt/raid1/zhengna/CVPR_2026/food-101/images"  # 修改为你的实际路径
BATCH_SIZE = 64
device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

class LocalFood101(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir): continue
            for img_name in os.listdir(cls_dir):
                self.images.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls]))

    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform: image = self.transform(image)
        return image, label

dataset = LocalFood101(DATA_PATH, transform=preprocess)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 2. 准备检测偏见的 Prompt ---
classes = dataset.classes
text_inputs = torch.cat([clip.tokenize(f"a photo of {c.replace('_', ' ')}, a type of food.") for c in classes]).to(device)

# --- 3. 执行推理并记录偏见 ---
def analyze_bias():
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        for images, labels in tqdm(loader, desc="Detecting Bias"):
            images = images.to(device)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = (100.0 * image_features @ text_features.T)
            preds = logits.argmax(dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # --- 4. 统计分析 ---
    class_correct = collections.defaultdict(int)
    class_total = collections.defaultdict(int)
    confusion = collections.defaultdict(lambda: collections.defaultdict(int))

    for p, l in zip(all_preds, all_labels):
        class_total[l] += 1
        if p == l:
            class_correct[l] += 1
        else:
            confusion[l][p] += 1 # 记录 l 被错认成了 p

    # 计算各类别准确率
    accs = {classes[i]: (class_correct[i] / class_total[i]) for i in range(len(classes))}
    sorted_accs = sorted(accs.items(), key=lambda x: x[1])

    print("\n--- [Bias Analysis] Accuracy 较低的类 (最容易受攻击) ---")
    for name, acc in sorted_accs[:10]:
        print(f"{name}: {acc:.2%}")

    print("\n--- [Bias Analysis] 最典型的混淆对 (偏见轴 b 的来源) ---")
    bias_pairs = []
    for l_idx, p_dict in confusion.items():
        for p_idx, count in p_dict.items():
            bias_pairs.append((classes[l_idx], classes[p_idx], count))
    
    # 按混淆次数排序
    for l_name, p_name, count in sorted(bias_pairs, key=lambda x: x[2], reverse=True)[:10]:
        print(f"真理是 [{l_name}] 但被错认成 [{p_name}]: {count} 次")

if __name__ == "__main__":
    analyze_bias()