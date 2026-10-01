import torchvision.datasets as datasets

# Load the Food-101 dataset (without downloading)
dataset = datasets.Food101(root="/home/zhengna/Semi/datasets/food-101", split='train', download=False)

# Extract class names
food101_classes = dataset.classes

# Display total number of classes and some sample classes
print(f"Total Food-101 classes: {len(food101_classes)}")
print("Sample classes:", food101_classes[:10])

# Optional: Save class names to a text file
with open("food101_classes.txt", "w") as file:
    for class_name in food101_classes:
        file.write(f"{class_name}\n")
