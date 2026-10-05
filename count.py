import os

label_dir = "dataset/train/labels"
helmet = 0
no_helmet = 0

for file in os.listdir(label_dir):
    with open(os.path.join(label_dir, file)) as f:
        for line in f:
            cls = line.strip().split()[0]
            if cls == "0":
                helmet += 1
            elif cls == "1":
                no_helmet += 1

print("Helmet:", helmet)
print("No Helmet:", no_helmet)
