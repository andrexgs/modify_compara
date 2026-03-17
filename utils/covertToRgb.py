import os
import cv2

root_dir = "../data/all/"

for class_name in os.listdir(root_dir):
    class_dir = os.path.join(root_dir, class_name)

    if os.path.isdir(class_dir):
        print(f"Converting images in {class_name} folder...")

        for filename in os.listdir(class_dir):
            image_path = os.path.join(class_dir, filename)

            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                image = cv2.imread(image_path)

                if image is not None:
                    if image.shape[-1] > 3:
                        image = image[:, :, :3]
                    elif image.shape[-1] == 1:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                    cv2.imwrite(image_path, image)
                else:
                    print(f"Error loading {filename}")

print("Image conversion and overwriting completed.")