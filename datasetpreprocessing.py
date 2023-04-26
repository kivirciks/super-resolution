from IPython.display import clear_output 
!python3 --version
!pip install opencv-python-headless
!pip install numpy
!pip install tqdm
!pip install torch
!pip install natsort
!pip install typing
!pip install torchvision
!pip install scipy
clear_output()

import multiprocessing
import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm

def main(args) -> None:
    if os.path.exists(args["output_dir"]):
        shutil.rmtree(args["output_dir"])
    os.makedirs(args["output_dir"])
    
    # Get all image paths
    image_file_names = os.listdir(args["images_dir"])

    # Splitting images with multiple threads
    progress_bar = tqdm(total=len(image_file_names), unit="image", desc="Prepare split image")
    workers_pool = multiprocessing.Pool(args["num_workers"])
#     workers_pool.apply_async(worker, args=(image_file_names[0], args), callback=lambda arg: progress_bar.update(1))
    for image_file_name in image_file_names:
#         print(image_file_name)
        workers_pool.apply_async(worker, args=(image_file_name, args), callback=lambda arg: progress_bar.update(1))
    workers_pool.close()
    workers_pool.join()
    progress_bar.close()
    
    
def worker(image_file_name, args) -> None:
#     print("starting")
    image = cv2.imread(f"{args['images_dir']}/{image_file_name}", cv2.IMREAD_UNCHANGED)
#     print("starting2")
    image_height, image_width = image.shape[0:2]
#     print(str(image_height) + "  " + str(image_width) + "\n")
#     print(args["image_size"])
    index = 1
    if image_height >= args["image_size"] and image_width >= args["image_size"]:
        for pos_y in range(0, image_height - args["image_size"] + 1, args["step"]):
            for pos_x in range(0, image_width - args["image_size"] + 1, args["step"]):
#                 print("HELLO1")
                # Crop
                crop_image = image[pos_y: pos_y + args["image_size"], pos_x:pos_x + args["image_size"], ...]
#                 print("hello2")
                crop_image = np.ascontiguousarray(crop_image)
#                 print("HELLO3")
                # Save image
#                 print(f"{args['output_dir']}/{image_file_name.split('.')[-2]}_{index:04d}.{image_file_name.split('.')[-1]}")
                cv2.imwrite(f"{args['output_dir']}/{image_file_name.split('.')[-2]}_{index:04d}.{image_file_name.split('.')[-1]}", crop_image)

                index += 1
    print("done")
    
    
if __name__ == "__main__":
#     --images_dir ../data/ImageNet/original --output_dir ../data/ImageNet/SRGAN/train --image_size 128 --step 64 --num_workers 16"
#     parser = argparse.ArgumentParser(description="Prepare database scripts.")
#     parser.add_argument("--images_dir", type=str, help="Path to input image directory.")
#     parser.add_argument("--output_dir", type=str, help="Path to generator image directory.")
#     parser.add_argument("--image_size", type=int, help="Low-resolution image size from raw image.")
#     parser.add_argument("--step", type=int, help="Crop image similar to sliding window.")
#     parser.add_argument("--num_workers", type=int, help="How many threads to open at the same time.")
#     args = parser.parse_args()
    args={"images_dir":"/kaggle/input/div2-k-dataset-for-super-resolution","output_dir":"/kaggle/working/div2-k-dataset-for-super-resolution/new_image","image_size":128,"step":64,"num_workers":16}
    main(args)
