from __future__ import print_function
import argparse
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO
import base64
import sys


#import sys
#sys.argv=['']
#del sys

st.set_page_config(layout="wide", page_title="Super-Resolution")

st.write("## Super-Resolution")
st.write(
    ":dog: Разработано специально для дисциплины Архитектура систем ИИ. Строкова Настя, P4140"
)
st.sidebar.write("## Загрузка изображений :gear:")


# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


def fix_image(upload):
    image = Image.open(upload)
    col1.write("Исходное изображение :camera:")
    col1.image(image)

    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--input', type=str, required=False, default=upload, help='input image to use')
    parser.add_argument('--model', type=str, default=(r'C:\Users\n.strokova\Pictures\super-resolution\models\SUB_model_path.pth'), help='model file to use')
    parser.add_argument('--output', type=str, default='test.jpg', help='where to save the output image')
    args = parser.parse_args()
    print(args)

    GPU_IN_USE = torch.cuda.is_available()
    img = Image.open(args.input).convert('YCbCr')
    y, cb, cr = img.split()

    device = torch.device('cuda' if GPU_IN_USE else 'cpu')
    model = torch.load(args.model, map_location=lambda storage, loc: storage)
    model = model.to(device)
    data = (ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
    data = data.to(device)

    if GPU_IN_USE:
        cudnn.benchmark = True

    out = model(data)
    out = out.cpu()
    out_img_y = out.data[0].numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

    fixed = out_img
    col2.write("Преобразованное изображение :wrench:")
    col2.image(fixed)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Скачать преобразованное изображение", convert_image(fixed), "fixed.png", "image/png")


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Обзор изображений", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    fix_image(upload=my_upload)
else:
    fix_image(r'C:\Users\n.strokova\Pictures\super-resolution\models\Dog_Color.jpg')

print(sys.exit())
