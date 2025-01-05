import argparse
import os
import cv2
import numpy as np
import sys

def open_image(img_name):
    img = cv2.imread(img_name, cv2.IMREAD_COLOR)
    if img is None:
        print('Erro abrindo a imagem.\n')
        sys.exit()
    return img

def open_all_images(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".bmp"):
            img_path = os.path.join(folder, filename)
            img = open_image(img_path)
            if img is not None:
                images.append((img, filename))
    return images

def create_hdr_image(images):
    # Convert images to float32
    images = [img.astype(np.float32) for img in images]
    
    # Create an empty array for the HDR image
    hdr_image = np.zeros_like(images[0])
    
    # Simple HDR algorithm: average the pixel values
    for img in images:
        hdr_image += img
    hdr_image /= len(images)
    
    # Convert back to uint8
    hdr_image = np.clip(hdr_image, 0, 255).astype(np.uint8)
    
    return hdr_image

def main():
    parser = argparse.ArgumentParser(description='Create an HDR image from a directory of images.')
    parser.add_argument('-d', '--directory', required=True, help='Directory containing images')
    parser.add_argument('-o', '--output', required=True, help='Output HDR image file name')
    
    args = parser.parse_args()
    
    images = [img for img, _ in open_all_images(args.directory)]
    
    hdr_image = create_hdr_image(images)
    
    cv2.imwrite(args.output, hdr_image)

if __name__ == '__main__':
    main()