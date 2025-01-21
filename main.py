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
    n = len(images)
    height, width = images[0].shape[:2]

    max_mag = np.zeros((height, width), dtype=np.uint8)
    map = np.full((height, width), n // 2, dtype=np.uint8)

    for i, img in enumerate(images):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #gradient magnitude
        grad_x = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)

        # Blur the magnitude image
        magnitude_8u = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        blurred_magnitude = cv2.medianBlur(magnitude_8u, 11)

        # Update max_mag and map
        mask = blurred_magnitude > max_mag
        max_mag[mask] = blurred_magnitude[mask]
        map[mask] = i
    
    sigma = min(width, height) / 24
    map_blurred = cv2.GaussianBlur(map.astype(np.float32), (0, 0), sigma)

    # inerpola varias imgs
    hdr = np.zeros_like(images[0], dtype=np.float32)
    for y in range(height):
        for x in range(width):
            i = int(map_blurred[y, x])  # qual imagem
            alpha = map_blurred[y, x] - i #ex. 2.5 - 2 = 0.5
            if i < n - 1:   #se não for a última imagem
                hdr[y, x] = images[i][y, x] * (1 - alpha) + images[i + 1][y, x] * alpha
            else:
                hdr[y, x] = images[i][y, x] * (1 - alpha) + images[i - 1][y, x] * alpha

    hdr = cv2.normalize(hdr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    map_blurred = cv2.normalize(map_blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
    return map_blurred, hdr  

def create_guided_filter(images, map):
    guided_filter = np.zeros_like(images[0])
    
    for img in images:
        mag = cv2.Laplacian(img, cv2.CV_64F)
        mag = np.absolute(mag)
        guided_filter = np.maximum(guided_filter, mag)
     
    return guided_filter

def main():
    parser = argparse.ArgumentParser(description='Create an HDR image from a directory of images.')
    parser.add_argument('-d', '--directory', required=True, help='Directory containing images')
    parser.add_argument('-o', '--output', required=True, help='Output HDR image file name')
    
    args = parser.parse_args()
    
    images = [img for img, _ in open_all_images(args.directory)]
    
    # Sort images by brightness
    images.sort(key=lambda img: np.average(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
    
    for index, (img, filename) in enumerate(open_all_images(args.directory)):
        print(f"Index: {index}, Image Name: {filename}")
    
    map, hdr = create_hdr_image(images)

    cv2.imwrite(args.output+"_hdr.png", hdr)

    cv2.imwrite(args.output+"_map.png", map)

if __name__ == '__main__':
    main()