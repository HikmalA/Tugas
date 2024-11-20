# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 08:45:26 2024

@author: Lenovo
"""

import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Fungsi untuk menghitung SSIM antara dua gambar
def calculate_ssim(imageA, imageB):
    # Mengubah gambar menjadi grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # Menghitung SSIM
    score, diff = ssim(grayA, grayB, full=True)
    print(f"SSIM: {score:.4f}")

    # Mengubah perbedaan (diff) menjadi tipe data uint8
    diff = (diff * 255).astype("uint8")

    # Menampilkan gambar asli dan hasil perbedaan
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Gambar 1")
    ax[0].axis("off")

    ax[1].imshow(cv2.cvtColor(imageB, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Gambar 2")
    ax[1].axis("off")

    ax[2].imshow(diff, cmap='gray')
    ax[2].set_title("Perbedaan")
    ax[2].axis("off")

    plt.show()

    return score

# Membaca dua gambar
imageA = cv2.imread("D:/ss/winter.jpg")
imageB = cv2.imread("D:/ss/winter.jpg")

# Memastikan gambar tidak kosong
if imageA is None or imageB is None:
    print("Gambar tidak ditemukan. Pastikan file gambar sudah benar.")
else:
    # Menghitung SSIM
    score = calculate_ssim(imageA, imageB)