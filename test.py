import cv2
import numpy as np

# GPU kullanımı kontrolü
if cv2.cuda.getCudaEnabledDeviceCount() == 0:
    print("CUDA destekli bir GPU bulunamadı!")
else:
    print("CUDA destekli bir GPU bulundu.")

# Örnek bir gri görüntü oluştur
cpu_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

# GpuMat'e yükleme
gpu_img = cv2.cuda_GpuMat()
gpu_img.upload(cpu_img)

# Tekrar CPU'ya çekme
downloaded_img = gpu_img.download()

# Karşılaştırma
if np.array_equal(cpu_img, downloaded_img):
    print("cv.cuda.GpuMat() başarıyla çalışıyor!")
else:
    print("Bir şeyler ters gitti.")
