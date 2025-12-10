import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.metrics import structural_similarity as ssim

import tkinter as tk
from tkinter import filedialog, ttk, messagebox

from PIL import Image, ImageTk

# ================== 1) Griye çevirme ==================

def load_image_gray(path):
    # Unicode path'ler için workaround
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"GÖRÜNTÜ OKUNAMADI: {path}")

    img = img.astype(np.float32) / 255.0
    return img


# ================== 2) Gürültü fonksiyonları ==================

def add_gaussian_noise(image, sigma=0.1):
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image + noise
    noisy = np.clip(noisy, 0.0, 1.0)
    return noisy

def add_salt_pepper_noise(image, amount=0.01, salt_vs_pepper=0.5):
    noisy = image.copy()
    num_pixels = image.size
    num_salt = int(num_pixels * amount * salt_vs_pepper)
    num_pepper = int(num_pixels * amount * (1.0 - salt_vs_pepper))

    coords_salt = (
        np.random.randint(0, image.shape[0], num_salt),
        np.random.randint(0, image.shape[1], num_salt)
    )
    noisy[coords_salt] = 1.0

    coords_pepper = (
        np.random.randint(0, image.shape[0], num_pepper),
        np.random.randint(0, image.shape[1], num_pepper)
    )
    noisy[coords_pepper] = 0.0

    return noisy

def add_speckle_noise(image, std=0.1):
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    noisy = image + image * noise
    noisy = np.clip(noisy, 0.0, 1.0)
    return noisy

# ================== 3) Filtre fonksiyonları ==================

def apply_mean_filter(image, ksize=5):
    img_255 = (image * 255).astype(np.uint8)
    filtered_255 = cv2.blur(img_255, (ksize, ksize))
    filtered = filtered_255.astype(np.float32) / 255.0
    return filtered

def apply_median_filter(image, ksize=5):
    img_255 = (image * 255).astype(np.uint8)
    filtered_255 = cv2.medianBlur(img_255, ksize)
    filtered = filtered_255.astype(np.float32) / 255.0
    return filtered

def apply_gaussian_filter(image, ksize=5, sigma=1.0):
    img_255 = (image * 255).astype(np.uint8)
    filtered_255 = cv2.GaussianBlur(img_255, (ksize, ksize), sigmaX=sigma)
    filtered = filtered_255.astype(np.float32) / 255.0
    return filtered

def apply_bilateral_filter(image, d=25, sigma_color=75, sigma_space=75):
    img_255 = (image * 255).astype(np.uint8)
    filtered_255 = cv2.bilateralFilter(img_255, d, sigma_color, sigma_space)
    filtered = filtered_255.astype(np.float32) / 255.0
    return filtered

# ================== 4) Metrik ve yardımcı fonksiyonlar ==================

def mse(original, denoised):
    return np.mean((original - denoised) ** 2)

def psnr(original, denoised, max_pixel=1.0):
    m = mse(original, denoised)
    if m == 0:
        return float('inf')
    return 10 * np.log10((max_pixel ** 2) / m)

def compute_ssim(original, denoised):
    return ssim(original, denoised, data_range=1.0)

def apply_with_time(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    runtime_ms = (end - start) * 1000
    return result, runtime_ms

def show_images(images, titles, cmap="gray"):
    n = len(images)
    plt.figure(figsize=(4*n, 4))
    for i, (img_, title) in enumerate(zip(images, titles), 1):
        plt.subplot(1, n, i)
        plt.imshow(img_, cmap=cmap, vmin=0, vmax=1)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def plot_metrics(names, mses, psnrs, ssims, times, noise_info):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Performans Analizi: {noise_info}", fontsize=14, fontweight='bold')

    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']

    # 1. Grafik: MSE (Düşük olan iyidir)
    axs[0, 0].bar(names, mses, color=colors[0])
    axs[0, 0].set_title('MSE (Hata Kareler Ort.) - Düşük İyidir')
    axs[0, 0].set_ylabel('Değer')
    
    # 2. Grafik: PSNR (Yüksek olan iyidir)
    axs[0, 1].bar(names, psnrs, color=colors[1])
    axs[0, 1].set_title('PSNR (dB) - Yüksek İyidir')
    
    # 3. Grafik: SSIM (1.0'a yakın olan iyidir)
    axs[1, 0].bar(names, ssims, color=colors[2])
    axs[1, 0].set_title('SSIM (Yapısal Benzerlik) - 1.0 İyidir')
    axs[1, 0].set_ylim(0, 1.1) # SSIM max 1 olur

    # 4. Grafik: Süre (Düşük olan iyidir)
    axs[1, 1].bar(names, times, color=colors[3])
    axs[1, 1].set_title('İşlem Süresi (ms) - Düşük İyidir')
    axs[1, 1].set_ylabel('Milisaniye')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# ================== 5) Basit arayüz mantığı ==================

current_image = None
current_path = None

def select_image():
    global current_image, current_path
    
    path = filedialog.askopenfilename(
        title="Görsel seç",
        filetypes=[("Görüntü dosyaları", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")]
    )
    if not path:
        return
        
    try:
        # 1. Dosyayı ham veri olarak oku (Türkçe karakter sorunu olmasın diye)
        file_data = np.fromfile(path, dtype=np.uint8)
        
        # 2. OpenCV ile RENKLİ (BGR) olarak decode et
        # IMREAD_COLOR bayrağı ile yüklüyoruz ki ekranda renkli görelim
        img_bgr = cv2.imdecode(file_data, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise ValueError(f"GÖRÜNTÜ OKUNAMADI: {path}")

        current_path = path

        # --- ARAYÜZDE GÖSTERME (RENKLİ) ---
        # OpenCV BGR kullanır, Arayüz (Pillow) RGB ister. Dönüştürüyoruz:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Pillow imajına çevir
        pil_image = Image.fromarray(img_rgb)
        
        # Boyutlandırma (Görüntü çok büyükse arayüzü kaplamasın)
        base_height = 200
        w_percent = (base_height / float(pil_image.size[1]))
        w_size = int((float(pil_image.size[0]) * float(w_percent)))
        pil_image = pil_image.resize((w_size, base_height), Image.Resampling.LANCZOS)
        
        # Tkinter uyumlu yap ve etikete koy
        tk_image = ImageTk.PhotoImage(pil_image)
        image_preview_label.config(image=tk_image, text="") 
        image_preview_label.image = tk_image 

        # --- ARKA PLAN İÇİN HAZIRLIK (GRİ) ---
        # Algoritmalar gri tonlama ve 0-1 float aralığı bekliyor.
        # Bu yüzden işlem yapılacak 'current_image' değişkenini burada griye çeviriyoruz.
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        current_image = img_gray.astype(np.float32) / 255.0
        
        status_label.config(text="")

    except Exception as e:
        status_label.config(text=f"HATA: {e}")
        print(e)

def apply_filters():
    if current_image is None:
        status_label.config(text="Önce görsel seç.")
        return

    noise_type = noise_combo.get()
    level = level_scale.get()
    
    # Kernel boyutu okuma
    try:
        ksize = int(kernel_combo.get())
    except:
        ksize = 3

    # Gürültü ekleme
    if noise_type == "Gaussian":
        sigma = level / 100.0
        noisy = add_gaussian_noise(current_image, sigma=sigma)
        noise_name = f"Gaussian (σ={sigma:.2f})"
    elif noise_type == "Salt & Pepper":
        amount = level / 100.0
        noisy = add_salt_pepper_noise(current_image, amount=amount, salt_vs_pepper=0.5)
        noise_name = f"Salt & Pepper (amount={amount:.2f})"
    elif noise_type == "Speckle":
        std = level / 100.0
        noisy = add_speckle_noise(current_image, std=std)
        noise_name = f"Speckle (std={std:.2f})"
    else:
        status_label.config(text="Gürültü tipi seç.")
        return

    # Filtreleri uygula
    den_mean,   t_mean   = apply_with_time(apply_mean_filter,     noisy, ksize)
    den_median, t_median = apply_with_time(apply_median_filter,   noisy, ksize)
    den_gauss,  t_gauss  = apply_with_time(apply_gaussian_filter, noisy, ksize, 1.0)
    den_bilat,  t_bilat  = apply_with_time(apply_bilateral_filter, noisy, ksize, 75, 75)

    # --- GRAFİK İÇİN VERİ TOPLAMA ---
    filter_names = ["Mean", "Median", "Gaussian", "Bilateral"]
    # Listeler oluşturuyoruz
    results_mse = []
    results_psnr = []
    results_ssim = []
    results_time = []
    
    results_lines = [f"{noise_name} için metrikler:\n"]

    # Döngü ile hem metin hazırla hem listeleri doldur
    denoised_images = [den_mean, den_median, den_gauss, den_bilat]
    times = [t_mean, t_median, t_gauss, t_bilat]

    for name, img_den, t in zip(filter_names, denoised_images, times):
        m = mse(current_image, img_den)
        p = psnr(current_image, img_den)
        s = compute_ssim(current_image, img_den)

        # Listelere ekle (Grafik için)
        results_mse.append(m)
        results_psnr.append(p)
        results_ssim.append(s)
        results_time.append(t)

        # Metin hazırla (Popup için)
        line = (
            f"{name} filtresi:\n"
            f"  MSE  : {m:.6f}\n"
            f"  PSNR : {p:.2f} dB\n"
            f"  SSIM : {s:.4f}\n"
            f"  Time : {t:.3f} ms\n"
        )
        results_lines.append(line)
        print(line)

    summary_text = "\n".join(results_lines)
    
    # 1. Metrik Sonuçları Popup
    messagebox.showinfo("Metrik Sonuçları", summary_text)

    # 2. Filtrelenmiş Görüntüleri Göster
    show_images(
        [current_image, noisy, den_mean, den_median, den_gauss, den_bilat],
        ["Orijinal", noise_name, "Mean", "Median", "Gaussian", "Bilateral"]
    )
    
    # 3. Performans Grafiklerini Çiz (YENİ)
    plot_metrics(filter_names, results_mse, results_psnr, results_ssim, results_time, noise_name)

    status_label.config(text=f"{noise_name} uygulandı ve grafikler çizildi.")


# ================== 6) Tkinter arayüzü ==================

root = tk.Tk()
root.title("Görüntü Gürültü ve Filtre Uygulaması")
root.geometry("600x550")

root.columnconfigure(0, weight=1)
root.columnconfigure(5, weight=1)

for i in range(1, 5):
    root.columnconfigure(i, weight=0)

for i in range(7): 
    root.rowconfigure(i, weight=1)

my_font = ("Arial", 10, "bold")

image_preview_label = tk.Label(root, text="HENÜZ GÖRSEL SEÇİLMEDİ", bg="#f0f0f0", relief="sunken", font=my_font, fg="red" )
image_preview_label.grid(row=0, column=0, columnspan=6, padx=20, pady=20, sticky="nsew")

btn_select = tk.Button(root, text="GÖRSEL SEÇ", command=select_image, bg="#e1f5fe", font=my_font, width=20)
btn_select.grid(row=1, column=0, columnspan=6, padx=10, pady=10)

tk.Label(root, text="GÜRÜLTÜ TİPİ:", font=my_font).grid(row=2, column=1, padx=(0, 5), sticky="e")
noise_combo = ttk.Combobox(root, values=["Gaussian", "Salt & Pepper", "Speckle"], font=("Arial", 10), width=12, state="readonly")
noise_combo.current(0)
noise_combo.grid(row=2, column=2, padx=(0, 20), sticky="w")

tk.Label(root, text="KERNEL BOYUTU:", font=my_font).grid(row=2, column=3, padx=(0, 5), sticky="e")
kernel_combo = ttk.Combobox(root, values=["3", "5", "7", "9", "11"], font=("Arial", 10), width=5, state="readonly")
kernel_combo.current(0)
kernel_combo.grid(row=2, column=4, padx=0, sticky="w")

tk.Label(root, text="GÜRÜLTÜ SEVİYESİ:", font=my_font).grid(row=3, column=0, columnspan=6, pady=(10, 0))
level_scale = tk.Scale(root, from_=1, to=50, orient=tk.HORIZONTAL, length=300)
level_scale.set(25)
level_scale.grid(row=4, column=0, columnspan=6, pady=(0, 10))

btn_apply = tk.Button(root, text="GÜRÜLTÜ EKLE + FİLTRELE", command=apply_filters, bg="#e1f5fe", font=my_font, width=30)
btn_apply.grid(row=5, column=0, columnspan=6, padx=10, pady=10)

status_label = tk.Label(root, text="LÜTFEN GÖRSEL SEÇİNİZ", fg="red", font=("Arial", 9, "bold"))
status_label.grid(row=6, column=0, columnspan=6, padx=10, pady=10)

root.mainloop()
