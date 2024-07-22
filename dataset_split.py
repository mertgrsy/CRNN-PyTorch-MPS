import os
import shutil
import random
from glob import glob

def find_files(directory, extensions):
    files = []
    for extension in extensions:
        files.extend(glob(os.path.join(directory, f'*.{extension}')))
    return files

def split_dataset(folder_path, train_ratio=0.9):
    # Desteklenen görsel dosya uzantıları
    extensions = ['jpg', 'jpeg', 'png', 'PNG', 'JPEG', 'JPG']

    # Görsellerin yollarını bul
    images = find_files(folder_path, extensions)
    labels = [os.path.splitext(image)[0] + '.def' for image in images]

    # Dosyaları karıştır ve ayır
    combined = list(zip(images, labels))
    random.shuffle(combined)

    num_train = int(len(combined) * train_ratio)
    train_files = combined[:num_train]
    val_files = combined[num_train:]

    # Eğitim ve doğrulama klasör yolları
    train_folder = os.path.join(folder_path, 'train')
    val_folder = os.path.join(folder_path, 'val')

    # Klasörleri oluştur
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # Dosyaları ilgili klasörlere taşı
    for files, target_folder in [(train_files, train_folder), (val_files, val_folder)]:
        for image, label in files:
            shutil.move(image, os.path.join(target_folder, os.path.basename(image)))
            if os.path.exists(label):  # Eğer etiket dosyası varsa taşı
                shutil.move(label, os.path.join(target_folder, os.path.basename(label)))

# Kullanım:
split_dataset('/Users/mert/Downloads/OCR_Data/')
