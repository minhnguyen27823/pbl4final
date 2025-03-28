import h5py
import numpy as np
import os

# Đường dẫn đến file h5 gốc
input_h5_file = "features.h5"

# Thư mục đích để lưu các file đã tách ra
output_folder = "feaetures_extracted_data"
os.makedirs(output_folder, exist_ok=True)

# Mở file HDF5 và đọc toàn bộ dữ liệu
with h5py.File(input_h5_file, "r") as f:
    for key in f.keys():
        data = f[key][()]  # Đọc dữ liệu vào mảng NumPy
        
        # Lưu dưới dạng file .npy
        np.save(os.path.join(output_folder, f"{key}.npy"), data)
        
        # Nếu muốn lưu thành file .csv
        np.savetxt(os.path.join(output_folder, f"{key}.csv"), data, delimiter=",")

print(f"Dữ liệu từ {input_h5_file} đã được trích xuất vào thư mục {output_folder}")
