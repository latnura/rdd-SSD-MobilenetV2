import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0

import torch
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys
import time
import numpy as np
from datetime import datetime
import signal
import csv  # Untuk menulis hasil ke CSV

# ========== SETUP ==========
timer = Timer()
net_type = "mb2-ssd-lite"
model_path = "./models_scheduler_plateau/512_bs16_lr_0.000304/mb2-ssd-lite-Epoch-150-Loss-2.7006.pth"
label_path = "labels4.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"INFO: Using device: {device}\n")

if not torch.cuda.is_available():
    print("WARNING: CUDA is not available, running on CPU. Performance will be slow.")

# ========== LOAD MODEL ==========
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

# Fixed color mapping for each class (pastel tones in BGR)
label_colors = {
    'L00': (203, 192, 255),  # pink pastel
    'R02': (221, 160, 221),  # purple pastel
    'R03': (255, 255, 128),  # cyan pastel
}
default_color = (255, 255, 255)

if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(num_classes, is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(num_classes, is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(num_classes, is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(num_classes, is_test=True)
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(num_classes, is_test=True)
else:
    print("The net type is wrong.")
    sys.exit(1)

net.load(model_path)
net.to(device)
net.eval()

if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200, device=device)
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200, device=device)
elif net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200, device=device)
elif net_type == 'mb2-ssd-lite':
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200, device=device)
elif net_type == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200, device=device)
else:
    print("The net type is wrong.")
    sys.exit(1)

print("Model loaded!\nStarting detection...\n")

# ========== VIDEO SETUP ==========
video_path = "harmoni_10kph.webm"
#video_path = 0
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Gagal membuka video: {video_path}")
    sys.exit(1)

ret, one_image = cap.read()
if not ret or one_image is None:
    print(f"❌ Gagal membaca frame pertama.")
    sys.exit(1)

print("Input Shape: ", one_image.shape)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
# Pastikan folder videos ada
os.makedirs("videos", exist_ok=True)
output_path = f"videos/output-{timestamp}.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (one_image.shape[1], one_image.shape[0]))

frame_cnt = 0
inference_times = []

# ========== CLEANUP & CSV HANDLER ==========
def summarize():
    if inference_times:
        # Gunakan inference_times langsung, karena 5 frame awal sudah dibuang saat di dalam loop
        stable_times = inference_times 
        if stable_times:
            avg_time = sum(stable_times) / len(stable_times)
            avg_fps = 1 / avg_time if avg_time > 0 else 0
            
            print("\n=== Inference Summary ===")
            print(f"Total Frames Processed: {frame_cnt}")
            print(f"Average Inference Time: {avg_time*1000:.2f} ms")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Output video: {output_path}")
            
            # --- SIMPAN KE CSV ---
            csv_file = "inference_summary.csv"
            file_exists = os.path.isfile(csv_file)
            
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                # Tulis header jika file baru dibuat
                if not file_exists:
                    writer.writerow(["Timestamp", "Video Input", "Model", "Total Frames", "Avg Inference Time (ms)", "Avg FPS", "Output Video"])
                
                # Tulis data rata-rata video yang baru selesai
                writer.writerow([
                    timestamp, 
                    video_path, 
                    net_type, 
                    frame_cnt, 
                    f"{avg_time*1000:.2f}", 
                    f"{avg_fps:.2f}", 
                    output_path
                ])
            print(f"✅ Data hasil inferensi berhasil disimpan ke {csv_file}")
            
    print("Check the result!")

def handle_sigint(sig, frame):
    print("\nInterrupted. Cleaning up...")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    summarize()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)

# ========== INFERENCE LOOP ==========
while cap.isOpened():
    ret, orig_image = cap.read()
    if not ret or orig_image is None:
        break

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    frame_cnt += 1

    # SINKRONISASI GPU SEBELUM INFERENSI
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    timer.start()
    
    # PROSES PREDIKSI OLEH MODEL
    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    
    # SINKRONISASI GPU SETELAH INFERENSI
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    interval = timer.end()

    # Buang 5 frame pertama untuk pemanasan (warm-up), sisanya simpan
    if frame_cnt > 5:
        inference_times.append(interval)

    print('Time: {:.4f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)), "Frame: ", frame_cnt)
    fps = 1 / interval if interval > 0 else 0

    # ===== VISUALISASI BOUNDING BOX =====
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label_name = class_names[labels[i]]
        label_text = f"{label_name}: {probs[i]:.2f}"

        # Ambil warna berdasarkan nama kelas
        i_color = label_colors.get(label_name, default_color)

        # Draw bounding box
        cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), i_color, 3)

        # Draw label background
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
        top_left = (int(box[0]), int(box[1]) - text_height - 4)
        bottom_right = (int(box[0]) + text_width, int(box[1]))
        cv2.rectangle(orig_image, top_left, bottom_right, i_color, thickness=-1)

        # Draw label text
        cv2.putText(orig_image, label_text, (int(box[0]), int(box[1]) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    # Tambahkan teks informasi model dan FPS secara real-time ke video
    cv2.putText(orig_image, "SSD MobileNetV2", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.putText(orig_image, f"FPS: {fps:.2f}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    cv2.putText(orig_image, f"Inference: {interval*1000:.2f} ms", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    cv2.imshow("Detection", orig_image)
    out.write(orig_image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

# PEMBERSIHAN DAN EKSEKUSI SUMMARIZE() SAAT VIDEO SELESAI
cap.release()
out.release()
cv2.destroyAllWindows()
summarize()