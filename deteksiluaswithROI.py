# Import library yang dibutuhkan
import cv2
import csv
import os
import pynmea2
import serial
import threading
import math
import time
import requests
import json
import torch
import sys
import numpy as np
from datetime import datetime
from geopy.distance import geodesic

# =================================================================================
# --- IMPORT ARSITEKTUR SSD (WAJIB ADA FOLDER 'vision') ---
# =================================================================================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
except ImportError:
    print("\n❌ CRITICAL ERROR: Folder 'vision' tidak ditemukan!")
    sys.exit(1)

# =================================================================================
# --- KONFIGURASI ---
# =================================================================================
CONFIG = {
    "app": {
        "use_camera": False,
        "camera_index": 0,
        "cooldown_seconds": 1,
        "frame_width": 800,
        "frame_height": 600,  
    },
    "roi": {
        "y_upper": 104,
        "y_lower": 396
    },
    "camera_params": {
        "H_meter": 1.05,
        "theta_deg": 35.0,
        "alpha_vfov_deg": 61.9,
        "f_calib": 15.0      
    },
    "model": {
        "path": "./models/mb2-ssd-lite.pth", 
        "class_names": ["BACKGROUND", "Potholes", "Linear Crack", "Alligator Crack"],
        "video_path": "./data/35degree.webm",
        "confidence_threshold" : 0.5 
    },
    "api": { 
        "url": "https://domain-anda.com/upload.php",
        "key": "KODE_API_RAHASIA_ANDA"
    },
    "gps": {
        "port": "/dev/ttyACM0",
        "baudrate": 115200,
    },
    "file_paths": {
        "save_dir": "deteksi_kerusakan",
        "uploads_dir": "deteksi_kerusakan/uploads",
        "gps_log_file": "deteksi_kerusakan/log_gps_aktif.csv",
    },
    "checkpoint": {
        "distance_meters": 100
    }
}

# =================================================================================
# --- FUNGSI LOAD MODEL ---
# =================================================================================
def load_ssd_model(model_path, device):
    print(f"⏳ Building Model Architecture & Loading Weights from {model_path}...")
    class_names = CONFIG["model"]["class_names"]
    num_classes = len(class_names) 
    net = create_mobilenetv2_ssd_lite(num_classes, is_test=True)
    try:
        net.load_state_dict(torch.load(model_path, map_location=device))
        predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200, device=device)
        print("✅ Model Loaded Successfully!")
        return predictor
    except Exception as e:
        print(f"❌ Error Loading Model: {e}")
        sys.exit(1)

# =================================================================================
# --- KAMERA & API ---
# =================================================================================
def setup_camera_controls():
    if not CONFIG["app"]["use_camera"]: return
    camera_device = f"/dev/video{CONFIG['app']['camera_index']}"
    os.system(f"v4l2-ctl -d {camera_device} -c exposure_auto=3")

def kirim_rekap_via_api(hasil_deteksi):
    if not hasil_deteksi: return
    api_url = "https://revalyze.xyz/simpan_rekap.php"
    payload = {'api_key': CONFIG["api"]["key"], 'data': json.dumps(hasil_deteksi)}
    try: requests.post(api_url, data=payload, timeout=5)
    except: pass

def send_data_to_api(data_payload, image_path):
    try:
        with open(image_path, 'rb') as image_file:
            files = {'foto': image_file}
            requests.post(CONFIG["api"]["url"], data=data_payload, files=files, timeout=30)
            print(f"📦 Data & Foto terkirim!")
    except Exception as e:
        print(f"❌ API Error: {e}")

def process_and_save_data(frame_to_save, lat, lon, jalan, sta, km, jenis, luas):
    try:
        event_dt = datetime.now()
        file_name = f"foto_{event_dt.strftime('%Y%m%d_%H%M%S')}.jpg"
        save_path_full = os.path.join(CONFIG["file_paths"]["uploads_dir"], file_name)
        cv2.imwrite(save_path_full, frame_to_save)
        
        data_untuk_api = {
            'api_key': CONFIG["api"]["key"], 'latitude': lat, 'longitude': lon,
            'nama_jalan': jalan, 'sta': sta, 'Km': km, 'jenis': jenis,
            'luas': luas, 'waktu_deteksi': event_dt.isoformat()
        }
        send_data_to_api(data_untuk_api, save_path_full)
    except Exception as e:
        print(f"❌ Save Error: {e}")

def draw_overlays(frame, detections, data_info):
    """
    Menampilkan Box Deteksi dan Luas untuk SEMUA kelas.
    """
    luas_detected = 0.0
    y_upper = CONFIG["roi"]["y_upper"]
    y_lower = CONFIG["roi"]["y_lower"]

    # --- Gambar Garis ROI ---
    cv2.line(frame, (0, y_upper), (CONFIG["app"]["frame_width"], y_upper), (0, 0, 255), 2)
    cv2.line(frame, (0, y_lower), (CONFIG["app"]["frame_width"], y_lower), (0, 0, 255), 2)
    cv2.putText(frame, "ROI", (10, (y_upper + y_lower) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # --- Gambar Box Deteksi ---
    for (box, label_name, score) in detections:
        x1, y1, x2, y2 = map(int, box)
        
        if y1 >= y_upper and y2 <= y_lower:
            # Tampilkan Label & Confidence
            label_text = f"{label_name}: {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # --- [MODIFIKASI] HITUNG LUAS UNTUK SEMUA JENIS KERUSAKAN ---
            # Kecuali 'BACKGROUND' jika ada
            if label_name != "BACKGROUND":
                H = CONFIG["camera_params"]["H_meter"]
                theta_deg = CONFIG["camera_params"]["theta_deg"]
                alpha_vfov_deg = CONFIG["camera_params"]["alpha_vfov_deg"]
                f_calib = CONFIG["camera_params"]["f_calib"]
                frame_width_px = CONFIG["app"]["frame_width"]

                theta_rad = math.radians(theta_deg)
                alpha_rad = math.radians(alpha_vfov_deg)
                
                rx = (2 * H * math.cos(theta_rad) * math.tan(alpha_rad / 2 + theta_rad)) / frame_width_px
                ry = (H * math.tan(theta_rad) * math.sin(theta_rad)) / f_calib
                
                w_px, h_px = x2 - x1, y2 - y1
                A_real_m2 = abs((w_px * h_px) * rx * ry)
                
                area_text = f"Luas: {A_real_m2:.3f} m2"
                cv2.putText(frame, area_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Simpan luas terbesar yang terdeteksi di frame ini
                if A_real_m2 > luas_detected:
                    luas_detected = A_real_m2

    # --- Info Text (Statistik) ---
    fw = frame.shape[1]
    cv2.putText(frame, f"Jalan: {data_info['jalan']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"KM: {data_info['km']} | STA: {data_info['sta']}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(frame, f"FPS: {data_info['fps']:.2f}", (fw - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Infer: {data_info['inference_time']:.1f} ms", (fw - 250, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Lat: {data_info['lat']:.6f}", (fw - 250, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Lon: {data_info['lon']:.6f}", (fw - 250, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return luas_detected

def handle_distance_checkpoint(current_coords, last_checkpoint_coords, threshold_meters):
    checkpoint_crossed = False
    if last_checkpoint_coords is None:
        return current_coords, checkpoint_crossed
    distance_in_meters = geodesic(last_checkpoint_coords, current_coords).meters
    if distance_in_meters > threshold_meters:
        print(f"🏁 SYSTEM : Checkpoint {threshold_meters}m Reached.")
        last_checkpoint_coords = current_coords
        checkpoint_crossed = True
    return last_checkpoint_coords, checkpoint_crossed

# =================================================================================
# --- GPS MANAGER ---
# =================================================================================
class GPSManager:
    def __init__(self, port, baudrate):
        self.port, self.baudrate = port, baudrate; self.serial_port = None; self.latitude, self.longitude = 0.0, 0.0; self.is_running = False; self.lock = threading.Lock()
    def _connect(self):
        try: self.serial_port = serial.Serial(self.port, baudrate=self.baudrate, timeout=1); return True
        except: return False
    def _read_data(self):
        while self.is_running:
            if not self.serial_port or not self.serial_port.is_open:
                if not self._connect(): time.sleep(5); continue
            try:
                line = self.serial_port.readline().decode('ascii', errors='replace')
                if line.startswith(('$GPGGA', '$GNGLL', '$GPRMC')):
                    msg = pynmea2.parse(line)
                    if hasattr(msg, 'latitude') and msg.latitude != 0.0:
                        with self.lock: self.latitude, self.longitude = msg.latitude, msg.longitude
            except: continue
    def start(self):
        if self.is_running: return
        self.is_running = True; threading.Thread(target=self._read_data, daemon=True).start()
    def stop(self): self.is_running = False
    def get_coordinates(self):
        with self.lock: return self.latitude, self.longitude, "N/A"

# =================================================================================
# --- MAIN PROGRAM ---
# =================================================================================
def main():
    os.makedirs(CONFIG["file_paths"]["uploads_dir"], exist_ok=True)
    setup_camera_controls()

    print("\n=========== PENS-REVALYZE SYSTEM (ALL CLASSES) ============")
    jalan_input = input("Nama Jalan: ")
    km_survey_input = input("KM Survey (contoh: KM 14): ")
    
    gps_manager = GPSManager(port=CONFIG["gps"]["port"], baudrate=CONFIG["gps"]["baudrate"])
    gps_manager.start()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Running on: {device}")

    predictor = load_ssd_model(CONFIG["model"]["path"], device)
    
    video_source = CONFIG["app"]["camera_index"] if CONFIG["app"]["use_camera"] else CONFIG["model"]["video_path"]
    cap = cv2.VideoCapture(video_source)
    
    target_w, target_h = CONFIG["app"]["frame_width"], CONFIG["app"]["frame_height"]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)

    # --- TRACKING VARS ---
    is_ready_to_capture, last_capture_time, prev_frame_time = True, 0.0, 0.0
    sta_awal, panjang_segmen, last_checkpoint_coords = 0, CONFIG["checkpoint"]["distance_meters"], None
    
    tracked_objects = {} # Renamed from tracked_potholes
    next_object_id = 0
    damage_count_per_segment = 0 # Renamed
    total_damage_count = 0 # Renamed
    hasil_semua_segmen = []
    class_names = CONFIG["model"]["class_names"]

    with open(CONFIG["file_paths"]["gps_log_file"], 'w', newline='', encoding='utf-8') as gps_log_file:
        gps_writer = csv.writer(gps_log_file)
        gps_writer.writerow(['timestamp', 'latitude', 'longitude'])
        print("🚀 System Start. Press 'q' to Exit.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.resize(frame, (target_w, target_h))
                
                # --- INFERENCE ---
                start_infer = time.time()
                boxes, labels, probs = predictor.predict(frame, 10, CONFIG["model"]["confidence_threshold"])
                end_infer = time.time()
                inference_time_ms = (end_infer - start_infer) * 1000 

                valid_detections = []
                for i in range(boxes.size(0)):
                    box = boxes[i, :] 
                    label_idx = labels[i].item()
                    score = probs[i].item()
                    if label_idx < len(class_names):
                        label_name = class_names[label_idx]
                        valid_detections.append((box.numpy(), label_name, score))

                fps = 1 / (time.time() - prev_frame_time) if (time.time() - prev_frame_time) > 0 else 0
                prev_frame_time = time.time()
                lat, lon, _ = gps_manager.get_coordinates()

                if lat != 0.0 or lon != 0.0:
                    last_checkpoint_coords, checkpoint_crossed = handle_distance_checkpoint(
                        (lat, lon), last_checkpoint_coords, panjang_segmen
                    )
                    if checkpoint_crossed:
                        sta_segmen_selesai = f"{sta_awal}-{sta_awal + panjang_segmen}"
                        hasil_semua_segmen.append({'nama_jalan': jalan_input, 'sta': sta_segmen_selesai, 'jumlah': damage_count_per_segment})
                        sta_awal += panjang_segmen
                        damage_count_per_segment = 0
                sta_string_display = f"{sta_awal}+{sta_awal + panjang_segmen}"

                # --- [MODIFIKASI] TRACKING LOGIC UNTUK SEMUA KELAS ---
                current_centroids = []
                for (box, label_name, _) in valid_detections:
                    # Ambil semua kecuali Background
                    if label_name != "BACKGROUND": 
                        x1, y1, x2, y2 = map(int, box)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        current_centroids.append((cx, cy, y1, y2))
                
                updated_tracked = {}
                y_upper_roi, y_lower_roi = CONFIG["roi"]["y_upper"], CONFIG["roi"]["y_lower"]
                
                # 1. Update existing tracks
                for obj_id, (prev_cx, prev_cy, prev_y1, prev_y2, counted) in tracked_objects.items():
                    min_dist = float('inf'); best_match = None
                    for i, (cx, cy, y1, y2) in enumerate(current_centroids):
                        dist = math.hypot(cx - prev_cx, cy - prev_cy)
                        if dist < 50 and dist < min_dist: min_dist = dist; best_match = i
                    
                    if best_match is not None:
                        n_cx, n_cy, n_y1, n_y2 = current_centroids.pop(best_match)
                        in_roi = (n_y1 >= y_upper_roi and n_y2 <= y_lower_roi)
                        
                        if in_roi and not counted and is_ready_to_capture:
                            damage_count_per_segment += 1; total_damage_count += 1; counted = True
                            print(f"✅ Counted Damage ID {obj_id}")
                        
                        updated_tracked[obj_id] = [n_cx, n_cy, n_y1, n_y2, counted]
                
                # 2. Add new tracks
                for cx, cy, y1, y2 in current_centroids:
                    in_roi = (y1 >= y_upper_roi and y2 <= y_lower_roi)
                    new_cnt = False
                    if in_roi and is_ready_to_capture:
                        damage_count_per_segment += 1; total_damage_count += 1; new_cnt = True
                        print(f"✅ Counted New Damage ID {next_object_id}")
                    
                    updated_tracked[next_object_id] = [cx, cy, y1, y2, new_cnt]
                    next_object_id += 1
                tracked_objects = updated_tracked

                # --- COOLDOWN ---
                if not is_ready_to_capture:
                    if time.time() - last_capture_time > CONFIG['app']['cooldown_seconds']: is_ready_to_capture = True
                
                # --- DRAW & DISPLAY ---
                info_data = {
                    "jalan": jalan_input, "km": km_survey_input, "sta": sta_string_display,
                    "fps": fps, "lat": lat, "lon": lon,
                    "inference_time": inference_time_ms
                }
                luas_m2 = draw_overlays(frame, valid_detections, info_data)

                cv2.putText(frame, f"Count (Seg): {damage_count_per_segment}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Total Count: {total_damage_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                if not is_ready_to_capture:
                    rem = max(0, CONFIG['app']['cooldown_seconds'] - (time.time() - last_capture_time))
                    cv2.putText(frame, f"Cooldown {rem:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                else:
                    cv2.putText(frame, "Ready!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # --- [MODIFIKASI] SAVE TRIGGER UNTUK SEMUA KELAS ---
                if is_ready_to_capture and len(valid_detections) > 0:
                    damage_detected_in_roi = False
                    jenis = ""
                    for (box, label_name, _) in valid_detections:
                        if label_name != "BACKGROUND":
                            _, y1, _, y2 = map(int, box)
                            if y1 >= y_upper_roi and y2 <= y_lower_roi:
                                damage_detected_in_roi = True; jenis = label_name; break
                    
                    if damage_detected_in_roi:
                        threading.Thread(target=process_and_save_data, args=(frame.copy(), lat, lon, jalan_input, sta_string_display, km_survey_input, jenis, luas_m2)).start()
                        is_ready_to_capture = False
                        last_capture_time = time.time()

                if (lat != 0.0 or lon != 0.0) and is_ready_to_capture:
                    gps_writer.writerow([datetime.now().isoformat(), lat, lon])

                cv2.imshow("Deteksi Kerusakan Jalan (All Classes)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

        finally:
            print("\n👋 Exiting...")
            kirim_rekap_via_api(hasil_semua_segmen)
            cap.release()
            cv2.destroyAllWindows()
            gps_manager.stop()

if __name__ == "__main__":
    main()
