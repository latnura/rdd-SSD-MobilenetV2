import torch
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_large_ssd_lite, create_mobilenetv3_small_ssd_lite
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.utils import box_utils, measurements
from vision.utils.misc import str2bool, Timer
from thop import profile

import argparse
import pathlib
import numpy as np
import logging
import sys


# ================================
# Argument parser
# ================================
parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
parser.add_argument('--net', default="mb2-ssd-lite", help="The network architecture.")
parser.add_argument("--trained_model", type=str)
parser.add_argument("--dataset_type", default="voc", type=str)
parser.add_argument("--dataset", type=str)
parser.add_argument("--label_file", type=str)
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument("--use_2007_metric", type=str2bool, default=True)
parser.add_argument("--nms_method", type=str, default="hard")
parser.add_argument("--iou_threshold", type=float, default=0.5)
parser.add_argument("--eval_dir", default="eval_results", type=str)
parser.add_argument('--mb2_width_mult', default=1.0, type=float)
args = parser.parse_args()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")


# ================================
# Helper: Load GT annotation
# ================================
def group_annotation_by_class(dataset):
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        gt_boxes = torch.from_numpy(gt_boxes)
        for i, difficult in enumerate(is_difficult):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            if not difficult:
                true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1
            all_gt_boxes.setdefault(class_index, {}).setdefault(image_id, []).append(gt_box)
            all_difficult_cases.setdefault(class_index, {}).setdefault(image_id, []).append(difficult)
    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])
    for class_index in all_difficult_cases:
        for image_id in all_difficult_cases[class_index]:
            all_difficult_cases[class_index][image_id] = torch.tensor(all_difficult_cases[class_index][image_id])
    return true_case_stat, all_gt_boxes, all_difficult_cases


# ================================
# Helper: Compute AP per class
# ================================
def compute_average_precision_per_class(num_true_cases, gt_boxes, difficult_cases, prediction_file, iou_threshold, use_2007_metric):
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0
            boxes.append(box)

        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]

        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()

        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue
            gt_box = gt_boxes[image_id]
            ious = box_utils.iou_of(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases

    if use_2007_metric:
        return measurements.compute_voc2007_average_precision(precision, recall)
    else:
        return measurements.compute_average_precision(precision, recall)


# ================================
# Main Evaluation
# ================================
if __name__ == '__main__':
    eval_path = pathlib.Path(args.eval_dir)
    eval_path.mkdir(exist_ok=True)

    timer = Timer()
    class_names = [name.strip() for name in open(args.label_file).readlines()]
    dataset = VOCDataset(args.dataset, is_test=True) if args.dataset_type == "voc" else OpenImagesDataset(args.dataset, dataset_type="test")

    true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)

    # (Kode untuk load model dan predictor tetap sama, saya singkat)
    if args.net == 'vgg16-ssd':
        net = create_vgg_ssd(len(class_names), is_test=True)
        predictor = create_vgg_ssd_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd':
        net = create_mobilenetv1_ssd(len(class_names), is_test=True)
        predictor = create_mobilenetv1_ssd_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd-lite':
        net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
        predictor = create_mobilenetv1_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'sq-ssd-lite':
        net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
        predictor = create_squeezenet_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)
    else: # Default ke mb2 atau model lain yang menggunakan predictor yang sama
        net = create_mobilenetv2_ssd_lite(len(class_names), width_mult=args.mb2_width_mult, is_test=True)
        predictor = create_mobilenetv2_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)

    timer.start("Load Model")
    net.load(args.trained_model)
    net = net.to(DEVICE)
    print(f'It took {timer.end("Load Model")} seconds to load the model.')
    
    # <<< BAGIAN BARU: Kalkulasi GFLOPs dan Parameter Model >>>
    # Ganti ukuran input menjadi 512x512, sesuai dengan model resolusi tinggi
    input_size = (1, 3, 512, 512) 
    dummy_input = torch.randn(input_size).to(DEVICE)
    
    # Hitung FLOPs dan Params menggunakan thop
    flops, params = profile(net, inputs=(dummy_input,))
    
    # Konversi ke GFLOPs (Giga) dan MParams (Mega/Juta)
    gflops = flops / 1e9
    params_m = params / 1e6
    print(f"Model Complexity: {gflops:.2f} GFLOPs, {params_m:.2f} M Params")
    # <<< AKHIR BAGIAN BARU >>>

    # (Kode prediksi dan penulisan file tetap sama, saya singkat)
    THRESHOLD = 0.01
    print(f"\n=== Using confidence threshold = {THRESHOLD} ===\n")
    results = []
    print("Running predictions on all images...")
    for i in range(len(dataset)):
        image = dataset.get_image(i)
        boxes, labels, probs = predictor.predict(image)
        mask = probs > THRESHOLD
        boxes, labels, probs = boxes[mask], labels[mask], probs[mask]
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        results.append(torch.cat([indexes.reshape(-1, 1), labels.reshape(-1, 1).float(), probs.reshape(-1, 1), boxes + 1.0], dim=1))
    results = torch.cat(results)
    print("Prediction finished.")
    print("Writing prediction files...")
    for class_index, class_name in enumerate(class_names):
        if class_index == 0: continue
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        with open(prediction_path, "w") as f:
            sub = results[results[:, 1] == class_index, :]
            for i in range(sub.size(0)):
                prob_box = sub[i, 2:].numpy()
                image_id = dataset.ids[int(sub[i, 0])]
                print(f"{image_id} {' '.join([str(v) for v in prob_box])}", file=f)
    print("Prediction files written.")

    # =================================================================
    # BAGIAN UTAMA: Evaluasi Menyeluruh
    # =================================================================
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    all_maps = []
    
    precisions_at_50, recalls_at_50, f1s_at_50 = [], [], []

    print(f"\n=== Starting Evaluation for IoU thresholds from {iou_thresholds[0]:.2f} to {iou_thresholds[-1]:.2f} ===\n")

    for iou_threshold in iou_thresholds:
        aps = []
        is_iou_50 = np.isclose(iou_threshold, 0.5)
        
        # <<< BARIS BARU: Menambahkan header untuk output per kelas
        if is_iou_50:
            print(f"--- Per-Class Metrics @ IoU: {iou_threshold:.2f} ---")
        
        for class_index, class_name in enumerate(class_names):
            if class_index == 0: continue
            prediction_path = eval_path / f"det_test_{class_name}.txt"
            if class_index not in true_case_stat: continue

            ap = compute_average_precision_per_class(
                true_case_stat[class_index], all_gb_boxes[class_index],
                all_difficult_cases[class_index], prediction_path,
                iou_threshold, args.use_2007_metric
            )
            aps.append(ap)

            if is_iou_50:
                with open(prediction_path) as f:
                    predictions = [line.strip().split() for line in f.readlines()]
                
                matched_gt, tp, fp = set(), 0, 0
                for p in predictions:
                    image_id, box = p[0], torch.tensor([float(x) for x in p[2:]]).unsqueeze(0) - 1
                    if image_id not in all_gb_boxes[class_index]:
                        fp += 1; continue
                    
                    gt_boxes = all_gb_boxes[class_index][image_id]
                    difficult = all_difficult_cases[class_index][image_id]
                    ious = box_utils.iou_of(box, gt_boxes)
                    max_iou, max_idx = torch.max(ious, dim=0)
                    
                    if max_iou.item() >= iou_threshold and not difficult[max_idx.item()]:
                        if (image_id, max_idx.item()) not in matched_gt:
                            tp += 1
                            matched_gt.add((image_id, max_idx.item()))
                        else: fp += 1
                    else: fp += 1

                fn = true_case_stat[class_index] - tp
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                # <<< BAGIAN BARU: Cetak metrik untuk kelas saat ini
                ap_at_50 = ap # Ambil nilai AP yang baru dihitung untuk IoU 0.5
                print(f"  - Class: {class_name}")
                print(f"    AP       : {ap_at_50:.4f}")
                print(f"    Precision: {precision:.4f}")
                print(f"    Recall   : {recall:.4f}")
                print(f"    F1-Score : {f1:.4f}")
                
                precisions_at_50.append(precision)
                recalls_at_50.append(recall)
                f1s_at_50.append(f1)

        if aps:
            current_map = sum(aps) / len(aps)
            all_maps.append(current_map)
    
    # <<< BARIS BARU: Menambahkan pemisah sebelum ringkasan akhir
    if precisions_at_50: print("-" * 40) 

    # ==========================================================
    # BAGIAN AKHIR: Tampilkan ringkasan seperti tabel Anda
    # ==========================================================
    if all_maps:
        avg_precision = sum(precisions_at_50)/len(precisions_at_50) if precisions_at_50 else 0.0
        avg_recall = sum(recalls_at_50)/len(recalls_at_50) if recalls_at_50 else 0.0
        avg_f1 = sum(f1s_at_50)/len(f1s_at_50) if f1s_at_50 else 0.0
        map_50_to_95 = sum(all_maps) / len(all_maps)

        print("\n\n" + "="*40)
        print("      MODEL EVALUATION SUMMARY")
        print("="*40)
        print(f"Precision        : {avg_precision:.4f}")
        print(f"Recall           : {avg_recall:.4f}")
        print(f"F1-Score         : {avg_f1:.4f}")
        print("-" * 40)
        print(f"mAP @ 0.5        : {all_maps[0]:.4f}")
        print(f"mAP @ 0.5:0.95   : {map_50_to_95:.4f}")
        print("="*40)
        print("\nNOTE: 'Total Loss' and 'Time' are training metrics.")
    else:
        print("\n[ERROR] Evaluation failed. No mAP scores were calculated.")