import argparse
import os
import logging
import sys
import itertools
import torch
import csv
import numpy as np

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True

from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, ReduceLROnPlateau
from torch.optim import Adam, SGD
from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels

# Import Arsitektur & Predictor
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_large_ssd_lite, create_mobilenetv3_small_ssd_lite
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor

from vision.ssd.ssd import MatchPrior
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import vgg_ssd_config, mobilenetv1_ssd_config, squeezenet_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform

# Import Box Utils dan Measurements untuk mAP
from vision.utils import box_utils, measurements

# =============================
# Argument parser
# =============================
parser = argparse.ArgumentParser(description='SSD Training with Grid Search & Box Metrics Support')

parser.add_argument('--results_file', default='gridsearch_results.csv', type=str,
                    help='Nama file CSV untuk menyimpan hasil')

parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')
parser.add_argument('--datasets', nargs='+', help='Dataset directory path')
parser.add_argument('--validation_dataset', help='Dataset directory path')
parser.add_argument('--balance_data', action='store_true')

parser.add_argument('--net', default="mb2-ssd-lite",
                    help="The network architecture.")
parser.add_argument('--freeze_base_net', action='store_true')
parser.add_argument('--freeze_net', action='store_true')
parser.add_argument('--mb2_width_mult', default=1.0, type=float)

# Training params
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--base_net_lr', default=None, type=float)
parser.add_argument('--extra_layers_lr', default=None, type=float)

# Model loading
parser.add_argument('--base_net')
parser.add_argument('--pretrained_ssd')
parser.add_argument('--resume', default=None, type=str)

# Scheduler
# ---> [MODIFIKASI] Menambahkan 'none' sebagai opsi default atau pilihan <---
parser.add_argument('--scheduler', default="none", type=str, 
                    help="Pilih scheduler: plateau, multi-step, cosine, atau none")
parser.add_argument('--milestones', default="80,100", type=str)
parser.add_argument('--t_max', default=120, type=float)
parser.add_argument('--lr_patience', default=3, type=int)
parser.add_argument('--lr_factor', default=0.5, type=float)
parser.add_argument('--min_lr', default=1e-6, type=float)

# Train params
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_epochs', default=120, type=int)
parser.add_argument('--num_workers', default=12, type=int)
parser.add_argument('--validation_epochs', default=5, type=int)
parser.add_argument('--debug_steps', default=100, type=int)
parser.add_argument('--use_cuda', default=True, type=str2bool)
parser.add_argument('--checkpoint_folder', default='models_scheduler_plateau/')

# Optimizer
parser.add_argument('--optimizer', default="adam", type=str,
                    help="Optimizer: adam, momentum, rmsprop, sgd")

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
args = parser.parse_args()

# --- W&B: INISIALISASI ---
import wandb
wandb.init(
    project="RMS_FOV-Road-Damage",  
    name=f"{args.net}_{args.optimizer}_lr{args.lr}", 
    config=vars(args) 
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Use Cuda.")

# =============================
# Helper & Training Functions
# =============================

def smart_load_weights(model, path):
    logging.info(f"SMART LOAD: Membuka file {path}...")
    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    model_dict = model.state_dict()
    
    new_state_dict = {k: v for k, v in state_dict.items() 
                      if k in model_dict and v.size() == model_dict[k].size()}
    
    total_layers = len(state_dict)
    loaded_layers = len(new_state_dict)
    skipped_layers = total_layers - loaded_layers
    
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    
    logging.info(f"SMART LOAD BERHASIL:")
    logging.info(f"  - Loaded : {loaded_layers} layer (Backbone & Head lama)")
    logging.info(f"  - Skipped: {skipped_layers} layer (Head baru yang akan dilatih ulang)")
    
    return model

def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss, running_regression_loss, running_classification_loss = 0.0, 0.0, 0.0
    epoch_loss, epoch_reg_loss, epoch_clf_loss = 0.0, 0.0, 0.0

    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        l_val = loss.item()
        r_val = regression_loss.item()
        c_val = classification_loss.item()

        running_loss += l_val
        running_regression_loss += r_val
        running_classification_loss += c_val
        epoch_loss += l_val
        epoch_reg_loss += r_val
        epoch_clf_loss += c_val

        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(
                f"Epoch: {epoch}, Step: {i}, "
                f"Avg Loss: {avg_loss:.4f}, "
                f"Avg Reg Loss: {avg_reg_loss:.4f}, "
                f"Avg Clf Loss: {avg_clf_loss:.4f}"
            )
            running_loss, running_regression_loss, running_classification_loss = 0.0, 0.0, 0.0

    num_batches = len(loader)
    return epoch_loss / num_batches, epoch_reg_loss / num_batches, epoch_clf_loss / num_batches

def test(loader, net, criterion, device):
    """Fungsi ini sekarang murni hanya menghitung Validation Loss"""
    net.eval()
    running_loss, running_regression_loss, running_classification_loss = 0.0, 0.0, 0.0
    num = 0

    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()

    return running_loss / num, running_regression_loss / num, running_classification_loss / num

def evaluate_map_and_metrics(dataset, predictor, map_prob_threshold=0.01, stat_conf_threshold=0.21, use_2007_metric=True):
    """Menghitung mAP dengan threshold rendah, dan metrik statis dengan threshold spesifik."""
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    all_preds = {}
    
    # --- 1. Kumpulkan Ground Truth & Prediksi ---
    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        gt_boxes = torch.from_numpy(gt_boxes)
        
        for j, difficult in enumerate(is_difficult):
            class_index = int(classes[j])
            if not difficult:
                true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1
            all_gt_boxes.setdefault(class_index, {}).setdefault(i, []).append(gt_boxes[j])
            all_difficult_cases.setdefault(class_index, {}).setdefault(i, []).append(difficult)
            
        image = dataset.get_image(i)
        
        # [MODIFIKASI] Gunakan map_prob_threshold (default 0.01) untuk pengumpulan prediksi awal
        pred_boxes, pred_labels, pred_probs = predictor.predict(image, top_k=200, prob_threshold=map_prob_threshold)
        
        for j in range(pred_boxes.size(0)):
            class_index = int(pred_labels[j].item())
            if class_index == 0: continue
            all_preds.setdefault(class_index, []).append((i, pred_probs[j].item(), pred_boxes[j]))

    # Stack tensor GT
    for c in all_gt_boxes:
        for img_idx in all_gt_boxes[c]:
            all_gt_boxes[c][img_idx] = torch.stack(all_gt_boxes[c][img_idx])
            all_difficult_cases[c][img_idx] = torch.tensor(all_difficult_cases[c][img_idx])

    # --- 2. Hitung mAP & Metrik Statis ---
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    all_maps = []
    
    # Variabel untuk Precision/Recall statis
    TP_static, FP_static, FN_static = 0, 0, sum(true_case_stat.values())

    for iou_thresh in iou_thresholds:
        aps = []
        is_iou_50 = np.isclose(iou_thresh, 0.5)
        
        for class_index in true_case_stat.keys():
            if class_index not in all_preds:
                aps.append(0.0)
                continue
                
            preds = all_preds[class_index]
            preds.sort(key=lambda x: x[1], reverse=True) # Urutkan dari prob terbesar
            
            nd = len(preds)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            matched = set()
            
            for d, (img_idx, prob, box) in enumerate(preds):
                if img_idx not in all_gt_boxes.get(class_index, {}):
                    fp[d] = 1
                    continue
                    
                gt_box_group = all_gt_boxes[class_index][img_idx]
                ious = box_utils.iou_of(box.unsqueeze(0), gt_box_group)
                max_iou = torch.max(ious).item()
                max_arg = torch.argmax(ious).item()
                
                if max_iou > iou_thresh:
                    if all_difficult_cases[class_index][img_idx][max_arg] == 0:
                        if (img_idx, max_arg) not in matched:
                            tp[d] = 1
                            matched.add((img_idx, max_arg))
                            
                            # [MODIFIKASI] Gunakan stat_conf_threshold (default 0.21) untuk TP Statis
                            if is_iou_50 and prob >= stat_conf_threshold:
                                TP_static += 1
                        else:
                            fp[d] = 1
                else:
                    fp[d] = 1
                    
                # [MODIFIKASI] Gunakan stat_conf_threshold (default 0.21) untuk FP Statis
                if is_iou_50 and fp[d] == 1 and prob >= stat_conf_threshold:
                    FP_static += 1
                    
            fp_cumsum = np.cumsum(fp)
            tp_cumsum = np.cumsum(tp)
            recall = tp_cumsum / true_case_stat[class_index]
            precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)
            
            if use_2007_metric:
                ap = measurements.compute_voc2007_average_precision(precision, recall)
            else:
                ap = measurements.compute_average_precision(precision, recall)
            aps.append(ap)
            
        all_maps.append(sum(aps) / len(aps) if aps else 0.0)
        
    map_50 = all_maps[0] if all_maps else 0.0
    map_50_95 = sum(all_maps) / len(all_maps) if all_maps else 0.0
    
    # Kalkulasi akhir metrik statis pada stat_conf_threshold
    FN_static = FN_static - TP_static
    prec_static = TP_static / (TP_static + FP_static) if (TP_static + FP_static) > 0 else 0.0
    rec_static = TP_static / (TP_static + FN_static) if (TP_static + FN_static) > 0 else 0.0
    f1_static = 2 * prec_static * rec_static / (prec_static + rec_static) if (prec_static + rec_static) > 0 else 0.0
    
    return map_50, map_50_95, prec_static, rec_static, f1_static

# =============================
# Main Training Loop
# =============================
if __name__ == '__main__':
    timer = Timer()
    logging.info(args)

    # Pilih arsitektur dan simpan constructor predictor-nya
    if args.net == 'vgg16-ssd':
        create_net = create_vgg_ssd
        config = vgg_ssd_config
        create_predictor = create_vgg_ssd_predictor
    elif args.net == 'mb2-ssd-lite':
        create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=args.mb2_width_mult)
        config = mobilenetv1_ssd_config
        create_predictor = create_mobilenetv2_ssd_lite_predictor
    elif args.net == 'mb1-ssd':
        create_net = create_mobilenetv1_ssd
        config = mobilenetv1_ssd_config
        create_predictor = create_mobilenetv1_ssd_predictor
    elif args.net == 'mb1-ssd-lite':
        create_net = create_mobilenetv1_ssd_lite
        config = mobilenetv1_ssd_config
        create_predictor = create_mobilenetv1_ssd_lite_predictor
    elif args.net == 'sq-ssd-lite':
        create_net = create_squeezenet_ssd_lite
        config = squeezenet_ssd_config
        create_predictor = create_squeezenet_ssd_lite_predictor
    elif args.net == 'mb3-large-ssd-lite':
        create_net = lambda num: create_mobilenetv3_large_ssd_lite(num)
        config = mobilenetv1_ssd_config
        create_predictor = create_mobilenetv2_ssd_lite_predictor # Default fallback
    elif args.net == 'mb3-small-ssd-lite':
        create_net = lambda num: create_mobilenetv3_small_ssd_lite(num)
        config = mobilenetv1_ssd_config
        create_predictor = create_mobilenetv2_ssd_lite_predictor # Default fallback
    else:
        logging.fatal("Net type salah.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    # Data augmentation
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    # Load dataset
    logging.info("Load training dataset.")
    datasets = []
    for dataset_path in args.datasets:
        if args.dataset_type == 'voc':
            dataset = VOCDataset(dataset_path, transform=train_transform, target_transform=target_transform)
            label_file = os.path.join(args.checkpoint_folder, "voc-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            num_classes = len(dataset.class_names)
        elif args.dataset_type == 'open_images':
            dataset = OpenImagesDataset(dataset_path, transform=train_transform,
                                        target_transform=target_transform, dataset_type="train",
                                        balance_data=args.balance_data)
            label_file = os.path.join(args.checkpoint_folder, "open-images-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            num_classes = len(dataset.class_names)
        else:
            raise ValueError(f"Unsupported dataset type {args.dataset_type}")
        datasets.append(dataset)

    train_dataset = ConcatDataset(datasets)
    train_loader = DataLoader(train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=args.num_workers,
                          drop_last=True)
    
    logging.info("Load validation dataset.")
    if args.dataset_type == "voc":
        val_dataset = VOCDataset(args.validation_dataset, transform=test_transform,
                                 target_transform=target_transform, is_test=True)
        # Note: Kita simpan val_dataset_raw untuk evaluate_box_metrics yang tidak butuh target_transform (mendapatkan GT asli)
        val_dataset_raw = VOCDataset(args.validation_dataset, is_test=True) 
    elif args.dataset_type == "open_images":
        val_dataset = OpenImagesDataset(args.validation_dataset, transform=test_transform,
                                        target_transform=target_transform, dataset_type="test")
        val_dataset_raw = OpenImagesDataset(args.validation_dataset, dataset_type="test")

    val_loader = DataLoader(val_dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        drop_last=True)

    # Build network
    logging.info("Build network.")
    net = create_net(num_classes)

    # Parameter training
    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    
    if args.freeze_base_net:
        freeze_net_layers(net.base_net)
        params = [
            {'params': itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters()), 'lr': extra_layers_lr},
            {'params': itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())}
        ]
    elif args.freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
    else:
        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters()), 'lr': extra_layers_lr},
            {'params': itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())}
        ]

    # Load model
    timer.start("Load Model")
    if args.resume:
        logging.info(f"Resuming from {args.resume}")
        smart_load_weights(net, args.resume) 
    elif args.base_net:
        logging.info(f"Init from base net {args.base_net}")
        net.init_from_base_net(args.base_net)
    elif args.pretrained_ssd:
        logging.info(f"Loading pretrained SSD from {args.pretrained_ssd}")
        smart_load_weights(net, args.pretrained_ssd) 
        
    logging.info(f"Took {timer.end('Load Model'):.2f} seconds to load model.")
    net.to(DEVICE)

    # Loss
    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)

    # Optimizer
    if args.optimizer == "adam":
        optimizer = Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "momentum":
        optimizer = SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(params, lr=args.lr, alpha=0.99, eps=1e-8,
                                        weight_decay=args.weight_decay, momentum=0.9)
    elif args.optimizer == "sgd":
        optimizer = SGD(params, lr=args.lr, momentum=0.0, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Scheduler
    # ---> [MODIFIKASI] Menambahkan logika untuk 'none' <---
    logging.info(f"Menggunakan scheduler: {args.scheduler}")
    if args.scheduler == 'multi-step':
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.t_max)
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr)
    elif args.scheduler == 'none':
        scheduler = None
        logging.info("Training berjalan dengan Learning Rate konstan (tanpa scheduler).")
    else:
        logging.fatal(f"Tipe scheduler tidak didukung: {args.scheduler}")
        sys.exit(1)

    # Training loop
    for epoch in range(args.num_epochs):
        train_loss, train_reg_loss, train_clf_loss = train(
            train_loader, net, criterion, optimizer, device=DEVICE,
            debug_steps=args.debug_steps, epoch=epoch
        )

        wandb_metrics = {
            "epoch": epoch,
            "Train/Total_Loss": train_loss,
            "Train/Regression_Loss": train_reg_loss,
            "Train/Classification_Loss": train_clf_loss,
            "Learning_Rate": optimizer.param_groups[0]['lr']
        }

        # Validasi
        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            val_loss, val_reg_loss, val_clf_loss = test(val_loader, net, criterion, DEVICE)
            
            net.eval()
            predictor = create_predictor(net, nms_method="hard", device=DEVICE)
            
            # Panggil fungsi yang baru
            map_50, map_50_95, prec, rec, f1 = evaluate_map_and_metrics(
                val_dataset_raw, predictor, 
                map_prob_threshold=0.01,      # Threshold untuk perhitungan luas kurva mAP
                stat_conf_threshold=0.21,     # Threshold untuk memutus metrik operasional (P, R, F1)
                use_2007_metric=True
            )
            
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)

            logging.info(
                f"Epoch: {epoch}, ValLoss: {val_loss:.4f}, "
                f"mAP@0.5: {map_50:.4f}, mAP@0.5:0.95: {map_50_95:.4f}, "
                f"Prec(0.21): {prec:.4f}, Rec(0.21): {rec:.4f}, F1: {f1:.4f}, "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

            model_path = os.path.join(args.checkpoint_folder, f"{args.net}-Epoch-{epoch}-Loss-{val_loss:.4f}.pth")
            net.save(model_path)

            wandb_metrics.update({
                "Val/Total_Loss": val_loss,
                "Val/Regression_Loss": val_reg_loss,
                "Val/Classification_Loss": val_clf_loss,
                "Metrics/mAP_0.5": map_50,
                "Metrics/mAP_0.5_0.95": map_50_95,
                "Metrics/Precision_0.21": prec,
                "Metrics/Recall_0.21": rec,
                "Metrics/F1_Score_0.21": f1
            })

            # Simpan ke CSV
            # Simpan ke CSV (Setiap kali validasi berjalan)
            results_file = args.results_file
            if not os.path.exists(results_file):
                with open(results_file, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Optimizer", "BatchSize", "Epoch", "ValLoss", "mAP_50", "mAP_50_95", "Precision", "Recall", "F1"])
            with open(results_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([args.optimizer, args.batch_size, epoch, val_loss, map_50, map_50_95, prec, rec, f1])
                logging.info(f"Results saved to {results_file}")

        if args.scheduler in ['multi-step', 'cosine']:
            scheduler.step()
        
        wandb.log(wandb_metrics)

    wandb.finish()