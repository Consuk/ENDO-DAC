# eval_endovis_corruptions.py
from __future__ import absolute_import, division, print_function
import os
import argparse
import csv
import numpy as np
import cv2
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from PIL import Image as PILImage

from utils.layers import disp_to_depth
from utils.utils import readlines, compute_errors
from datasets.scared_dataset import SCAREDRAWDataset
import models.encoders as encoders
import models.decoders as decoders
import models.endodac as endodac

# ===== Constantes/metas =====
STEREO_SCALE_FACTOR = 5.4
MIN_DEPTH = 1e-3
MAX_DEPTH = 150.0

cv2.setNumThreads(0)


def _parse_split_line(line: str):
    """
    Soporta formato tokenizado tipo:
        dataset3 keyframe4 390 l
    Devuelve: ds, keyf, frame_idx:int, side:str
    """
    parts = line.strip().split()
    if len(parts) < 4:
        raise ValueError(f"Línea de split inválida: {line!r}")
    ds, keyf, frame_str, side = parts[0], parts[1], parts[2], parts[3]
    return ds, keyf, int(frame_str), side


class SimpleSCAREDWrapperDataset(Dataset):
    """
    Dataset minimalista que reutiliza la lógica de SCAREDRAWDataset
    pero fijando data_path_root (corruptions_root/.../endovis_data).

    Usamos exactamente el parser de paths de SCAREDRAWDataset para
    respetar cómo se construyen las rutas (datasetX/keyframeY/...).
    """
    def __init__(self, data_path_root, filenames, height, width, img_ext=".png"):
        # SCAREDRAWDataset ya implementa get_color / get_image_path
        # y espera (data_path, filenames, height, width, frame_idxs, num_scales, is_train, img_ext)
        self._inner = SCAREDRAWDataset(
            data_path_root,
            filenames,
            height,
            width,
            frame_idxs=[0],
            num_scales=4,
            is_train=False,
            img_ext=img_ext
        )

    def __len__(self):
        return len(self._inner)

    def __getitem__(self, idx):
        sample = self._inner[idx]
        # Sólo nos interesa la imagen de entrada
        return sample


def build_depther_from_args(args, device):
    """
    Construye el "depther" que se comporta igual que en evaluate_depth.py:
    - Para model_type == 'endodac': instancia models.endodac.endodac
      y carga depth_model.pth
    - Para model_type == 'afsfm'  : usa ResnetEncoder + DepthDecoder
      con encoder.pth y depth.pth
    Devuelve una función/callable: depther(images) -> output dict con ("disp", 0)
    """
    load_folder = os.path.expanduser(args.load_weights_folder)
    if not os.path.isdir(load_folder):
        raise FileNotFoundError(f"Cannot find weights folder: {load_folder}")

    print("-> Loading weights from {}".format(load_folder))

    if args.model_type == "endodac":
        # depth_model.pth contiene los pesos del depther "afinados"
        depther_path = os.path.join(load_folder, "depth_model.pth")
        if not os.path.isfile(depther_path):
            raise FileNotFoundError(f"Missing depth_model.pth in {load_folder}")

        depther_dict = torch.load(depther_path, map_location=device)

        # Parsear índices residuales desde string (ej. "0,1,2,3")
        if isinstance(args.residual_block_indexes, str):
            residual_block_indexes = [
                int(x) for x in args.residual_block_indexes.split(",") if x.strip() != ""
            ]
        else:
            residual_block_indexes = list(args.residual_block_indexes)

        model = endodac.endodac(
            backbone_size="base",
            r=args.lora_rank,
            lora_type=args.lora_type,
            image_shape=(224, 280),
            pretrained_path=args.pretrained_path,
            residual_block_indexes=residual_block_indexes,
            include_cls_token=args.include_cls_token
        )
        model_dict = model.state_dict()
        model.load_state_dict({k: v for k, v in depther_dict.items() if k in model_dict})
        model.to(device)
        model.eval()

        def depther(x):
            return model(x)

        return depther

    elif args.model_type == "afsfm":
        # Estilo AF-SfMLearner: encoder.pth + depth.pth
        encoder_path = os.path.join(load_folder, "encoder.pth")
        decoder_path = os.path.join(load_folder, "depth.pth")

        if not os.path.isfile(encoder_path) or not os.path.isfile(decoder_path):
            raise FileNotFoundError(
                f"Missing encoder.pth or depth.pth in {load_folder}"
            )

        encoder = encoders.ResnetEncoder(args.num_layers, False)
        depth_decoder = decoders.DepthDecoder(encoder.num_ch_enc, scales=range(4))

        encoder_dict = torch.load(encoder_path, map_location=device)
        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path, map_location=device))

        encoder.to(device)
        depth_decoder.to(device)
        encoder.eval()
        depth_decoder.eval()

        def depther(x):
            feats = encoder(x)
            out = depth_decoder(feats)
            return out

        return depther

    else:
        raise ValueError(f"model_type desconocido: {args.model_type} (usa 'endodac' o 'afsfm').")


def evaluate_one_root(data_path_root,
                      filenames,
                      gt_depths,
                      depther,
                      args,
                      device="cuda"):
    """
    Evalúa una raíz de corrupciones (p.ej.,
    .../brightness/severity_1/endovis_data) usando la misma lógica que
    evaluate_depth.py pero sobre el dataset corrupto.

    - Usa SCAREDRAWDataset internamente para respetar la estructura
      de rutas (datasetX/keyframeY/...).
    - Hace batching manual.
    """
    img_ext = ".png" if args.png else ".jpg"
    try:
        dataset = SimpleSCAREDWrapperDataset(
            data_path_root, filenames, args.height, args.width, img_ext=img_ext
        )
    except Exception as e:
        raise RuntimeError(f"No se pudo inicializar SCAREDRAWDataset en {data_path_root}: {e}")

    n = len(filenames)
    kept_indices = []
    preds_list = []

    buffer_imgs = []
    buffer_ids = []

    def flush_buffer():
        if len(buffer_imgs) == 0:
            return
        with torch.no_grad():
            batch = torch.stack(buffer_imgs, dim=0).to(device)  # [B,3,H,W]
            output = depther(batch)
            # Compatible con evaluate_depth: salida es dict con ("disp", 0)
            if isinstance(output, dict):
                output_disp = output[("disp", 0)]
            else:
                # Por seguridad, si alguien devuelve directamente disp
                try:
                    output_disp = output[("disp", 0)]
                except Exception:
                    raise RuntimeError(
                        "La salida de depther no es un dict con clave ('disp', 0)."
                    )
            pred_disp, _ = disp_to_depth(
                output_disp, args.min_depth, args.max_depth
            )
            preds_list.append(pred_disp[:, 0].cpu().numpy())

    missing = 0
    for i in range(n):
        try:
            sample = dataset[i]
            img_t = sample[("color", 0, 0)]  # tensor [3,H,W]
            if not isinstance(img_t, torch.Tensor):
                img_t = torch.as_tensor(img_t)
            buffer_imgs.append(img_t)
            buffer_ids.append(i)

            if len(buffer_imgs) == args.batch_size:
                flush_buffer()
                kept_indices.extend(buffer_ids)
                buffer_imgs.clear()
                buffer_ids.clear()

        except FileNotFoundError:
            missing += 1
            if args.strict:
                raise FileNotFoundError(
                    f"[STRICT] Falta la muestra del split idx={i} en {data_path_root}"
                )
            # lenient: se salta
        except Exception as e:
            missing += 1
            if args.strict:
                raise RuntimeError(
                    f"[STRICT] Error cargando idx={i} en {data_path_root}: {e}"
                )
            # lenient: se salta

    # Flush final
    flush_buffer()
    kept_indices.extend(buffer_ids)

    if len(kept_indices) == 0:
        mode = "STRICT" if args.strict else "LENIENT"
        raise FileNotFoundError(
            f"[{mode}] Ninguna imagen utilizable en {data_path_root} "
            f"(faltantes/errores: {missing}/{n})."
        )

    if (not args.strict) and missing > 0:
        print(f"   [INFO] {data_path_root}: usando {len(kept_indices)}/{n} frames del split "
              f"(faltaron {missing}).")

    pred_disps = np.concatenate(preds_list, axis=0)  # [M,H',W']
    sel_gt = gt_depths[kept_indices]

    # Métricas por muestra (misma lógica que evaluate_depth.py)
    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):
        gt_depth = sel_gt[i]
        gt_h, gt_w = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_w, gt_h))
        pred_depth = 1.0 / (pred_disp + 1e-8)

        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        pd = pred_depth[mask]
        gd = gt_depth[mask]

        # Escalado base (estéreo/mono)
        if args.pred_depth_scale_factor != 1.0:
            pd *= args.pred_depth_scale_factor

        # Median scaling monocular (igual que evaluate_depth)
        if not args.disable_median_scaling:
            median_pred = np.median(pd)
            if median_pred < 1e-6:
                continue  # saltamos esta muestra
            ratio = np.median(gd) / (median_pred + 1e-8)
            ratios.append(ratio)
            pd *= ratio

        pd[pd < MIN_DEPTH] = MIN_DEPTH
        pd[pd > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gd, pd))

    if len(errors) == 0:
        raise RuntimeError(
            f"No se pudieron calcular métricas en {data_path_root} (quizá todas las muestras fueron inválidas)."
        )

    if not args.disable_median_scaling and len(ratios) > 0:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(f"    Scaling ratios | med: {med:0.3f} | std: {np.std(ratios / med):0.3f}")

    mean_errors = np.array(errors).mean(0)
    # abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
    return mean_errors


def list_corruption_dirs(root):
    """
    Devuelve los directorios de primer nivel que representan corrupciones.
    Si 'root' ya es una carpeta de una corrupción (que contiene severity_*), la devuelve tal cual.
    """
    if not os.path.isdir(root):
        return []
    severities = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and d.startswith("severity_")
    ]
    if len(severities) > 0:
        return [root]
    return [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))
    ]


def main():
    parser = argparse.ArgumentParser(
        "Evaluate EndoVIS corruptions (16x5) with ENDO-DAC / AF-SfM weights"
    )
    parser.add_argument("--corruptions_root", type=str, required=True,
                        help="Raíz de las corrupciones (o una sola corrupción). Ej: /workspace/endovis_corruptions_test")
    parser.add_argument("--load_weights_folder", type=str, required=True,
                        help="Carpeta con depth_model.pth (endodac) o encoder.pth+depth.pth (afsfm)")
    parser.add_argument("--splits_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "splits"))
    parser.add_argument("--split", type=str, default="endovis",
                        help="Nombre del split (carpeta dentro de splits/)")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)  # no se usa mucho aquí
    parser.add_argument("--png", action="store_true",
                        help="Usa .png en lugar de .jpg")
    parser.add_argument("--eval_stereo", action="store_true",
                        help="Forzar estéreo (desactiva median scaling y usa x5.4)")
    parser.add_argument("--output_csv", type=str, default="corruptions_summary.csv")
    parser.add_argument("--strict", action="store_true",
                        help="Modo estricto: exige que todas las entradas del split existan en cada severidad.")

    # ---- Parámetros de modelo (alineados con evaluate_depth) ----
    parser.add_argument("--model_type", type=str, default="endodac",
                        choices=["endodac", "afsfm"],
                        help="Tipo de modelo a evaluar.")
    parser.add_argument("--num_layers", type=int, default=18,
                        help="Número de capas del ResNet (para afsfm).")
    parser.add_argument("--min_depth", type=float, default=1e-3)
    parser.add_argument("--max_depth", type=float, default=150.0)
    parser.add_argument("--disable_median_scaling", action="store_true",
                        help="Desactiva median scaling (para mono).")
    parser.add_argument("--pred_depth_scale_factor", type=float, default=1.0,
                        help="Factor para escalar las profundidades predichas (ej. 5.4 en estéreo).")

    # EndoDAC-specific
    parser.add_argument("--pretrained_path", type=str, default="",
                        help="Ruta al checkpoint de Depth Anything para EndoDAC.")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_type", type=str, default="svd")
    parser.add_argument("--include_cls_token", action="store_true")
    parser.add_argument("--residual_block_indexes", type=str, default="0,1,2,3",
                        help="Índices de bloques residuales para LoRA, separados por comas.")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Configuración mono/estéreo coherente
    if args.eval_stereo:
        args.disable_median_scaling = True
        args.pred_depth_scale_factor = STEREO_SCALE_FACTOR

    # Leer split y GTs (como en evaluate_depth para endovis)
    test_files_path = os.path.join(args.splits_dir, args.split, "test_files.txt")
    if not os.path.isfile(test_files_path):
        raise FileNotFoundError(f"No se encontró test_files.txt en {test_files_path}")

    test_files = readlines(test_files_path)

    gt_path = os.path.join(args.splits_dir, args.split, "gt_depths.npz")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"No se encontró gt_depths.npz en {gt_path}")
    gt_depths = np.load(gt_path, fix_imports=True, encoding="latin1")["data"]

    if len(test_files) != gt_depths.shape[0]:
        print("[WARN] test_files.txt y gt_depths.npz difieren en longitud. "
              "El script usará sólo los índices que se puedan alinear correctamente.")

    # Construir modelo (igual que evaluate_depth, pero factorizado)
    depther = build_depther_from_args(args, device)

    # Detectar corrupciones
    corr_dirs = list_corruption_dirs(args.corruptions_root)
    if len(corr_dirs) == 0:
        raise FileNotFoundError(
            f"No se encontraron carpetas de corrupción en {args.corruptions_root}"
        )

    rows = []
    print("-> Iniciando evaluación de corrupciones")
    for corr_dir in corr_dirs:
        corr_name = os.path.basename(corr_dir.rstrip("/"))

        severities = sorted(
            [
                d for d in os.listdir(corr_dir)
                if os.path.isdir(os.path.join(corr_dir, d)) and d.startswith("severity_")
            ],
            key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else 9999
        )

        for sev in severities:
            data_root = os.path.join(corr_dir, sev, "endovis_data")
            print(f"\n>> {corr_name} / {sev} :: data_path = {data_root}")
            if not os.path.isdir(data_root):
                print(f"   [WARN] No existe {data_root}, se omite.")
                continue

            try:
                mean_errors = evaluate_one_root(
                    data_path_root=data_root,
                    filenames=test_files,
                    gt_depths=gt_depths,
                    depther=depther,
                    args=args,
                    device=device,
                )
                abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = mean_errors.tolist()
                rows.append([corr_name, sev, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3])

                print("   Métricas (promedio): "
                      f"abs_rel={abs_rel:.3f} | sq_rel={sq_rel:.3f} | rmse={rmse:.3f} | "
                      f"rmse_log={rmse_log:.3f} | a1={a1:.3f} | a2={a2:.3f} | a3={a3:.3f}")

            except FileNotFoundError as e:
                print(f"   [SKIP] {e}")
            except RuntimeError as e:
                print(f"   [SKIP] {e}")

    # Guardar CSV
    if rows:
        header = ["corruption", "severity", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
        with open(args.output_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in rows:
                w.writerow(r)

        print(f"\n-> Resumen guardado en: {args.output_csv}")

        bucket = defaultdict(list)
        for r in rows:
            bucket[r[0]].append(r)

        print("\n======= RESUMEN (por corrupción) =======")
        for corr in sorted(bucket.keys()):
            print(f"\n{corr}")
            print("severity | abs_rel |  sq_rel |  rmse  | rmse_log |   a1   |   a2   |   a3")
            for _, sev, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 in sorted(
                bucket[corr],
                key=lambda x: int(x[1].split("_")[-1]) if x[1].split("_")[-1].isdigit() else 9999
            ):
                print(f"{sev:>9} | {abs_rel:7.3f} | {sq_rel:7.3f} | {rmse:7.3f} |  {rmse_log:7.3f} | "
                      f"{a1:6.3f} | {a2:6.3f} | {a3:6.3f}")
    else:
        print("\n-> No se generaron filas. Revisa rutas/archivos o estructura de corrupciones.")


if __name__ == "__main__":
    main()
