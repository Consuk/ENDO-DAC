# eval_endovis_corruptions.py
from __future__ import absolute_import, division, print_function
import os
import argparse
import csv
import numpy as np
import cv2
from collections import defaultdict
from datasets import SCAREDRAWDataset

import torch
from torch.utils.data import DataLoader

# Reutilizamos utilidades y redes del repo
from options import MonodepthOptions  # solo para consistencia de estilos (no usamos aquí)
from utils import readlines
import networks
from layers import disp_to_depth

try:
    from PIL import Image as PILImage
except Exception as e:
    raise ImportError("Pillow es requerido: pip install pillow") from e

# ===== Constantes/metas =====
STEREO_SCALE_FACTOR = 5.4
MIN_DEPTH = 1e-3
MAX_DEPTH = 150.0


def compute_errors(gt, pred):
    """Métricas estándar de Monodepth/EndoDepth."""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel  = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


# ===================== MOD ENDODAC: loader robusto ResNet / EndoDAC =====================
def _looks_like_vit_state_dict(sd_keys):
    """Heurística para detectar ViT/DepthAnything/Transformer por nombres de keys."""
    vit_hints = ("patch_embed", "pos_embed", "blocks.", "transformer", "attn", "mlp", "norm")
    return any(any(h in k for h in vit_hints) for k in sd_keys)

def _looks_like_endodac_decoder(sd_keys):
    """Heurística ligera para decoder tipo DPT/Reassemble-Fusion."""
    dec_hints = ("reassemble", "fusion", "dpt", "head", "upsample", "refine")
    return any(any(h in k.lower() for h in dec_hints) for k in sd_keys)

def _instantiate_first_available(module, candidates, fallback_msg):
    """
    Intenta instanciar la primera clase disponible de `candidates`.
    candidates: lista de tuplas (name, kwargs)
    """
    for name, kwargs in candidates:
        cls = getattr(module, name, None)
        if cls is None:
            continue
        try:
            return cls(**kwargs)
        except TypeError:
            # por si el ctor no acepta kwargs
            try:
                return cls()
            except Exception:
                continue
        except Exception:
            continue
    raise RuntimeError(fallback_msg)

def load_model(load_weights_folder, num_layers, device):
    """
    Carga encoder.pth y depth.pth una sola vez.
    Soporta tanto ResNet (AF-SfMLearner clásico) como EndoDAC (ViT/DepthAnything + LoRA).
    """
    encoder_path = os.path.join(load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(load_weights_folder, "depth.pth")

    if not os.path.isdir(load_weights_folder):
        raise FileNotFoundError(f"Cannot find weights folder: {load_weights_folder}")
    if not os.path.isfile(encoder_path) or not os.path.isfile(decoder_path):
        raise FileNotFoundError("Missing encoder.pth or depth.pth in weights folder")

    encoder_dict = torch.load(encoder_path, map_location=device)
    depth_dict   = torch.load(decoder_path, map_location=device)

    enc_keys = list(encoder_dict.keys())
    dec_keys = list(depth_dict.keys())

    is_vit = _looks_like_vit_state_dict(enc_keys) or _looks_like_endodac_decoder(dec_keys)

    if is_vit:
        # --- EndoDAC / ViT path ---
        # Ajusta aquí SOLO si tu clase se llama distinto en tu repo:
        ENCODER_CANDIDATES = [
            ("DeepNet", {}),               # nombre usado con MonoViT/EndoDAC en tu repo
            ("DepthNet", {}),
            ("DepthAnythingEncoder", {}),
            ("DADepthEncoder", {}),
            ("EndoDACEncoder", {}),
            ("mpvit_small", {}),          # fallback común si lo usas como backbone
        ]

        # mpvit_small a veces está en networks.mpvit
        enc = None
        for name, kwargs in ENCODER_CANDIDATES:
            cls = getattr(networks, name, None)
            if cls is None and name == "mpvit_small":
                try:
                    from networks.mpvit import mpvit_small
                    cls = mpvit_small
                except Exception:
                    cls = None
            if cls is None:
                continue
            try:
                enc = cls(**kwargs)
            except TypeError:
                try:
                    enc = cls()
                except Exception:
                    enc = None
            if enc is not None:
                break

        if enc is None:
            avail = [n for n in dir(networks) if "Encoder" in n or "Net" in n or "vit" in n.lower()]
            raise RuntimeError(
                "No se pudo construir encoder EndoDAC/ViT. "
                "Edita ENCODER_CANDIDATES con el nombre real.\n"
                f"Disponibles en networks: {avail}"
            )

        # Decoder EndoDAC (DPT/Reassemble&Fusion). Ajusta nombres si aplica.
        DECODER_CANDIDATES = [
            ("DADepthDecoder", {"scales": range(4)}),
            ("DepthAnythingDecoder", {"scales": range(4)}),
            ("DPTDecoder", {"scales": range(4)}),
            ("DepthNetDecoder", {"scales": range(4)}),
            ("EndoDACDecoder", {"scales": range(4)}),
            ("DepthDecoder", {"scales": range(4)}),  # fallback si tu DepthDecoder soporta ViT feats
        ]

        dec = None
        for name, kwargs in DECODER_CANDIDATES:
            cls = getattr(networks, name, None)
            if cls is None:
                continue
            try:
                # algunos decoders requieren num_ch_enc
                if "num_ch_enc" in cls.__init__.__code__.co_varnames:
                    dec = cls(enc.num_ch_enc, **kwargs)
                else:
                    dec = cls(**kwargs)
            except TypeError:
                try:
                    dec = cls(enc.num_ch_enc)
                except Exception:
                    try:
                        dec = cls()
                    except Exception:
                        dec = None
            if dec is not None:
                break

        if dec is None:
            avail = [n for n in dir(networks) if "Decoder" in n or "DPT" in n or "Depth" in n]
            raise RuntimeError(
                "No se pudo construir decoder EndoDAC. "
                "Edita DECODER_CANDIDATES con el nombre real.\n"
                f"Disponibles en networks: {avail}"
            )

        # Load con strict=False para permitir LoRA / cambios menores de keys
        enc.load_state_dict(encoder_dict, strict=False)
        dec.load_state_dict(depth_dict, strict=False)

        encoder = enc
        depth_decoder = dec

    else:
        # --- ResNet (AF-SfMLearner clásico) ---
        encoder = networks.ResnetEncoder(num_layers, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(4))

        encoder_dict = torch.load(encoder_path, map_location=device)
        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict}, strict=False)
        depth_decoder.load_state_dict(depth_dict, strict=False)

    encoder.to(device).eval()
    depth_decoder.to(device).eval()
    return encoder, depth_decoder
# ===================== FIN MOD ENDODAC =====================


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


def _build_img_path(root, ds, keyf, frame_idx, png=False):
    """
    Construye la ruta real:
      <root>/<dataset>/<keyframe>/data/<frame>.<ext>
    """
    ext = ".png" if png else ".jpg"
    return os.path.join(root, ds, keyf, "data", f"{frame_idx}{ext}")


class SimpleImageDataset(torch.utils.data.Dataset):
    """Dataset mínimo que toma rutas absolutas ya resueltas."""
    def __init__(self, paths, height, width):
        self.paths = paths
        self.h = height
        self.w = width

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = PILImage.open(self.paths[idx]).convert('RGB')
        img = img.resize((self.w, self.h), PILImage.LANCZOS)
        img = np.asarray(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # C,H,W
        return {("color", 0, 0): img}


def map_split_to_existing_paths(data_path_root, filenames, png=False, strict=False):
    """
    Traduce cada línea tokenizada del split a una ruta real y
    - strict=False: devuelve SOLO las que existan (lenient).
    - strict=True : exige que existan TODAS; si falta alguna, lanza FileNotFoundError.

    Returns:
      idx_keep: indices del split que sí existen
      real_paths: rutas absolutas correspondientes
    """
    idx_keep, real_paths, missing = [], [], []
    for i, line in enumerate(filenames):
        try:
            ds, keyf, frame_idx, side = _parse_split_line(line)
            p = _build_img_path(data_path_root, ds, keyf, frame_idx, png=png)
            if os.path.isfile(p):
                idx_keep.append(i)
                real_paths.append(p)
            else:
                missing.append(line.strip())
        except Exception:
            missing.append(line.strip())

    if strict and len(missing) > 0:
        first = missing[0] if len(missing) else "N/A"
        raise FileNotFoundError(
            f"[STRICT] En {data_path_root} faltan {len(missing)} entradas del split. "
            f"Primera ausente: {first}"
        )

    return idx_keep, real_paths, missing


def evaluate_one_root(data_path_root,
                      filenames,
                      gt_depths,
                      encoder,
                      depth_decoder,
                      height=256,
                      width=320,
                      batch_size=16,
                      num_workers=4,
                      png=False,
                      disable_median_scaling=False,
                      pred_depth_scale_factor=1.0,
                      strict=False,
                      device="cuda"):
    """
    Evalúa una raíz (p.ej., .../brightness/severity_1/endovis_data) usando
    EXACTAMENTE el parser de rutas de SCAREDRAWDataset. Evita DataLoader
    para poder saltar muestras que falten o fallen al cargar.
    - strict=True  -> exige que TODAS las entradas del split carguen; de lo contrario lanza.
    - strict=False -> procesa sólo las que se puedan cargar (lenient).
    """
    img_ext = '.png' if png else '.jpg'
    try:
        dataset = SCAREDRAWDataset(
            data_path_root, filenames, height, width,
            [0], 4, is_train=False, img_ext=img_ext
        )
    except Exception as e:
        raise RuntimeError(f"No se pudo inicializar SCAREDRAWDataset en {data_path_root}: {e}")

    n = len(filenames)
    kept_indices = []
    preds_list   = []

    buffer_imgs = []
    buffer_ids  = []

    # ===================== MOD ENDODAC: inferencia tolerante a outputs ViT =====================
    def _get_disp0(out):
        """Extrae disp a resolución 0 desde dict/list/tensor."""
        if isinstance(out, dict):
            if ("disp", 0) in out:
                return out[("disp", 0)]
            if "disp" in out:
                return out["disp"]
            if ("pred_disp", 0) in out:
                return out[("pred_disp", 0)]
        if isinstance(out, (list, tuple)):
            return out[0]
        return out  # tensor directo

    def flush_buffer():
        """Corre inferencia sobre el buffer actual y guarda las disps."""
        if len(buffer_imgs) == 0:
            return
        with torch.no_grad():
            batch = torch.stack(buffer_imgs, dim=0).to(device)  # [B,3,H,W]
            feats = encoder(batch)
            out   = depth_decoder(feats)
            disp0 = _get_disp0(out)
            pred_disp, _ = disp_to_depth(disp0, MIN_DEPTH, MAX_DEPTH)
            # aseguramos shape [B,H,W]
            if pred_disp.ndim == 4:
                preds_list.append(pred_disp[:, 0].cpu().numpy())
            else:
                preds_list.append(pred_disp.cpu().numpy())
    # ===================== FIN MOD ENDODAC =====================

    missing = 0
    for i in range(n):
        try:
            sample = dataset[i]
            img_t  = sample[("color", 0, 0)]
            if not isinstance(img_t, torch.Tensor):
                img_t = torch.as_tensor(img_t)
            buffer_imgs.append(img_t)
            buffer_ids.append(i)

            if len(buffer_imgs) == batch_size:
                flush_buffer()
                kept_indices.extend(buffer_ids)
                buffer_imgs.clear()
                buffer_ids.clear()

        except FileNotFoundError:
            missing += 1
            if strict:
                raise FileNotFoundError(
                    f"[STRICT] Falta la muestra del split idx={i} en {data_path_root}"
                )
        except Exception as e:
            missing += 1
            if strict:
                raise RuntimeError(
                    f"[STRICT] Error cargando idx={i} en {data_path_root}: {e}"
                )

    flush_buffer()
    kept_indices.extend(buffer_ids)

    if len(kept_indices) == 0:
        mode = "STRICT" if strict else "LENIENT"
        raise FileNotFoundError(
            f"[{mode}] Ninguna imagen utilizable en {data_path_root} con el parser del dataset "
            f"(faltantes/errores: {missing}/{n})."
        )

    if (not strict) and missing > 0:
        print(f"   [INFO] {data_path_root}: usando {len(kept_indices)}/{n} frames del split "
              f"(faltaron {missing}).")

    pred_disps = np.concatenate(preds_list, axis=0)
    sel_gt     = gt_depths[kept_indices]

    errors, ratios = [], []
    for i in range(pred_disps.shape[0]):
        gt_depth = sel_gt[i]
        gt_h, gt_w = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_w, gt_h))
        pred_depth = 1.0 / (pred_disp + 1e-8)

        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        pd = pred_depth[mask]
        gd = gt_depth[mask]

        if pred_depth_scale_factor != 1.0:
            pd *= pred_depth_scale_factor

        if not disable_median_scaling:
            ratio = np.median(gd) / (np.median(pd) + 1e-8)
            ratios.append(ratio)
            pd *= ratio

        pd[pd < MIN_DEPTH] = MIN_DEPTH
        pd[pd > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gd, pd))

    if not disable_median_scaling and len(ratios) > 0:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(f"    Scaling ratios | med: {med:0.3f} | std: {np.std(ratios / med):0.3f}")

    mean_errors = np.array(errors).mean(0)
    return mean_errors


def list_corruption_dirs(root):
    """
    Devuelve los directorios de primer nivel que representan corrupciones.
    Si 'root' ya es una carpeta de una corrupción (que contiene severity_*), la devuelve tal cual.
    """
    if not os.path.isdir(root):
        return []
    severities = [d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)) and d.startswith("severity_")]
    if len(severities) > 0:
        return [root]
    return [os.path.join(root, d) for d in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, d))]


def main():
    parser = argparse.ArgumentParser("Evaluate EndoVIS corruptions (16x5) with AF-SfMLearner weights")
    parser.add_argument("--corruptions_root", type=str, required=True,
                        help="Raíz de las corrupciones (o una sola corrupción). Ej: /workspace/endovis_corruptions_test")
    parser.add_argument("--load_weights_folder", type=str, required=True,
                        help="Carpeta con encoder.pth y depth.pth")
    parser.add_argument("--splits_dir", type=str, default=os.path.join(os.path.dirname(__file__), "splits"))
    parser.add_argument("--split", type=str, default="endovis", help="Nombre del split (carpeta dentro de splits/)")
    parser.add_argument("--num_layers", type=int, default=18)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--png", action="store_true", help="Usa .png en lugar de .jpg")
    parser.add_argument("--eval_stereo", action="store_true", help="Forzar estéreo (desactiva median scaling y usa x5.4)")
    parser.add_argument("--output_csv", type=str, default="corruptions_summary.csv")
    parser.add_argument("--strict", action="store_true",
                        help="Modo estricto: exige que todas las entradas del split existan en cada severidad.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cv2.setNumThreads(0)

    test_files_path = os.path.join(args.splits_dir, args.split, "test_files.txt")
    if not os.path.isfile(test_files_path):
        raise FileNotFoundError(f"No se encontró test_files.txt en {test_files_path}")

    test_files = readlines(test_files_path)
    gt_path = os.path.join(args.splits_dir, args.split, "gt_depths.npz")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"No se encontró gt_depths.npz en {gt_path}")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

    if len(test_files) != gt_depths.shape[0]:
        print("[WARN] test_files.txt y gt_depths.npz difieren en longitud. "
              "Esto es típico cuando hay líneas de split inválidas o extra. "
              "El script filtrará por líneas parseables y existentes por severidad.")

    disable_median_scaling = args.eval_stereo
    pred_depth_scale_factor = STEREO_SCALE_FACTOR if args.eval_stereo else 1.0

    print("-> Cargando pesos:", args.load_weights_folder)
    encoder, depth_decoder = load_model(args.load_weights_folder, args.num_layers, device)

    corr_dirs = list_corruption_dirs(args.corruptions_root)
    if len(corr_dirs) == 0:
        raise FileNotFoundError(f"No se encontraron carpetas de corrupción en {args.corruptions_root}")

    rows = []
    print("-> Iniciando evaluación de corrupciones")
    for corr_dir in corr_dirs:
        corr_name = os.path.basename(corr_dir.rstrip("/"))
        severities = sorted([d for d in os.listdir(corr_dir)
                             if os.path.isdir(os.path.join(corr_dir, d)) and d.startswith("severity_")],
                            key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else 9999)

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
                    encoder=encoder,
                    depth_decoder=depth_decoder,
                    height=args.height,
                    width=args.width,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    png=args.png,
                    disable_median_scaling=disable_median_scaling,
                    pred_depth_scale_factor=pred_depth_scale_factor,
                    strict=args.strict,
                    device=device
                )
                abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = mean_errors.tolist()
                rows.append([corr_name, sev, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3])

                print("   Métricas (promedio): "
                      f"abs_rel={abs_rel:.3f} | sq_rel={sq_rel:.3f} | rmse={rmse:.3f} | "
                      f"rmse_log={rmse_log:.3f} | a1={a1:.3f} | a2={a2:.3f} | a3={a3:.3f}")

            except FileNotFoundError as e:
                print(f"   [SKIP] {e}")

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
                bucket[corr], key=lambda x: int(x[1].split('_')[-1]) if x[1].split('_')[-1].isdigit() else 9999
            ):
                print(f"{sev:>9} | {abs_rel:7.3f} | {sq_rel:7.3f} | {rmse:7.3f} |  {rmse_log:7.3f} | "
                      f"{a1:6.3f} | {a2:6.3f} | {a3:6.3f}")
    else:
        print("\n-> No se generaron filas. Revisa rutas/archivos faltantes o estructura de corrupciones.")

if __name__ == "__main__":
    main()
