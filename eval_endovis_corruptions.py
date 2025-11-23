# eval_endovis_corruptions.py
from __future__ import absolute_import, division, print_function

import os
import argparse
import csv
import numpy as np
import cv2
from collections import defaultdict

import torch

from datasets import SCAREDRAWDataset

# ---------- imports alineados a tu estructura utils/ ----------
# readlines + compute_errors están en utils/utils.py (según tu archivo)
try:
    from utils.utils import readlines, compute_errors
except ImportError:
    from utils import readlines, compute_errors

# disp_to_depth está en utils/layers.py (según tu archivo)
try:
    from utils.layers import disp_to_depth
except ImportError:
    from utils import disp_to_depth

try:
    from PIL import Image as PILImage
except Exception as e:
    raise ImportError("Pillow es requerido: pip install pillow") from e


# ===== Constantes/metas =====
STEREO_SCALE_FACTOR = 5.4
MIN_DEPTH = 1e-3
MAX_DEPTH = 150.0


# ===================== ENDODAC LOADER (solo models/, nombres reales) =====================
def load_model(load_weights_folder, num_layers, device):
    """
    Carga pesos Endo-DAC (DepthAnything ViT + depth_model).
    Nombres soportados:
      - encoder.pth / depth.pth (si existieran)
      - depth_anything*.pth (encoder)
      - depth_model.pth (depth head EndoDAC)
    No depende de networks/ porque en tu estructura no existe.
    """
    import importlib, pkgutil, inspect

    if not os.path.isdir(load_weights_folder):
        raise FileNotFoundError(f"Cannot find weights folder: {load_weights_folder}")

    # ---- resolver nombres de archivo ----
    encoder_candidates = [
        "encoder.pth",
        "depth_anything_vitb14.pth",
    ]
    encoder_candidates += sorted([
        f for f in os.listdir(load_weights_folder)
        if f.startswith("depth_anything") and f.endswith(".pth")
    ])

    decoder_candidates = [
        "depth.pth",
        "depth_model.pth",
    ]

    encoder_path = None
    for name in encoder_candidates:
        p = os.path.join(load_weights_folder, name)
        if os.path.isfile(p):
            encoder_path = p
            break

    decoder_path = None
    for name in decoder_candidates:
        p = os.path.join(load_weights_folder, name)
        if os.path.isfile(p):
            decoder_path = p
            break

    if encoder_path is None or decoder_path is None:
        raise FileNotFoundError(
            "Missing EndoDAC weights. Busqué encoder en: "
            f"{encoder_candidates} y decoder en: {decoder_candidates} dentro de {load_weights_folder}"
        )

    print(f"   [INFO] Encoder weights: {os.path.basename(encoder_path)}")
    print(f"   [INFO] Decoder weights: {os.path.basename(decoder_path)}")

    encoder_dict = torch.load(encoder_path, map_location=device)
    depth_dict   = torch.load(decoder_path, map_location=device)

    enc_keys = list(encoder_dict.keys())
    dec_keys = list(depth_dict.keys())

    def _looks_like_vit_state_dict(sd_keys):
        vit_hints = ("patch_embed", "pos_embed", "blocks.", "transformer", "attn", "mlp", "norm")
        return any(any(h in k for h in vit_hints) for k in sd_keys)

    def _looks_like_endodac_decoder(sd_keys):
        dec_hints = ("endodac", "reassemble", "fusion", "dpt", "head", "upsample", "refine")
        return any(any(h in k.lower() for h in dec_hints) for k in sd_keys)

    is_vit = _looks_like_vit_state_dict(enc_keys) or _looks_like_endodac_decoder(dec_keys)

    # -------- auto-descubrir clases en models/ ----------
    def iter_candidate_classes(patterns):
        patterns = [p.lower() for p in patterns]
        try:
            pkg = importlib.import_module("models")
        except Exception:
            return

        if hasattr(pkg, "__path__"):
            for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
                try:
                    mod = importlib.import_module(m.name)
                except Exception:
                    continue
                for cname, cls in inspect.getmembers(mod, inspect.isclass):
                    if cls.__module__ == mod.__name__ and any(p in cname.lower() for p in patterns):
                        yield cls

    def try_instantiate(cls, prefer_num_ch_enc=None):
        try:
            return cls()
        except Exception:
            pass
        try:
            return cls(scales=range(4))
        except Exception:
            pass
        if prefer_num_ch_enc is not None:
            try:
                return cls(prefer_num_ch_enc, scales=range(4))
            except Exception:
                pass
            try:
                return cls(prefer_num_ch_enc)
            except Exception:
                pass
        return None
    # ----------------------------------------------------

    if is_vit:
        # ======= EndoDAC/ViT encoder + modelo depth =======
        encoder_patterns = ["depthanything", "vit", "transformer", "encoder", "backbone"]
        decoder_patterns = ["endodac", "depthmodel", "dpt", "decoder", "depth"]

        enc = None
        for cls in iter_candidate_classes(encoder_patterns):
            if "resnet" in cls.__name__.lower():
                continue
            enc = try_instantiate(cls)
            if enc is not None:
                break
        if enc is None:
            raise RuntimeError("No se pudo construir encoder ViT desde models/.")

        enc.load_state_dict(encoder_dict, strict=False)

        dec = None
        prefer_num_ch_enc = getattr(enc, "num_ch_enc", None)
        for cls in iter_candidate_classes(decoder_patterns):
            if "resnet" in cls.__name__.lower():
                continue
            dec = try_instantiate(cls, prefer_num_ch_enc=prefer_num_ch_enc)
            if dec is not None:
                break
        if dec is None:
            raise RuntimeError("No se pudo construir modelo/depth head EndoDAC desde models/.")

        dec.load_state_dict(depth_dict, strict=False)

        encoder = enc
        depth_decoder = dec

    else:
        raise RuntimeError(
            "Los pesos no parecen ViT/EndoDAC y no hay networks/. "
            "Este script está pensado para Endo-DAC."
        )

    encoder.to(device).eval()
    depth_decoder.to(device).eval()
    return encoder, depth_decoder
# ===================== FIN ENDODAC LOADER =====================


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
    Evalúa una raíz (corrupción/severidad) con EndoDAC.
    """
    import inspect as pyinspect

    img_ext = '.png' if png else '.jpg'
    dataset = SCAREDRAWDataset(
        data_path_root, filenames, height, width,
        [0], 4, is_train=False, img_ext=img_ext
    )

    n = len(filenames)
    kept_indices = []
    preds_list   = []

    buffer_imgs = []
    buffer_ids  = []

    def _get_disp0(out):
        """Extrae disp escala 0 desde dict/list/tensor."""
        if isinstance(out, dict):
            if ("disp", 0) in out:
                return out[("disp", 0)]
            if "disp" in out:
                return out["disp"]
            if ("pred_disp", 0) in out:
                return out[("pred_disp", 0)]
        if isinstance(out, (list, tuple)):
            return out[0]
        return out

    def flush_buffer():
        if len(buffer_imgs) == 0:
            return

        with torch.no_grad():
            batch = torch.stack(buffer_imgs, dim=0).to(device)  # [B,3,H,W]

            # feats opcionales (pero EndoDAC winner suele NO usarlos en forward)
            feats = None
            if encoder is not None:
                try:
                    feats = encoder(batch)
                except Exception:
                    feats = None

            # firma del forward del modelo EndoDAC
            try:
                sig = pyinspect.signature(depth_decoder.forward)
                params = [p.name for p in sig.parameters.values() if p.name != "self"]
            except Exception:
                params = []

            # --- Caso EndoDAC típico: forward(pixel_values) ---
            if len(params) == 1:
                out = depth_decoder(batch)
            else:
                # si el modelo acepta más args, intentamos pasar imagen+feats
                called = False
                if feats is None:
                    out = depth_decoder(batch); called = True
                else:
                    if len(params) >= 2:
                        p0, p1 = params[0].lower(), params[1].lower()
                        if ("pixel" in p0) or ("image" in p0) or ("rgb" in p0):
                            out = depth_decoder(batch, feats); called = True
                        elif ("pixel" in p1) or ("image" in p1) or ("rgb" in p1):
                            out = depth_decoder(feats, batch); called = True

                    if not called:
                        try:
                            out = depth_decoder(feats); called = True
                        except Exception:
                            pass
                    if not called:
                        try:
                            out = depth_decoder(batch, feats)
                        except Exception:
                            out = depth_decoder(feats, batch)

            disp0 = _get_disp0(out)
            pred_disp, _ = disp_to_depth(disp0, MIN_DEPTH, MAX_DEPTH)

            if pred_disp.ndim == 4:
                preds_list.append(pred_disp[:, 0].cpu().numpy())
            else:
                preds_list.append(pred_disp.cpu().numpy())

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

        except Exception:
            missing += 1
            if strict:
                raise

    flush_buffer()
    kept_indices.extend(buffer_ids)

    if len(kept_indices) == 0:
        mode = "STRICT" if strict else "LENIENT"
        raise FileNotFoundError(
            f"[{mode}] Ninguna imagen utilizable en {data_path_root} "
            f"(faltantes/errores: {missing}/{n})."
        )

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

        pd = np.clip(pd, MIN_DEPTH, MAX_DEPTH)

        errors.append(compute_errors(gd, pd))

    if not disable_median_scaling and len(ratios) > 0:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(f"    Scaling ratios | med: {med:0.3f} | std: {np.std(ratios / med):0.3f}")

    return np.array(errors).mean(0)


def list_corruption_dirs(root):
    if not os.path.isdir(root):
        return []
    severities = [d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)) and d.startswith("severity_")]
    if len(severities) > 0:
        return [root]
    return [os.path.join(root, d) for d in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, d))]


def main():
    parser = argparse.ArgumentParser("Evaluate EndoVIS corruptions (16x5) with EndoDAC weights")
    parser.add_argument("--corruptions_root", type=str, required=True)
    parser.add_argument("--load_weights_folder", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, default=os.path.join(os.path.dirname(__file__), "splits"))
    parser.add_argument("--split", type=str, default="endovis")
    parser.add_argument("--num_layers", type=int, default=18)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--eval_stereo", action="store_true")
    parser.add_argument("--output_csv", type=str, default="corruptions_summary.csv")
    parser.add_argument("--strict", action="store_true")
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

    if rows:
        header = ["corruption", "severity", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
        with open(args.output_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

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
        print("\n-> No se generaron filas. Revisa estructura de corrupciones.")


if __name__ == "__main__":
    main()
