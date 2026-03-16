import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image
import pillow_heif

# =========================================================
# 1. 核心映射：错误读取的三通道 -> 正确 RGB
#    已知映射：
#    Y  = G
#    Cb = B
#    Cr = R
#    使用 full-range YCbCr -> RGB
# =========================================================

def recover_rgb_from_misread_heif(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)

    Y = arr[:, :, 1]   # G
    Cb = arr[:, :, 2]  # B
    Cr = arr[:, :, 0]  # R

    R = Y + 1.402 * (Cr - 128.0)
    G = Y - 0.344136 * (Cb - 128.0) - 0.714136 * (Cr - 128.0)
    B = Y + 1.772 * (Cb - 128.0)

    rgb = np.stack([R, G, B], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


# =========================================================
# 2. 读取 HEIF/HEIC
# =========================================================
def read_heif_as_array(path: Path) -> np.ndarray:
    heif_file = pillow_heif.open_heif(str(path), convert_hdr_to_8bit=True)
    img = heif_file.to_pillow().convert("RGB")
    return np.asarray(img)


# =========================================================
# 3. 保存输出
# =========================================================
def save_image(
    rgb: np.ndarray,
    output_path: Path,
    output_format: str,
    heif_quality: int = -1,
    png_compress_level: int = 9,
):
    img = Image.fromarray(rgb)

    fmt = output_format.lower()

    if fmt in ("heif", "heic"):
        img.save(
            str(output_path),
            format="HEIF",
            quality=heif_quality,
            chroma="444",
            matrix_coefficients=0,
        )
    elif fmt == "png":
        img.save(
            str(output_path),
            format="PNG",
            optimize=True,
            compress_level=png_compress_level,
        )
    else:
        raise ValueError(f"不支持的输出格式: {output_format}")


# =========================================================
# 4. 收集输入文件
# =========================================================
def collect_input_files(input_dir: Path, recursive: bool):
    exts = {".heic", ".heif", ".HEIC", ".HEIF"}
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix in exts]
    else:
        files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix in exts]
    return sorted(files)


# =========================================================
# 5. 构造输出路径
# =========================================================
def build_output_path(
    input_path: Path,
    input_root: Path,
    output_root: Path,
    output_format: str,
    keep_subdir_structure: bool,
):
    suffix = ".HEIF" if output_format.lower() in ("heif", "heic") else ".png"

    if keep_subdir_structure:
        rel_path = input_path.relative_to(input_root)
        out_path = output_root / rel_path
        out_path = out_path.with_suffix(suffix)
    else:
        out_path = output_root / f"{input_path.stem}{suffix}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


# =========================================================
# 6. 单文件处理
# =========================================================
def process_one_file(task):
    (
        input_path_str,
        input_root_str,
        output_root_str,
        output_format,
        heif_quality,
        png_compress_level,
        keep_subdir_structure,
        overwrite,
    ) = task

    input_path = Path(input_path_str)
    input_root = Path(input_root_str)
    output_root = Path(output_root_str)

    try:
        output_path = build_output_path(
            input_path=input_path,
            input_root=input_root,
            output_root=output_root,
            output_format=output_format,
            keep_subdir_structure=keep_subdir_structure,
        )

        if output_path.exists() and not overwrite:
            return True, str(input_path), str(output_path), "跳过（已存在）"

        arr = read_heif_as_array(input_path)
        rgb = recover_rgb_from_misread_heif(arr)

        save_image(
            rgb=rgb,
            output_path=output_path,
            output_format=output_format,
            heif_quality=heif_quality,
            png_compress_level=png_compress_level,
        )

        return True, str(input_path), str(output_path), "完成"

    except Exception as e:
        return False, str(input_path), "", str(e)


# =========================================================
# 7. 读取配置文件
# =========================================================
def load_config(config_path="config.json"):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    return cfg


# =========================================================
# 8. 主流程
# =========================================================
def main():
    cfg = load_config("config.json")

    input_dir = Path(cfg["input_dir"])
    output_dir = Path(cfg["output_dir"])

    output_format = cfg.get("output_format", "heif")
    workers = cfg.get("workers", 0)
    heif_quality = cfg.get("heif_quality", -1)
    png_compress_level = cfg.get("png_compress_level", 9)
    recursive = cfg.get("recursive", False)
    keep_subdir_structure = cfg.get("keep_subdir_structure", False)
    overwrite = cfg.get("overwrite", True)

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    files = collect_input_files(input_dir, recursive=recursive)
    if not files:
        print("未找到 HEIC/HEIF 文件。")
        return

    if workers is None or workers <= 0:
        workers = os.cpu_count() or 4

    tasks = [
        (
            str(f),
            str(input_dir),
            str(output_dir),
            output_format,
            heif_quality,
            png_compress_level,
            keep_subdir_structure,
            overwrite,
        )
        for f in files
    ]

    print("========== 配置信息 ==========")
    print("输入目录:", input_dir)
    print("输出目录:", output_dir)
    print("输出格式:", output_format)
    print("文件数量:", len(files))
    print("并行进程数:", workers)
    print("递归处理:", recursive)
    print("保持子目录结构:", keep_subdir_structure)
    print("覆盖输出:", overwrite)
    print("==============================")

    ok_count = 0
    fail_count = 0
    skip_count = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_one_file, t) for t in tasks]

        for fut in as_completed(futures):
            ok, in_file, out_file, msg = fut.result()
            if ok:
                if msg.startswith("跳过"):
                    skip_count += 1
                    print(f"[跳过] {in_file} -> {out_file}")
                else:
                    ok_count += 1
                    print(f"[成功] {in_file} -> {out_file}")
            else:
                fail_count += 1
                print(f"[失败] {in_file}")
                print(f"       错误: {msg}")

    print("\n处理完成")
    print("成功:", ok_count)
    print("跳过:", skip_count)
    print("失败:", fail_count)


if __name__ == "__main__":
    main()