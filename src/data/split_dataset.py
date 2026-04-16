import argparse
import logging
import random
import shutil
from pathlib import Path

import pandas as pd

# ===================== CONFIG =====================
SOURCE_DIR = "data/raw"          # thư mục gốc chứa các folder class
OUTPUT_DIR = "data/splits"       # thư mục output sau khi chia

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

SEED        = 42
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COPY_FILES  = True              # True: copy, False: move
SAVE_CSV    = True              # xuất file split_metadata.csv
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────── helpers ───────────────────────

def is_image_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in IMAGE_EXTS


def validate_ratios(train: float, val: float, test: float) -> None:
    for name, value in [("TRAIN_RATIO", train), ("VAL_RATIO", val), ("TEST_RATIO", test)]:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} phải nằm trong [0, 1], nhận được: {value}")
    total = train + val + test
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            f"Tổng TRAIN_RATIO + VAL_RATIO + TEST_RATIO phải bằng 1.0, "
            f"nhận được: {total:.6f}"
        )


def get_class_names(source_dir: Path) -> list[str]:
    classes = sorted(d.name for d in source_dir.iterdir() if d.is_dir())
    return classes


def make_output_structure(output_dir: Path, class_names: list[str]) -> None:
    for split in ("train", "val", "test"):
        for class_name in class_names:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)


def prepare_output_dir(output_dir: Path, force_overwrite: bool) -> None:
    """Xóa output cũ nếu tồn tại; hỏi user nếu không dùng --force."""
    if not output_dir.exists():
        return

    if force_overwrite:
        logger.warning("Output dir '%s' đã tồn tại → xóa và tạo lại.", output_dir)
        shutil.rmtree(output_dir)
    else:
        answer = input(
            f"\n⚠️  Output dir '{output_dir}' đã tồn tại.\n"
            "Xóa và tạo lại? [y/N]: "
        ).strip().lower()
        if answer == "y":
            shutil.rmtree(output_dir)
            logger.info("Đã xóa output dir cũ.")
        else:
            raise SystemExit("Hủy. Chạy lại với --output trỏ đến thư mục khác hoặc dùng --force.")


# ─────────────────────── core split ───────────────────────

def split_one_class(
    file_list: list[str],
    train_ratio: float = 0.8,
    val_ratio: float   = 0.1,
    test_ratio: float  = 0.1,
) -> tuple[list[str], list[str], list[str]]:
    """Chia file list thành train / val / test, xử lý đúng mọi kích thước."""
    random.shuffle(file_list)
    n = len(file_list)

    if n == 0:
        return [], [], []

    if n == 1:
        # 1 ảnh → chỉ đưa vào train
        return file_list, [], []

    if n == 2:
        # 2 ảnh → 1 train, 1 val (không có test)
        return [file_list[0]], [file_list[1]], []

    # n >= 3: chia theo tỉ lệ, đảm bảo val và test mỗi bên ≥ 1
    train_count = int(n * train_ratio)
    val_count   = int(n * val_ratio)
    test_count  = n - train_count - val_count

    if val_ratio > 0 and val_count == 0:
        val_count   += 1
        train_count -= 1
    if test_ratio > 0 and test_count == 0:
        test_count  += 1
        train_count -= 1

    train_count = max(train_count, 0)
    val_count   = max(val_count,   0)
    test_count  = max(test_count,  0)

    # cân lại nếu tổng lệch do int-truncation
    diff = n - (train_count + val_count + test_count)
    train_count += diff

    return (
        file_list[:train_count],
        file_list[train_count : train_count + val_count],
        file_list[train_count + val_count :],
    )


def transfer_files(
    files: list[str],
    src_dir: Path,
    dst_dir: Path,
    copy_files: bool = True,
) -> None:
    for file_name in files:
        src = src_dir / file_name
        dst = dst_dir / file_name
        if copy_files:
            shutil.copy2(src, dst)
        else:
            shutil.move(str(src), dst)


# ─────────────────────── reporting ───────────────────────

def print_summary(summary: dict[str, dict]) -> None:
    logger.info("========== SPLIT SUMMARY ==========")
    for class_name, s in summary.items():
        logger.info(
            "%-14s total=%4d | train=%4d | val=%4d | test=%4d",
            class_name, s["total"], s["train"], s["val"], s["test"],
        )
    logger.info("====================================")


# ─────────────────────── main ───────────────────────

def main(
    source_dir: Path,
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    copy_files: bool,
    save_csv: bool,
    force_overwrite: bool,
    dry_run: bool,
    seed: int,
) -> None:
    validate_ratios(train_ratio, val_ratio, test_ratio)
    random.seed(seed)

    if not source_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy SOURCE_DIR: {source_dir}")

    class_names = get_class_names(source_dir)
    if not class_names:
        raise ValueError("Không tìm thấy folder class nào trong SOURCE_DIR.")

    logger.info("Classes found: %s", class_names)

    if dry_run:
        logger.info("[DRY-RUN] Chỉ hiển thị kế hoạch, không copy/move file.")
    else:
        prepare_output_dir(output_dir, force_overwrite)
        make_output_structure(output_dir, class_names)

    # Cảnh báo khi dùng move để người dùng biết rủi ro mất dữ liệu gốc
    if not copy_files and not dry_run:
        logger.warning(
            "COPY_FILES=False: file gốc sẽ bị di chuyển ra khỏi '%s'. "
            "Hãy đảm bảo bạn đã backup trước!",
            source_dir,
        )

    records: list[dict] = []
    summary: dict[str, dict] = {}

    for class_name in class_names:
        class_dir    = source_dir / class_name
        image_files  = sorted(
            f.name for f in class_dir.iterdir()
            if f.is_file() and is_image_file(f)
        )

        train_files, val_files, test_files = split_one_class(
            image_files, train_ratio, val_ratio, test_ratio
        )

        summary[class_name] = {
            "total": len(image_files),
            "train": len(train_files),
            "val":   len(val_files),
            "test":  len(test_files),
        }

        if not dry_run:
            transfer_files(train_files, class_dir, output_dir / "train" / class_name, copy_files)
            transfer_files(val_files,   class_dir, output_dir / "val"   / class_name, copy_files)
            transfer_files(test_files,  class_dir, output_dir / "test"  / class_name, copy_files)

        for split_name, split_files in (
            ("train", train_files),
            ("val",   val_files),
            ("test",  test_files),
        ):
            for fname in split_files:
                records.append({
                    "file_name": fname,
                    "label":     class_name,
                    "split":     split_name,
                    "path":      str(output_dir / split_name / class_name / fname),
                })

    print_summary(summary)

    if save_csv and not dry_run:
        df       = pd.DataFrame(records)
        csv_path = output_dir / "split_metadata.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info("Đã lưu file metadata: %s", csv_path)


# ─────────────────────── CLI ───────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chia dataset ảnh thành train / val / test."
    )
    parser.add_argument("--source",  default=SOURCE_DIR,   help="Thư mục dataset gốc")
    parser.add_argument("--output",  default=OUTPUT_DIR,   help="Thư mục output")
    parser.add_argument("--train",   type=float, default=TRAIN_RATIO)
    parser.add_argument("--val",     type=float, default=VAL_RATIO)
    parser.add_argument("--test",    type=float, default=TEST_RATIO)
    parser.add_argument("--seed",    type=int,   default=SEED)
    parser.add_argument("--move",    action="store_true",  help="Di chuyển thay vì copy")
    parser.add_argument("--no-csv",  action="store_true",  help="Không xuất CSV")
    parser.add_argument("--force",   action="store_true",  help="Tự động xóa output dir cũ")
    parser.add_argument("--dry-run", action="store_true",  help="Chạy thử, không thay đổi file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        source_dir      = Path(args.source),
        output_dir      = Path(args.output),
        train_ratio     = args.train,
        val_ratio       = args.val,
        test_ratio      = args.test,
        copy_files      = not args.move,
        save_csv        = not args.no_csv,
        force_overwrite = args.force,
        dry_run         = args.dry_run,
        seed            = args.seed,
    )