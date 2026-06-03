"""Verify LoveDA directory structure and print dataset statistics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loveda import LoveDADataset

ROOT = Path(__file__).resolve().parent.parent / 'data' / 'LoveDA'


def check():
    ok = True
    for split in ('train', 'val'):
        # Accept both 'train'/'val' and 'Train'/'Val'
        split_dir = None
        for candidate in (split, split.capitalize()):
            p = ROOT / candidate
            if p.exists():
                split_dir = p
                break
        if split_dir is None:
            print(f"  MISSING split dir for '{split}' under {ROOT}")
            ok = False
            continue
        for domain in ('Urban', 'Rural'):
            img_dir  = split_dir / domain / 'images_png'
            mask_dir = split_dir / domain / 'masks_png'
            if not img_dir.exists():
                print(f"  MISSING: {img_dir}")
                ok = False
                continue
            imgs  = list(img_dir.glob('*.png'))
            masks = list(mask_dir.glob('*.png')) if mask_dir.exists() else []
            print(f"  {split_dir.name}/{domain}: {len(imgs)} images, {len(masks)} masks")
    return ok


def main():
    print("=== Directory check ===")
    if not check():
        print("\nDataset not ready yet.")
        sys.exit(1)

    print("\n=== Dataset loader check ===")
    for split, labels in [('train', False), ('val', True)]:
        ds = LoveDADataset(str(ROOT), split=split, domain='all', return_labels=labels)
        img, lbl = ds[0]
        print(f"  {split}: {len(ds)} samples | "
              f"image {tuple(img.shape)} | "
              f"{'mask ' + str(tuple(lbl.shape)) if labels else 'idx=' + str(lbl)}")

    print("\nLoveDA OK — ready to train.")


if __name__ == '__main__':
    main()
