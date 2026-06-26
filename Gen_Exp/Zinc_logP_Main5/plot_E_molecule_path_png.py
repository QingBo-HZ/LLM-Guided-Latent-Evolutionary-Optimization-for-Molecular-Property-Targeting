#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Draw the representative molecule optimization path as a high-resolution PNG.

This avoids RDKit Cairo/SVG conversion dependencies by using RDKit 2D
coordinates and PIL drawing primitives.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType


OUT_DIR = Path("/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP_Main5/figures")
DATA_PATH = OUT_DIR / "E_representative_molecule_path_data.csv"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


FONT_TITLE = font(48, bold=True)
FONT_LEGEND = font(27, bold=True)
FONT_ATOM = font(26, bold=True)


def atom_label(atom: Chem.Atom) -> str:
    sym = atom.GetSymbol()
    charge = atom.GetFormalCharge()
    label = "" if sym == "C" and charge == 0 else sym
    if charge:
        sign = "+" if charge > 0 else "-"
        mag = abs(charge)
        label += sign if mag == 1 else f"{mag}{sign}"
    return label


def transform_points(mol: Chem.Mol, box_w: int, box_h: int, pad: int = 40):
    conf = mol.GetConformer()
    pts = []
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        pts.append((float(p.x), float(p.y)))

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-3)
    span_y = max(max_y - min_y, 1e-3)
    scale = min((box_w - 2 * pad) / span_x, (box_h - 2 * pad) / span_y)
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2

    out = []
    for x, y in pts:
        px = box_w / 2 + (x - cx) * scale
        py = box_h / 2 - (y - cy) * scale
        out.append((px, py))
    return out


def draw_parallel_line(draw: ImageDraw.ImageDraw, p1, p2, offset: float, fill, width: int):
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1.0
    ox, oy = -dy / length * offset, dx / length * offset
    draw.line((x1 + ox, y1 + oy, x2 + ox, y2 + oy), fill=fill, width=width)


def draw_bond(draw: ImageDraw.ImageDraw, p1, p2, bond: Chem.Bond):
    fill = (32, 32, 32)
    btype = bond.GetBondType()
    if btype == BondType.DOUBLE:
        draw_parallel_line(draw, p1, p2, 5.0, fill, 4)
        draw_parallel_line(draw, p1, p2, -5.0, fill, 4)
    elif btype == BondType.TRIPLE:
        draw.line((*p1, *p2), fill=fill, width=4)
        draw_parallel_line(draw, p1, p2, 8.0, fill, 3)
        draw_parallel_line(draw, p1, p2, -8.0, fill, 3)
    elif bond.GetIsAromatic():
        draw.line((*p1, *p2), fill=fill, width=4)
        draw_parallel_line(draw, p1, p2, 6.0, (120, 120, 120), 2)
    else:
        draw.line((*p1, *p2), fill=fill, width=4)


def draw_atom_label(draw: ImageDraw.ImageDraw, p, label: str):
    if not label:
        return
    x, y = p
    bbox = draw.textbbox((0, 0), label, font=FONT_ATOM)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 4
    rect = (x - w / 2 - pad, y - h / 2 - pad, x + w / 2 + pad, y + h / 2 + pad)
    draw.rounded_rectangle(rect, radius=5, fill=(255, 255, 255), outline=None)
    color = {
        "O": (210, 40, 40),
        "N": (40, 80, 200),
        "S": (184, 140, 20),
        "F": (20, 140, 65),
        "Cl": (20, 140, 65),
        "Br": (140, 70, 20),
        "I": (120, 40, 140),
        "P": (210, 100, 30),
    }.get(label.replace("+", "").replace("-", ""), (20, 20, 20))
    draw.text((x - w / 2, y - h / 2 - 1), label, font=FONT_ATOM, fill=color)


def draw_molecule(smiles: str, legend: str, box_w: int = 760, box_h: int = 470) -> Image.Image:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol)
    pts = transform_points(mol, box_w, box_h - 70, pad=46)

    img = Image.new("RGB", (box_w, box_h), "white")
    draw = ImageDraw.Draw(img)

    # Molecule drawing area starts at y=54.
    shifted = [(x, y + 55) for x, y in pts]
    for bond in mol.GetBonds():
        p1 = shifted[bond.GetBeginAtomIdx()]
        p2 = shifted[bond.GetEndAtomIdx()]
        draw_bond(draw, p1, p2, bond)

    for atom in mol.GetAtoms():
        draw_atom_label(draw, shifted[atom.GetIdx()], atom_label(atom))

    draw.rounded_rectangle((12, 12, box_w - 12, 48), radius=10, fill=(246, 248, 251), outline=(220, 224, 230), width=2)
    bbox = draw.textbbox((0, 0), legend, font=FONT_LEGEND)
    draw.text(((box_w - (bbox[2] - bbox[0])) / 2, 17), legend, font=FONT_LEGEND, fill=(35, 35, 35))
    return img


def main():
    df = pd.read_csv(DATA_PATH)
    cells = []
    for _, row in df.iterrows():
        legend = f"Gen {int(row['generation'])} | logP={row['rdkit_logP']:.3f} | err={row['rdkit_abs_error']:.4f}"
        cells.append(draw_molecule(row["smiles"], legend))

    cols = 3
    rows = math.ceil(len(cells) / cols)
    cell_w, cell_h = cells[0].size
    title_h = 90
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    title = "Representative Molecular Optimization Path"
    bbox = draw.textbbox((0, 0), title, font=FONT_TITLE)
    draw.text(((canvas.width - (bbox[2] - bbox[0])) / 2, 22), title, font=FONT_TITLE, fill=(20, 20, 20))

    for i, img in enumerate(cells):
        x = (i % cols) * cell_w
        y = title_h + (i // cols) * cell_h
        canvas.paste(img, (x, y))

    out = OUT_DIR / "E_representative_molecule_path.png"
    canvas.save(out, dpi=(300, 300))
    print(f"saved {out}")
    print(f"size_px={canvas.size}")


if __name__ == "__main__":
    main()
