import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import segyio


def parse_mapping(values: List[str]) -> Tuple[int, int, int]:
    parts: List[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if "," in text:
            parts.extend([p.strip() for p in text.split(",") if p.strip()])
        else:
            parts.append(text)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("mapping must be 2,1,0 or 2 1 0")
    try:
        axes = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("mapping must contain integers only") from exc
    if sorted(axes) != [0, 1, 2]:
        raise argparse.ArgumentTypeError("mapping must be a permutation of 0,1,2")
    return axes


def set_if_exists(header, trace_field, field_name: str, value: int) -> None:
    if hasattr(trace_field, field_name):
        header[getattr(trace_field, field_name)] = value


def xy_of(il: int, xl: int, x0: float, y0: float, dx: float, dy: float) -> Tuple[float, float]:
    x = x0 + (xl - 1) * dx
    y = y0 + (il - 1) * dy
    return x, y


def build_spec(n_il: int, n_xl: int, n_s: int) -> segyio.spec:
    spec = segyio.spec()
    spec.ilines = np.arange(1, n_il + 1, dtype=np.int32)
    spec.xlines = np.arange(1, n_xl + 1, dtype=np.int32)
    spec.samples = np.arange(n_s, dtype=np.int32)
    spec.format = 5
    return spec


def infer_reference_npy(input_npy: Path) -> Optional[Path]:
    repo_root = Path(__file__).resolve().parent
    candidates = []
    lower = str(input_npy).lower()

    if "stanford_vi" in lower:
        candidates.extend(
            [
                repo_root / "data" / "Stanford_VI" / "synth_40HZ.npy",
                repo_root / "data" / "Stanford_VI" / "AI.npy",
            ]
        )

    candidates.extend(
        [
            repo_root / "data" / "Stanford_VI" / "synth_40HZ.npy",
            repo_root / "data" / "Stanford_VI" / "AI.npy",
        ]
    )

    seen = set()
    for cand in candidates:
        key = str(cand).lower()
        if key in seen:
            continue
        seen.add(key)
        if cand.exists():
            return cand
    return None


def restore_cube_from_2d(
    arr2d: np.ndarray,
    ref_npy: Path,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    ref_cube = np.load(ref_npy, mmap_mode="r")
    if ref_cube.ndim != 3:
        raise ValueError(f"reference npy must be 3D, got shape {ref_cube.shape}")

    # test_3D.py stores flat predictions as (IL*XL, H), while the raw reference cube is (H, IL, XL)
    n_s, n_il, n_xl = tuple(ref_cube.shape)
    expected_traces = n_il * n_xl

    if arr2d.shape == (expected_traces, n_s):
        cube = arr2d.reshape(n_il, n_xl, n_s)
    elif arr2d.shape == (n_s, expected_traces):
        cube = arr2d.T.reshape(n_il, n_xl, n_s)
    else:
        raise ValueError(
            f"cannot restore 2D array of shape {arr2d.shape} using reference cube {ref_npy} "
            f"with flat target shape (IL*XL, H)=({expected_traces}, {n_s}); expected "
            f"({expected_traces}, {n_s}) or ({n_s}, {expected_traces})"
        )
    return cube.astype(np.float32, copy=False), (n_il, n_xl, n_s)


def load_as_cube(
    npy_path: Path,
    mapping: Tuple[int, int, int],
    ref_npy: Optional[Path],
) -> Tuple[np.ndarray, Tuple[int, int, int], Optional[Path], Tuple[int, ...]]:
    arr = np.load(npy_path).astype(np.float32, copy=False)
    original_shape = tuple(arr.shape)

    if arr.ndim == 3:
        cube = np.transpose(arr, mapping).astype(np.float32, copy=False)
        return cube, tuple(cube.shape), None, original_shape

    if arr.ndim != 2:
        raise ValueError(f"expected a 2D or 3D npy array, got shape {arr.shape}")

    ref_used = ref_npy if ref_npy is not None else infer_reference_npy(npy_path)
    if ref_used is None:
        raise ValueError(
            "2D input requires a reference 3D cube to infer (IL, XL, S). "
            "Use --ref-npy to provide one."
        )

    cube, shape3d = restore_cube_from_2d(arr, ref_npy=ref_used)
    return cube, shape3d, ref_used, original_shape


def convert_npy_to_sgy(
    npy_path: Path,
    out_sgy: Path,
    mapping: Tuple[int, int, int],
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    dt_us: int,
    ref_npy: Optional[Path] = None,
) -> None:
    cube, (n_il, n_xl, n_s), ref_used, original_shape = load_as_cube(
        npy_path=npy_path,
        mapping=mapping,
        ref_npy=ref_npy,
    )
    il_axis, xl_axis, s_axis = mapping

    print("NPY original shape:", original_shape)
    print("Reordered/restored shape (IL,XL,S):", cube.shape)
    print(f"Confirmed counts: IL={n_il}, XL={n_xl}, Samples={n_s}")
    if ref_used is not None:
        print("2D input restored to 3D cube using reference:", ref_used)
    print()
    print("==== XY CHECK ====")
    print("(IL=1, XL=1)         ->", xy_of(1, 1, x0, y0, dx, dy))
    print(f"(IL=1, XL={n_xl})      ->", xy_of(1, n_xl, x0, y0, dx, dy))
    print(f"(IL={n_il}, XL=1)      ->", xy_of(n_il, 1, x0, y0, dx, dy))
    print(f"(IL={n_il}, XL={n_xl}) ->", xy_of(n_il, n_xl, x0, y0, dx, dy))
    print("==================")

    out_sgy.parent.mkdir(parents=True, exist_ok=True)
    spec = build_spec(n_il, n_xl, n_s)
    tf = segyio.TraceField

    with segyio.create(str(out_sgy), spec) as f:
        f.bin[segyio.BinField.Interval] = int(dt_us)
        f.bin[segyio.BinField.Samples] = int(n_s)
        f.bin[segyio.BinField.Format] = 5

        f.text[0] = segyio.tools.create_text_header(
            {
                1: "NPY -> SEG-Y export",
                2: f"INPUT={npy_path.name} SHAPE={original_shape}",
                3: f"MAPPING=(IL,XL,S)=({il_axis},{xl_axis},{s_axis})",
                4: f"OUTPUT SHAPE(IL,XL,S)={n_il}x{n_xl}x{n_s}",
                5: f"XY RULE: X=X0+(XL-1)*dx, Y=Y0+(IL-1)*dy",
                6: f"X0={x0} Y0={y0} dx={dx} dy={dy} dt(us)={dt_us}",
                7: "XY written to Source/Group/CDP fields; scalar=1; coord_unit=meters",
                8: f"REF={ref_used if ref_used is not None else 'none'}",
            }
        )

        tr = 0
        for ii, il in enumerate(spec.ilines):
            for jj, xl in enumerate(spec.xlines):
                x, y = xy_of(int(il), int(xl), x0, y0, dx, dy)
                f.trace[tr] = cube[ii, jj, :]
                h = f.header[tr]

                h[tf.INLINE_3D] = int(il)
                h[tf.CROSSLINE_3D] = int(xl)

                set_if_exists(h, tf, "TRACE_SAMPLE_COUNT", int(n_s))
                set_if_exists(h, tf, "TRACE_SAMPLE_INTERVAL", int(dt_us))
                set_if_exists(h, tf, "ScalarCoordinate", 1)
                set_if_exists(h, tf, "SourceGroupScalar", 1)
                set_if_exists(h, tf, "CoordinateUnits", 1)
                set_if_exists(h, tf, "SourceX", int(round(x)))
                set_if_exists(h, tf, "SourceY", int(round(y)))
                set_if_exists(h, tf, "GroupX", int(round(x)))
                set_if_exists(h, tf, "GroupY", int(round(y)))
                set_if_exists(h, tf, "CDP_X", int(round(x)))
                set_if_exists(h, tf, "CDP_Y", int(round(y)))

                f.header[tr] = h
                tr += 1
        f.flush()

    print("Wrote SEG-Y:", out_sgy)
    print("If Petrel does not pick XY automatically, map SourceX/SourceY or GroupX/GroupY during import.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a 2D/3D NPY cube to SEG-Y.")
    parser.add_argument("--npy", required=True, help="Input .npy path")
    parser.add_argument("--out", required=True, help="Output .sgy path")
    parser.add_argument(
        "--mapping",
        nargs="+",
        default=["2,1,0"],
        help="Axis mapping from input cube to (IL,XL,S), e.g. 2,1,0 or 2 1 0",
    )
    parser.add_argument("--x0", type=float, default=0.0, help="Origin X for (IL=1,XL=1)")
    parser.add_argument("--y0", type=float, default=0.0, help="Origin Y for (IL=1,XL=1)")
    parser.add_argument("--dx", type=float, default=25.0, help="Crossline spacing in meters")
    parser.add_argument("--dy", type=float, default=25.0, help="Inline spacing in meters")
    parser.add_argument("--dt-us", type=int, default=1000, help="Sample interval in microseconds")
    parser.add_argument(
        "--ref-npy",
        default=None,
        help="Optional reference 3D npy used to restore 2D arrays into (IL,XL,S)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    convert_npy_to_sgy(
        npy_path=Path(args.npy),
        out_sgy=Path(args.out),
        mapping=parse_mapping(args.mapping),
        x0=float(args.x0),
        y0=float(args.y0),
        dx=float(args.dx),
        dy=float(args.dy),
        dt_us=int(args.dt_us),
        ref_npy=Path(args.ref_npy) if args.ref_npy else None,
    )


if __name__ == "__main__":
    main()
