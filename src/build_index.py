#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Broad genre mapping (keywords)
# -----------------------------
BROAD_GENRE_MAPPING: Dict[str, List[str]] = {
    "rock": [
        "classic rock", "soft rock", "progressive rock", "folk rock",
        "hard rock", "glam rock", "rockabilly", "post-hardcore",
        "alternative rock", "indie rock", "garage rock", "psychedelic rock",
        "punk", "grunge", "new wave", "madchester", "midwest emo",
        "neue deutsche welle", "rock",
    ],
    "pop": [
        "italian singer-songwriter",
        "synthpop", "electropop", "dance pop", "pop rock",
        "schlager", "doo-wop", "teen pop", "art pop",
        "pop",
    ],
    "hip-hop": [
        "memphis rap", "gangster rap", "southern hip hop",
        "hip hop", "trap", "drill", "grime", "g-funk",
        "miami bass", "rap",
    ],
    "electronic": [
        "happy hardcore", "bassline",
        "eurodance", "italo dance",
        "electronic", "edm", "house", "techno", "dubstep", "trance",
        "synthwave", "ambient", "new age", "idm", "downtempo",
        "trip hop", "big beat", "gabber", "uk garage",
    ],
    "jazz": [
        "adult standards", "easy listening",
        "smooth jazz", "jazz fusion", "bebop", "hard bop",
        "vocal jazz", "ragtime", "big band", "swing music", "boogie-woogie",
        "jazz",
    ],
    "classical": [
        "baroque", "chamber", "orchestra", "symphony", "neoclassical",
        "opera", "choral", "early music", "classical",
        "gregorian chant",
    ],
    "metal": [
        "deathcore", "black metal", "death metal", "doom metal",
        "industrial metal", "nu metal", "gothic metal", "rap metal",
        "metal",
    ],
    "r&b": [
        "motown", "neo soul", "new jack swing", "quiet storm",
        "r&b", "soul", "funk", "blues", "blues rock",
        "gospel",
    ],
    "country": [
        "traditional country", "honky tonk", "texas country", "red dirt",
        "americana", "country",
    ],
    "folk": [
        "singer-songwriter", "acoustic", "traditional music",
        "bluegrass", "newgrass", "celtic", "folk",
    ],
    "latin": [
        "corrido", "banda", "norteño", "norteño-sax", "vallenato",
        "bolero", "son cubano", "trova", "forró", "duranguense",
        "cuarteto", "sierreño", "electro corridos",
        "latin", "salsa", "bossa nova", "reggaeton", "latin pop",
        "bachata", "merengue", "cumbia", "sertanejo", "tejano",
    ],
    "disco": [
        "nu disco", "italo disco", "post-disco", "hi-nrg",
        "disco",
    ],
    "world": [
        "world", "chanson", "flamenco", "calypso", "gnawa",
        "dangdut", "variété française", "canzone napoletana",
        "entehno", "maluku", "agronejo", "dansband", "exotica",
    ],
    "holiday": ["christmas", "holiday"],
}


# -----------------------------
# Hard whitelist of training genres
# -----------------------------
TRAIN_GENRES = {
    "rock",
    "pop",
    "classical",
    "r&b",
    "electronic",
    "hip-hop",
    "jazz",
    "metal",
    "latin",
    "world",
    "disco",
    "country",
    "folk",
    "holiday",
}


def normalize_artist(s: str) -> str:
    s = str(s).lower().replace("_", " ")
    s = re.sub(r"[^\w\s]", "", s)
    return " ".join(s.split())


def normalize_genre(s: str) -> str:
    s = str(s).strip().lower()
    return " ".join(s.split())


def parse_genre_cell(cell) -> List[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    if isinstance(cell, list):
        return [normalize_genre(x) for x in cell if isinstance(x, str)]

    s = str(cell).strip()
    if not s:
        return []

    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [normalize_genre(x) for x in arr if isinstance(x, str)]
        except Exception:
            pass
        try:
            arr = ast.literal_eval(s)
            if isinstance(arr, list):
                return [normalize_genre(x) for x in arr if isinstance(x, str)]
        except Exception:
            pass

    if "|" in s:
        return [normalize_genre(x) for x in s.split("|") if x.strip()]

    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) > 1:
            return [normalize_genre(x) for x in parts if x]

    return [normalize_genre(s)]


def build_keyword_pairs(broad_map: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for broad, kws in broad_map.items():
        for kw in kws:
            pairs.append((normalize_genre(kw), broad))
    pairs.sort(key=lambda x: len(x[0]), reverse=True)  # longest keyword first
    return pairs


KEYWORD_PAIRS = build_keyword_pairs(BROAD_GENRE_MAPPING)


def map_to_broad(genres: List[str]) -> Tuple[str, Optional[str]]:
    if not genres:
        return "unknown", None

    for raw in genres:
        g = normalize_genre(raw)
        for kw, broad in KEYWORD_PAIRS:
            if kw and kw in g:
                return broad, raw

    # preserve info for auditing; will be collapsed by whitelist later
    return genres[0], genres[0]


def finalize_genre(broad: str) -> str:
    b = normalize_genre(broad)
    return b if b in TRAIN_GENRES else "other"


def iter_midi_files(data_root: Path) -> Iterable[Tuple[str, Path]]:
    for artist_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        artist = artist_dir.name
        for ext in ("*.mid", "*.midi", "*.MID", "*.MIDI"):
            for p in artist_dir.rglob(ext):
                if p.is_file():
                    yield artist, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Path containing artist folders")
    ap.add_argument("--artist_genres_csv", type=str, default=None, help="CSV with columns: artist, genre")
    ap.add_argument("--out", type=str, default="data/index.parquet")
    ap.add_argument("--min_files_per_artist", type=int, default=1)
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    rows = []
    for artist, midi_path in iter_midi_files(data_root):
        rows.append(
            {
                "artist": artist,
                "artist_norm": normalize_artist(artist),
                "midi_path": str(midi_path.resolve()),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No MIDI files found under {data_root}")

    counts = (
        df.groupby("artist_norm")["midi_path"]
        .count()
        .rename("n_files_for_artist")
        .reset_index()
    )
    df = df.merge(counts, on="artist_norm", how="left")
    df = df[df["n_files_for_artist"] >= args.min_files_per_artist].reset_index(drop=True)

    # Join genres from CSV if provided
    if args.artist_genres_csv:
        gdf = pd.read_csv(args.artist_genres_csv)
        if "artist" not in gdf.columns or "genre" not in gdf.columns:
            raise ValueError("artist_genres_csv must have columns: artist, genre")

        gdf["artist_norm"] = gdf["artist"].astype(str).map(normalize_artist)
        gdf["genre_list"] = gdf["genre"].apply(parse_genre_cell)

        mapped = gdf["genre_list"].apply(map_to_broad)
        gdf["genre_broad"] = mapped.apply(lambda x: x[0])
        gdf["genre_raw_primary"] = mapped.apply(lambda x: x[1] if x[1] is not None else "")
        gdf["genre_raw_joined"] = gdf["genre_list"].apply(lambda xs: "|".join(xs))
        gdf["genre_final"] = gdf["genre_broad"].apply(finalize_genre)

        gdf = gdf[
            ["artist_norm", "genre_broad", "genre_final", "genre_raw_primary", "genre_raw_joined"]
        ].drop_duplicates("artist_norm")

        df = df.merge(gdf, on="artist_norm", how="left")

    # Fill missing after merge
    df["genre_broad"] = df.get("genre_broad", pd.Series(index=df.index)).fillna("unknown")
    df["genre_final"] = df.get("genre_final", pd.Series(index=df.index)).fillna("other")
    df["genre_raw_primary"] = df.get("genre_raw_primary", pd.Series(index=df.index)).fillna("")
    df["genre_raw_joined"] = df.get("genre_raw_joined", pd.Series(index=df.index)).fillna("")

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"Saved index: {out_path}")
    print(f"Total MIDI files: {len(df)}")
    print(f"Unique artists: {df['artist_norm'].nunique()}")

    print("\nTop broad genres (genre_broad):")
    print(df["genre_broad"].value_counts().head(20).to_string())

    print("\nTop final genres (genre_final):")
    print(df["genre_final"].value_counts().to_string())


if __name__ == "__main__":
    main()
