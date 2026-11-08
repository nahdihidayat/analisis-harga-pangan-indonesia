"""
cleaning.py - Pipeline data PIHPS Nasional (Modul 0)

Alur:
    data/raw/*.xlsx  ->  data/interim/harga_panjang_gabungan.csv
                     ->  data/processed/harga_pangan_bersih.csv
                     ->  reports/cleaning_log.csv  (tabel keputusan cleaning)
                     ->  reports/audit_summary.csv (ringkasan cakupan data)

Cara menjalankan:
    # A. folder mentah dan hasil di luar repo (disarankan untuk tahap cleaning awal)
    python cleaning.py --raw "C:/PIHPS/raw" --out "C:/PIHPS/hasil"

    # B. di dalam repo (memakai data/raw dan data/processed)
    python src/cleaning.py

Aturan nama file di data/raw/ (WAJIB, karena isi file tidak memuat jenis
pasar dan komoditas):
    <jenis-pasar>_<komoditas>_<periode>.xlsx
    contoh: pasar-tradisional_cabai-merah-keriting_2021.xlsx
"""
from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Pengaturan (ubah di sini, jangan di tengah kode)
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

PASAR = {
    "pasar-tradisional": "Pasar Tradisional",
    "pedagang-besar": "Pedagang Besar",
    "produsen": "Produsen",
}
KOMODITAS = {
    "cabai-merah-keriting": "Cabai Merah Keriting",
    "bawang-merah": "Bawang Merah Ukuran Sedang",
    "beras-medium-1": "Beras Kualitas Medium I",
}

MIN_OBS = 10             # series dengan data valid < 10 pengamatan dianggap placeholder
MAX_FFILL = 3            # isi data kosong maksimal 3 pengamatan berturut-turut
OUTLIER_LOG_JUMP = 0.20  # tandai perubahan > ~22% dalam satu pengamatan (log-selisih 0.20)

FILE_PATTERN = re.compile(
    r"^(?P<pasar>[a-z-]+)_(?P<komoditas>[a-z0-9-]+)_(?P<periode>\d{4}(?:-\d{4})?)$"
)
KEYS_SERIES = ["jenis_pasar", "komoditas", "provinsi"]
KEYS_UNIK = ["tanggal", "jenis_pasar", "komoditas", "provinsi"]


# ----------------------------------------------------------------------------
# Membaca satu file
# ----------------------------------------------------------------------------
def parse_filename(path: Path) -> dict:
    # toleransi nama ganda seperti 'x_2021.xlsx.xlsx' (Windows menyembunyikan ekstensi) dan akhiran '_xlsx'
    stem = re.sub(r"(\.xlsx)+$", "", path.stem, flags=re.IGNORECASE)
    stem = re.sub(r"_xlsx$", "", stem)
    m = FILE_PATTERN.match(stem)
    if not m:
        raise ValueError(
            f"Nama file tidak sesuai pola: {path.name}\n"
            "Pola yang benar: jenis-pasar_komoditas_periode.xlsx "
            "(contoh: produsen_bawang-merah_2023.xlsx). "
            "Cek juga apakah ada akhiran seperti ' (1)' dari Chrome."
        )
    meta = m.groupdict()
    if meta["pasar"] not in PASAR:
        raise ValueError(f"Jenis pasar tidak dikenal di {path.name}: {meta['pasar']}. Pilihan: {list(PASAR)}")
    if meta["komoditas"] not in KOMODITAS:
        raise ValueError(f"Komoditas tidak dikenal di {path.name}: {meta['komoditas']}. Pilihan: {list(KOMODITAS)}")
    return meta


def read_one(path: Path) -> pd.DataFrame:
    """Baca satu file Excel (format lebar) dan ubah ke format panjang."""
    meta = parse_filename(path)
    raw = pd.read_excel(path, dtype=str)
    raw.columns = [str(c).strip() for c in raw.columns]

    label_col = raw.columns[1]          # kolom 'Komoditas (Rp)' berisi nama provinsi
    date_cols = list(raw.columns[2:])   # kolom sisanya = tanggal

    long = raw.melt(
        id_vars=[label_col],
        value_vars=date_cols,
        var_name="tanggal_teks",
        value_name="harga_teks",
    ).rename(columns={label_col: "provinsi"})

    long["provinsi"] = long["provinsi"].str.strip()
    # format tanggal di file: '01/ 01/ 2021' -> hapus spasi -> dd/mm/yyyy
    long["tanggal"] = pd.to_datetime(
        long["tanggal_teks"].str.replace(" ", "", regex=False), format="%d/%m/%Y"
    )
    long["jenis_pasar"] = PASAR[meta["pasar"]]
    long["komoditas"] = KOMODITAS[meta["komoditas"]]
    long["sumber_file"] = path.name
    return long.drop(columns="tanggal_teks")


# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------
def add_log(log, masalah, jumlah, penanganan, alasan):
    log.append(
        {"masalah": masalah, "jumlah": int(jumlah), "penanganan": penanganan, "alasan": alasan}
    )


def convert_price(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Teks '58,150' -> angka 58150. Tanda '-' -> kosong (NaN)."""
    txt = df["harga_teks"].astype("string").str.strip()
    is_dash = txt.isna() | txt.isin(["-", "", "nan", "<NA>"])
    num = pd.to_numeric(txt.str.replace(",", "", regex=False), errors="coerce").astype("float64")
    unreadable = (~is_dash) & num.isna()

    add_log(log, "Angka tersimpan sebagai teks dengan pemisah koma ('58,150')",
            (~is_dash).sum(), "Hapus koma, ubah ke angka",
            "Supaya bisa dihitung; pemisah ribuan di sumber bukan desimal")
    add_log(log, "Sel bertanda strip (-) atau kosong", is_dash.sum(),
            "Diubah menjadi nilai kosong (NaN), lalu ditangani aturan imputasi",
            "Strip berarti tidak ada data pada hari itu, bukan harga nol")
    if unreadable.any():
        add_log(log, "Sel berisi teks yang bukan angka maupun strip", unreadable.sum(),
                "Diubah menjadi NaN", "Tidak bisa diinterpretasi; cek contoh di console")
        print("PERINGATAN, contoh teks tak terbaca:", df.loc[unreadable, "harga_teks"].unique()[:10])

    df = df.drop(columns="harga_teks")
    df["harga_asli"] = num.where(~is_dash & ~unreadable)
    return df


def drop_empty_provinces(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Buang 'provinsi' yang hampir tidak punya data (< MIN_OBS pengamatan valid).

    Contoh nyata: baris 'Dum' (placeholder) yang hanya berisi satu angka 10,000.
    """
    n_valid = df.groupby(KEYS_SERIES)["harga_asli"].transform("count")
    mask = n_valid < MIN_OBS
    if mask.any():
        detail = (df[mask].groupby("provinsi")["harga_asli"].agg(lambda s: int(s.notna().sum())).to_dict())
        add_log(log, f"Baris dengan < {MIN_OBS} pengamatan valid sepanjang periode (pengamatan valid per nama: {detail})",
                mask.sum(), "Dibuang dari data bersih",
                "Placeholder atau tidak punya data yang bisa dianalisis (mis. baris 'Dum' hanya berisi satu angka 10,000)")
        df = df[~mask].copy()
    return df


def drop_duplicates(df: pd.DataFrame, log: list) -> pd.DataFrame:
    dup_all = df[df.duplicated(KEYS_UNIK, keep=False)]
    if len(dup_all):
        conflict = (dup_all.groupby(KEYS_UNIK)["harga_asli"].nunique(dropna=False) > 1).sum()
        n_dup = df.duplicated(KEYS_UNIK, keep="first").sum()
        add_log(log, "Duplikat (tanggal, jenis pasar, komoditas, provinsi), biasanya periode unduhan tumpang tindih",
                n_dup, "Simpan kemunculan pertama", "Satu kombinasi harus punya satu harga per hari")
        if conflict:
            add_log(log, "Duplikat dengan nilai harga BERBEDA", conflict,
                    "Simpan kemunculan pertama; PERIKSA manual file sumbernya",
                    "Nilai bertentangan antar file perlu diselidiki")
        df = df.drop_duplicates(KEYS_UNIK, keep="first").copy()
    return df


def remove_nonpositive(df: pd.DataFrame, log: list) -> pd.DataFrame:
    bad = df["harga_asli"] <= 0
    if bad.any():
        add_log(log, "Harga nol atau negatif", bad.sum(), "Diubah menjadi NaN",
                "Harga pangan tidak mungkin <= 0, kemungkinan salah input")
        df.loc[bad, "harga_asli"] = np.nan
    return df


def impute(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Isi maksimal MAX_FFILL pengamatan berturut-turut dengan nilai sebelumnya.

    Dilakukan SETELAH semua file digabung, supaya 1 Januari 2022 bisa memakai
    nilai 31 Desember 2021. Kolom harga_diisi menandai nilai hasil pengisian.
    """
    df = df.sort_values(KEYS_SERIES + ["tanggal"]).reset_index(drop=True)
    df["harga"] = df.groupby(KEYS_SERIES)["harga_asli"].ffill(limit=MAX_FFILL)
    df["harga_diisi"] = df["harga"].notna() & df["harga_asli"].isna()
    still_missing = df["harga"].isna().sum()

    add_log(log, "Data kosong yang bisa diisi (maks. %d pengamatan berturut-turut)" % MAX_FFILL,
            df["harga_diisi"].sum(), "Diisi nilai pengamatan sebelumnya (forward fill), ditandai harga_diisi=True",
            "Kekosongan umumnya hari libur nasional; harga hari libur dianggap sama dengan hari kerja sebelumnya")
    add_log(log, "Data kosong yang TIDAK diisi (lebih panjang dari batas, atau di awal periode)", still_missing,
            "Dibiarkan kosong", "Mengisi celah panjang akan mengarang data; dampaknya diuji di analisis sensitivitas")
    return df


def flag_outliers(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Tandai lonjakan ekstrem. TIDAK dibuang: bisa salah input, bisa lonjakan nyata."""
    df["log_harga"] = np.log(df["harga"])
    df["log_perubahan"] = df.groupby(KEYS_SERIES)["log_harga"].diff()
    df["outlier"] = (df["log_perubahan"].abs() > OUTLIER_LOG_JUMP).fillna(False).astype(bool)
    df = df.drop(columns="log_harga")
    add_log(log, f"Perubahan harga ekstrem (|log-selisih| > {OUTLIER_LOG_JUMP}) dalam satu pengamatan",
            df["outlier"].sum(), "Hanya ditandai (kolom outlier), tidak dibuang",
            "Perlu diperiksa apakah salah input atau lonjakan nyata (mis. menjelang Ramadan)")
    return df


# ----------------------------------------------------------------------------
# Validasi dan ringkasan
# ----------------------------------------------------------------------------
def validate(df: pd.DataFrame, log: list) -> None:
    assert not df.duplicated(KEYS_UNIK).any(), "Masih ada duplikat tanggal per kombinasi"
    assert (df["harga"].dropna() > 0).all(), "Ada harga nol/negatif"

    g = df.groupby(KEYS_SERIES)["tanggal"].agg(["min", "max", "nunique"])
    g["diharapkan"] = [
        np.busday_count(a.date(), (b + pd.Timedelta(days=1)).date()) for a, b in zip(g["min"], g["max"])
    ]
    g["hilang"] = g["diharapkan"] - g["nunique"]
    add_log(log, "Hari kerja (Senin-Jumat) yang tidak muncul sebagai kolom tanggal di file",
            g["hilang"].clip(lower=0).sum(),
            "Dilaporkan saja, tidak dibuat artifisial",
            "Sabtu-Minggu memang tidak ada di sumber (pencacahan Senin-Jumat), jadi analisis memakai hari kerja dan rata-rata mingguan; hari libur muncul sebagai kolom bertanda strip")


def province_coverage(df: pd.DataFrame, log: list) -> None:
    kombinasi = df[["provinsi", "jenis_pasar", "komoditas"]].drop_duplicates()
    total = kombinasi[["jenis_pasar", "komoditas"]].drop_duplicates().shape[0]
    cakupan = kombinasi.groupby("provinsi").size()
    partial = cakupan[cakupan < total]
    if len(partial):
        add_log(log, f"Nama provinsi yang tidak muncul di semua kombinasi pasar-komoditas: {list(partial.index)}",
                len(partial), "Dibiarkan; periksa apakah beda penulisan nama atau memang tidak ada datanya",
                "Nama provinsi harus konsisten antar file sebelum analisis spillover")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    prov = df[df["provinsi"] != "Semua Provinsi"]
    return (
        prov.groupby(["jenis_pasar", "komoditas"])
        .agg(
            n_provinsi=("provinsi", "nunique"),
            tanggal_awal=("tanggal", "min"),
            tanggal_akhir=("tanggal", "max"),
            n_baris=("harga", "size"),
            pct_kosong_asli=("harga_asli", lambda s: round(s.isna().mean() * 100, 2)),
            pct_kosong_akhir=("harga", lambda s: round(s.isna().mean() * 100, 2)),
            n_outlier=("outlier", "sum"),
        )
        .reset_index()
    )


def longest_nan_run(s: pd.Series) -> int:
    best = run = 0
    for v in s.isna().to_numpy():
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def audit_per_series(df: pd.DataFrame) -> pd.DataFrame:
    """Per (jenis pasar, komoditas, provinsi): kelengkapan, celah terpanjang, seberapa sering harga berubah."""
    rows = []
    for (pasar, kom, prov), g in df.groupby(KEYS_SERIES):
        g = g.sort_values("tanggal")
        v = g["harga_asli"].dropna().to_numpy()
        rows.append({
            "jenis_pasar": pasar, "komoditas": kom, "provinsi": prov,
            "n_hari": len(g), "n_valid_asli": len(v),
            "pct_kosong_asli": round(g["harga_asli"].isna().mean() * 100, 2),
            "pct_kosong_akhir": round(g["harga"].isna().mean() * 100, 2),
            "celah_terpanjang_hari": longest_nan_run(g["harga_asli"]),
            "pct_hari_harga_berubah": round((v[1:] != v[:-1]).mean() * 100, 1) if len(v) > 1 else np.nan,
        })
    return pd.DataFrame(rows)


def coverage_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Provinsi x (jenis pasar | komoditas): 1 = ada data, 0 = tidak ada."""
    k = df[["provinsi", "jenis_pasar", "komoditas"]].drop_duplicates().copy()
    k["kolom"] = k["jenis_pasar"] + " | " + k["komoditas"]
    k["ada"] = 1
    return k.pivot_table(index="provinsi", columns="kolom", values="ada", fill_value=0).astype(int).reset_index()


# ----------------------------------------------------------------------------
# Program utama
# ----------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Cleaning data PIHPS Nasional")
    ap.add_argument("--raw", type=Path, default=RAW, help="folder berisi file .xlsx mentah")
    ap.add_argument("--out", type=Path, default=None,
                    help="folder hasil. Kosong = pakai data/processed dan reports di dalam repo")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = args.raw
    if args.out is None:
        out_dir, rep_dir, interim_dir = PROCESSED, REPORTS, INTERIM
    else:
        out_dir = rep_dir = interim_dir = args.out
    for folder in (out_dir, rep_dir, interim_dir):
        folder.mkdir(parents=True, exist_ok=True)
    log: list = []

    files = sorted(p for p in raw_dir.glob("*.xlsx") if not p.name.startswith("~$"))
    if not files:
        raise SystemExit(f"Tidak ada file .xlsx di {raw_dir}")
    print(f"Membaca {len(files)} file dari {raw_dir} ...")

    df = pd.concat([read_one(p) for p in files], ignore_index=True)
    add_log(log, "Format lebar (satu kolom per tanggal), satu file per jenis pasar-komoditas-periode",
            len(files), "Digabung dan diubah ke format panjang (tanggal, provinsi, komoditas, jenis_pasar, harga)",
            "Format panjang dibutuhkan untuk analisis deret waktu dan penggabungan antar tingkat pasar")

    blank = df["provinsi"].isna() | (df["provinsi"] == "")
    if blank.any():
        add_log(log, "Baris tanpa nama provinsi", blank.sum(), "Dibuang", "Tidak bisa diidentifikasi")
        df = df[~blank].copy()

    df = convert_price(df, log)
    df = drop_empty_provinces(df, log)
    df = drop_duplicates(df, log)
    df = remove_nonpositive(df, log)

    df["level"] = np.where(df["provinsi"] == "Semua Provinsi", "nasional", "provinsi")

    df = impute(df, log)
    df = flag_outliers(df, log)
    validate(df, log)
    province_coverage(df, log)

    kolom = ["tanggal", "provinsi", "level", "komoditas", "jenis_pasar",
             "harga", "harga_asli", "harga_diisi", "log_perubahan", "outlier"]
    out = df[kolom]

    # satu file bersih per jenis pasar
    slug = {v: k for k, v in PASAR.items()}
    hasil = []
    for pasar, g in out.groupby("jenis_pasar"):
        nama = f"{slug[pasar]}_bersih"
        g.to_csv(out_dir / f"{nama}.csv", index=False, date_format="%Y-%m-%d")
        try:
            g.to_parquet(out_dir / f"{nama}.parquet", index=False)
        except ImportError:
            pass
        hasil.append((nama, len(g)))

    pd.DataFrame(log).to_csv(rep_dir / "cleaning_log.csv", index=False)
    ringkas = summarize(df)
    ringkas.to_csv(rep_dir / "audit_summary.csv", index=False, date_format="%Y-%m-%d")
    audit_per_series(df).to_csv(rep_dir / "audit_provinsi.csv", index=False)
    coverage_matrix(df).to_csv(rep_dir / "cakupan_provinsi.csv", index=False)

    print("\n=== RINGKASAN CAKUPAN ===")
    print(ringkas.to_string(index=False))
    print("\n=== TABEL KEPUTUSAN CLEANING ===")
    print(pd.DataFrame(log)[["masalah", "jumlah"]].to_string(index=False))
    print("\nSelesai. File bersih per jenis pasar:")
    for nama, n in hasil:
        print(f"  {out_dir / (nama + '.csv')}  ({n:,} baris)")
    print(f"Laporan di: {rep_dir}  (cleaning_log, audit_summary, audit_provinsi, cakupan_provinsi)")


if __name__ == "__main__":
    main()
