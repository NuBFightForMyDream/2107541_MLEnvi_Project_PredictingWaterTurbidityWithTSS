# Data

| File | Description |
|---|---|
| `aquasat.csv.gz` | AquaSat matchup table (`sr_wq_rs_join.csv`), gzip-compressed (74 MB). pandas reads it directly with `pd.read_csv("data/aquasat.csv.gz")`. |
| `aquasat_column_dictionary.pdf` | Description of every column in the table. |

## Source

**AquaSat** — Ross, M. R. V., Topp, S. N., Appling, A. P., Yang, X., Kuhn, C., Butman, D., Simard, M., & Pavelsky, T. M. (2019).
AquaSat: A data set to enable remote sensing of water quality for inland waters. *Water Resources Research*, 55, 10012–10025.
https://doi.org/10.1029/2019WR024883

- Download: https://figshare.com/articles/dataset/AquaSat/8139383 (file `sr_wq_rs_join.csv`)
- License: CC BY 4.0

AquaSat joins in-situ water quality samples (US Water Quality Portal and LAGOS-NE) with Landsat 5/7/8 surface reflectance
collected over the same site within about ±1 day.

## Why a gzip copy?

The original CSV is 291 MB, above GitHub's 100 MB file limit. The gzip copy is stored as a normal Git file (no Git LFS).

| | Size | MD5 |
|---|---|---|
| `aquasat.csv.gz` (as stored) | 74,401,314 bytes | `1864010cde071ae3ed26e024d6f7859f` |
| decompressed CSV | 291,065,762 bytes | `a1b32db1860e4d5ac5d3befb06951dd7` |

To check the copy against the Figshare original: `gzip -dc data/aquasat.csv.gz | md5sum`.

## Size of the data used

| | Rows |
|---|---|
| Raw matchups | 603,432 |
| After the four cleaning filters (see the notebook, Section 3) | 128,603 |
