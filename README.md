# gravitIO 🍎

`gravitio` is a Python package that provides parsers and basic corrections for major gravimeter data formats, including **AQG** (Absolute Quantum Gravimeter) and **iGrav** (Superconducting Gravimeter).

## Installation

Install `gravitio` directly from GitHub:

```bash
pip install git+https://github.com/lucamir/gravitIO
```

## Usage

`gravitio` provides a simple interface to extract data from gravimeter files into **Polars DataFrames** for fast processing.

The package supports two usage patterns:

- a high-level interface through the **facade API**
- a modular interface through **extractor**, **correction**, and **export** components

### High-level usage via the facade API

```python
from gravitio.instruments.igrav import IGrav
from gravitio.instruments.aqg import AQG
from gravitio.instruments.cg6 import CG6
from gravitio.transforms.resampling.resample import resample

# -------- iGrav Example --------
igrav = IGrav()

# Extract returns a Polars DataFrame
df_igrav = igrav.extract("path/to/file.tsf")

# Apply corrections
# Assuming column 1 is gravity and column 2 is pressure
df_igrav = igrav.apply_grav_conversion(
    df_igrav,
    grav_col_index=1,
    conversion_factor=-98.40,
)
df_igrav = igrav.apply_baro_correction(
    df_igrav,
    grav_col_index=1,
    baro_col_index=2,
)

# Resample and export
df_igrav = resample(df_igrav, freq=10)
igrav.to_csv(df_igrav, "igrav_results.csv")


# -------- AQG Example --------
aqg = AQG()
df_aqg = aqg.extract("path/to/aqg_dir/")
df_aqg = resample(df_aqg, freq=10)
aqg.to_csv(df_aqg, "aqg_results.csv")


# -------- Scintrex CG-6 Example --------
cg6 = CG6()
df_cg6 = cg6.extract("path/to/cg6_dir/")
df_cg6 = cg6.mean_by_station(df_cg6)
cg6.to_excel(df_cg6, "cg6_results.xlsx")
```

### Advanced usage via extractor, correction, and export components

```python
# -------- iGrav Example --------
from gravitio.exporters import Exporter
from gravitio.corrections.igrav import IGravCorrection
from gravitio.extractors.igrav import IGravExtractor

igrav_extractor = IGravExtractor()
igrav_correction = IGravCorrection()
exporter = Exporter()

df_igrav = igrav_extractor.extract("path/to/file.tsf")
df_igrav = igrav_correction.apply_grav_conversion(
    df_igrav,
    grav_col_number=1,
    conversion_factor=-98.40,
)
exporter.to_csv(df_igrav, "path/results.csv")


# -------- AQG Example --------
from gravitio.exporters import Exporter
from gravitio.extractors.aqg import AQGExtractor

aqg_extractor = AQGExtractor()
exporter = Exporter()

df_aqg = aqg_extractor.extract("path/to/aqg_dir/")
exporter.to_miniseed(
    df_aqg,
    path="path_to_output_dir",
    network="network_string_code",
    station="station_string_code",
    channels=[
        {
            "column": "index_of_the_column",
            "channel": "HHZ",
            "conversion": "optional_value_multiplied_by_all_values_in_column",
        }
    ],
    location="location_string_code",
)
```

## Supported Formats

- **AQG**: instrument CSV files (raw and averaged). Note: some corrections cannot be applied to averaged files.
- **iGrav**: instrument `.tsf` files (Tsoft format).
- **Scintrex CG-6**: instrument `.dat` files.
- **Scintrex CG-5**: instrument `.txt` files.
- **Micro-g LaCoste FG5**: instrument `.set.txt` project files.

## Development

Run linting:

```bash
ruff check .
```

Run type checking:

```bash
mypy .
```
