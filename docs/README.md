# gravitIO 🍎

`gravitio` is a Python package that provides parsers, basics corrections for most principal gravimeter data formats, including **AQG** (Absolute Quantum Gravimeter) and **iGrav** (Superconducting Gravimeter).

## Installation

To install `gravitio`, you can use pip (assuming you have the package locally or it's published):

```bash
pip install git+https://github.com/lucamir/gravitIO
```

## Usage

`gravitio` provides a simple interface to extract data from gravimeter files into **Polars DataFrames** for fast processing.

### Basic Usage

```python
from gravitio.instruments.igrav import IGrav
from gravitio.instruments.aqg import AQG
from gravitio.instruments.cg6 import CG6
from gravitio.transforms.resampling import resample

# -------- iGrav Example --------
igrav = IGrav()
# Extract returns a Polars DataFrame
df_igrav = igrav.extract("path/to/file.tsf")

# Apply barometric correction
# Assuming column 1 is gravity and column 2 is pressure
df_igrav = igrav.apply_grav_conversion(df_igrav, grav_col_index=1, conversion_factor=-98.40)
df_igrav = igrav.apply_baro_correction(df_igrav, grav_col_index=1, baro_col_index=2)

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

## Supported Formats

- **AQG**: Instrument CSV files (raw and averaged). Note: some corrections cannot be applied to the averaged files.
- **iGrav**: Instrument .tsf files (Tsoft format).
- **Scintrex CG-6**: Instrument .dat files.
- **Scintrex CG-5**: Instrument .txt files.
- **Micro-g LaCoste FG5**: Instrument .set.txt project files.

## Development

Run tests:

```bash
pytest
```

Run linting:

```bash
ruff check .
```

Run type checking:

```bash
mypy .
```
