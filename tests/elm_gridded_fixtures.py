"""Small ELM history files that copy the real layout, for the gridded tests.

Every value is a formula of (variable, label, cell), so a test can compute the
right answer without going through the reader it is testing. The layout is the
one measured on the Naches control run (2026-10-14):

  daily     file Y0:   one 30-minute start record [-1/48, 0], then 1 Jan-30 Dec
            file Y:    31 Dec of Y-1, then 1 Jan-30 Dec of Y
            last file: 31 Dec of the last year, alone
  sub-daily the same, in 3-hour steps (2920 records per file)
  monthly   file named <Y>-02-01-00000 with Jan-Dec of Y; January of the first
            year starts one 30-minute step early

Time is 'days since <Y0>-01-01', noleap; `time` is the END stamp, as ELM
writes it. Longitudes are 0-360 east; fill is 1e36.
"""
import math
from pathlib import Path

import numpy as np

FILL = np.float32(1e36)
DT = 1.0 / 48.0                      # one 30-minute model step, in days
MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
NLEV = 3


def grid(nlat=4, nlon=5):
    lat = 46.40417 + 0.0083333 * np.arange(nlat)
    lon = 238.40416 + 0.0083333 * np.arange(nlon)
    j, i = np.meshgrid(np.arange(nlat), np.arange(nlon), indexing="ij")
    area = 0.58 + 0.001 * (j * nlon + i)                 # km^2
    landfrac = np.ones((nlat, nlon))
    if nlon >= 2:
        landfrac[-1, -1] = 0.5
        landfrac[-1, -2] = 0.0                           # weighs nothing
    return {"lat": lat, "lon": lon, "area": area, "landfrac": landfrac,
            "shape": (nlat, nlon)}


def doy_to_md(doy):
    m = 1
    while doy > MONTH_DAYS[m - 1]:
        doy -= MONTH_DAYS[m - 1]
        m += 1
    return m, doy


# ── the formulas ─────────────────────────────────────────────────────────────
def value(var, year, doy, j, i, y0, nlon, step=0, layer=0):
    """The stored value at a labelled record (doy 1-365), before fill."""
    cell = j * nlon + i
    season = 1.0 + 0.5 * math.sin(2.0 * math.pi * doy / 365.0)
    if var == "DOYCODE":
        return (year - y0) * 1000 + doy
    if var == "CELLID":
        return cell
    if var == "STEPCODE":
        return ((year - y0) * 365 + doy - 1) * 8 + step + 1
    if var == "GPP":                                     # gC/m^2/s
        return 1e-5 * (1.0 + 0.01 * cell) * season * (1.0 + 0.002 * (year - y0))
    if var == "QVEGT":                                   # mm/s
        return 2e-5 * season * (1.0 + 0.003 * cell)
    if var == "QVEGE":
        return 5e-6 * (1.0 + 0.001 * doy)
    if var == "QSOIL":
        return 1e-5 * (1.0 + 0.002 * cell)
    if var == "H2OSNO":                                  # mm, a winter hump
        return max(0.0, 300.0 * math.cos(2.0 * math.pi * (doy - 45) / 365.0) - 50.0) \
            * (1.0 + 0.05 * j)
    if var == "FAREA_BURNED":                            # 'proportion', per second
        return 1e-10 * (1.0 + cell)
    if var == "TSA":
        return 270.0 + 20.0 * season + 0.1 * cell
    if var == "TBOT":                                    # K, the forcing's air
        return 268.0 + 22.0 * season + 0.2 * cell
    if var == "SOILLIQ":
        return 100.0 * (layer + 1) + cell + 0.01 * doy
    if var == "TLAI":                                    # m2/m2, a summer hump
        return 1.0 + 1.5 * max(0.0, math.sin(math.pi * (doy - 90) / 200.0)) + 0.01 * cell
    if var == "RAIN":                                    # mm/s
        return 3e-5 * (1.0 + 0.5 * math.cos(2.0 * math.pi * doy / 365.0))
    if var == "SNOW":
        return 1e-5 * max(0.0, math.cos(2.0 * math.pi * doy / 365.0))
    if var == "QRUNOFF":
        return 1.5e-5 * season
    if var == "TWS":                                     # mm
        return 5000.0 + 50.0 * math.sin(2.0 * math.pi * doy / 365.0) + cell
    if var == "TOTECOSYSC":                              # gC/m2
        return 5000.0 + 0.1 * doy + 10.0 * cell + 2.0 * (year - y0)
    if var == "TOTSOMC":
        return 3000.0 + 5.0 * cell + 0.01 * doy
    if var == "TOTVEGC":
        return 1800.0 + 3.0 * cell + 0.02 * doy
    if var == "LEAFC":
        return 60.0 + 0.5 * cell + 0.1 * doy
    if var == "BTRAN":                                   # 0-1, drier in summer
        return 0.85 - 0.2 * max(0.0, math.sin(math.pi * (doy - 120) / 180.0)) - 0.002 * cell
    raise KeyError(var)


UNITS = {"DOYCODE": "1", "CELLID": "1", "STEPCODE": "1", "GPP": "gC/m^2/s",
         "QVEGT": "mm/s", "QVEGE": "mm/s", "QSOIL": "mm/s", "H2OSNO": "mm",
         "FAREA_BURNED": "proportion", "TSA": "K", "TBOT": "K", "SOILLIQ": "kg/m2",
         "TLAI": "1", "RAIN": "mm/s", "SNOW": "mm/s", "QRUNOFF": "mm/s", "TWS": "mm",
         "TOTECOSYSC": "gC/m^2", "TOTSOMC": "gC/m^2", "TOTVEGC": "gC/m^2",
         "LEAFC": "gC/m^2", "BTRAN": "1"}


def is_fill(var, year, doy, j, i, y0):
    """Where the files carry 1e36 (or NaN): checked by the tests."""
    if var == "GPP":
        if (j, i) == (0, 0):
            return True                                  # a cell GPP never covers
        if (j, i) == (1, 1) and (year, doy) == (y0, 100):
            return True
        if (j, i) == (2, 2) and (year, doy) == (y0, 50):
            return True                                  # written as NaN
    if var == "QSOIL" and (j, i) == (1, 2) and (year, doy) == (y0, 10):
        return True
    return False


def expected(var, year, doy, j, i, y0, nlon, step=0, layer=0):
    if is_fill(var, year, doy, j, i, y0):
        return float("nan")
    return float(np.float32(value(var, year, doy, j, i, y0, nlon, step, layer)))


# ── writing ──────────────────────────────────────────────────────────────────
def _write(path, g, y0, tb, labels, variables, steps_per_day=1, nstep=True,
           cell_methods="time: mean"):
    """labels: per record (year, doy, step) or None for an unlabelled record.

    nstep is written as ELM writes it, the model step at the END of each
    record (30-minute steps), so the run's start record carries 0. Pass
    nstep=False for a file without it."""
    import netCDF4
    nlat, nlon = g["shape"]
    with netCDF4.Dataset(path, "w", format="NETCDF3_64BIT_OFFSET") as d:
        d.createDimension("time", None)
        d.createDimension("lat", nlat)
        d.createDimension("lon", nlon)
        d.createDimension("levgrnd", NLEV)
        d.createDimension("hist_interval", 2)
        t = d.createVariable("time", "f8", ("time",))
        t.long_name = "time"
        t.units = f"days since {y0:04d}-01-01 00:00:00"
        t.calendar = "noleap"
        t.bounds = "time_bounds"
        b = d.createVariable("time_bounds", "f8", ("time", "hist_interval"))
        mc = d.createVariable("mcdate", "i4", ("time",))
        for name, dims, data, extra in (
                ("lat", ("lat",), g["lat"], {"units": "degrees_north"}),
                ("lon", ("lon",), g["lon"], {"units": "degrees_east"}),
                ("area", ("lat", "lon"), g["area"], {"units": "km^2"}),
                ("landfrac", ("lat", "lon"), g["landfrac"], {}),
                ("topo", ("lat", "lon"), np.zeros(g["shape"]), {"units": "m"})):
            v = d.createVariable(name, "f4", dims, fill_value=FILL)
            v.missing_value = FILL
            for k, val in extra.items():
                setattr(v, k, val)
            v[:] = data
        n = len(tb)
        b[:] = np.asarray(tb, dtype=np.float64)
        t[:] = np.asarray(tb, dtype=np.float64)[:, 1]
        ends = []
        for _, e in tb:
            whole = int(math.floor(e + 1e-9))
            yy = y0 + whole // 365
            m, dd = doy_to_md(whole % 365 + 1)
            ends.append(yy * 10000 + m * 100 + dd)
        mc[:] = np.asarray(ends, dtype=np.int32)
        if nstep:
            ns = d.createVariable("nstep", "i4", ("time",))
            ns[:] = np.asarray([int(round(e / DT)) for _, e in tb], dtype=np.int32)
        for var in variables:
            dims = ("time", "levgrnd", "lat", "lon") if var == "SOILLIQ" \
                else ("time", "lat", "lon")
            v = d.createVariable(var, "f4", dims, fill_value=FILL)
            v.missing_value = FILL
            v.units = UNITS[var]
            v.cell_methods = cell_methods
            shape = (n, NLEV, nlat, nlon) if var == "SOILLIQ" else (n, nlat, nlon)
            a = np.zeros(shape, dtype=np.float32)
            for r, lab in enumerate(labels):
                for j in range(nlat):
                    for i in range(nlon):
                        if lab is None:
                            val = -999.0
                            if var == "SOILLIQ":
                                a[r, :, j, i] = val
                            else:
                                a[r, j, i] = val
                            continue
                        yy, doy, step = lab
                        if var == "SOILLIQ":
                            for k in range(NLEV):
                                a[r, k, j, i] = value(var, yy, doy, j, i, y0, nlon,
                                                      step, k)
                            continue
                        if is_fill(var, yy, doy, j, i, y0):
                            a[r, j, i] = np.nan if (j, i) == (2, 2) else FILL
                        else:
                            a[r, j, i] = value(var, yy, doy, j, i, y0, nlon, step)
            v[:] = a


DAILY_VARS = ("DOYCODE", "CELLID", "GPP", "QVEGT", "QVEGE", "QSOIL", "H2OSNO",
              "FAREA_BURNED", "TSA", "SOILLIQ")


def write_daily(dirpath, years, case="synth", stream="h1", g=None,
                start_step=True, last_file=True, variables=DAILY_VARS,
                steps_per_day=1):
    """Daily (steps_per_day=1) or sub-daily history in ELM's layout."""
    g = g or grid()
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    y0 = years[0]
    n = steps_per_day
    assert n in (1, 8), "STEPCODE's formula is written for 3-hourly steps"
    step = 1.0 / n
    files = []
    file_years = list(years) + ([years[-1] + 1] if last_file else [])
    for fy in file_years:
        off = 365.0 * (fy - y0)
        tb, labels = [], []
        if fy == y0:
            if start_step:
                tb.append((-DT, 0.0))
                labels.append(None)
        else:
            tb.append((off - step, off))                    # last step of fy-1
            labels.append((fy - 1, 365, n - 1))
        if fy in years:
            for k in range(365 * n - 1):                    # all but the last step
                tb.append((off + k * step, off + (k + 1) * step))
                labels.append((fy, k // n + 1, k % n))
        p = dirpath / f"{case}.elm.{stream}.{fy:04d}-01-01-00000.nc"
        vars_ = [v for v in variables if not (n > 1 and v == "DOYCODE")]
        if n > 1:
            vars_ = ["STEPCODE"] + [v for v in vars_ if v in ("CELLID", "TSA")]
        _write(p, g, y0, tb, labels, vars_, steps_per_day=n)
        files.append(p)
    return files


def write_monthly(dirpath, years, case="synth", stream="h0", g=None,
                  start_step=True, variables=("GPP", "TSA", "CELLID")):
    """Monthly history: one file per year, named for 1 February."""
    g = g or grid()
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    y0 = years[0]
    files = []
    for fy in years:
        off = 365.0 * (fy - y0)
        tb, labels = [], []
        cum = 0
        for m in range(1, 13):
            a = off + cum
            if fy == y0 and m == 1 and start_step:
                a -= DT
            cum += MONTH_DAYS[m - 1]
            tb.append((a, off + cum))
            # a monthly record carries its month's mid-month doy as a label
            labels.append((fy, sum(MONTH_DAYS[:m - 1]) + 15, 0))
        p = dirpath / f"{case}.elm.{stream}.{fy:04d}-02-01-00000.nc"
        _write(p, g, y0, tb, labels, list(variables) + ["DOYCODE"])
        files.append(p)
    return files


def write_single_cell(path, y0=2015):
    """A 1 x 1 history, the shape of a column run."""
    g1 = {"lat": np.array([46.5]), "lon": np.array([238.5]),
          "area": np.array([[0.58]]), "landfrac": np.array([[1.0]]), "shape": (1, 1)}
    tb = [(k, k + 1.0) for k in range(5)]
    labels = [(y0, k + 1, 0) for k in range(5)]
    _write(Path(path), g1, y0, tb, labels, ["TSA"])      # cell (0,0) has no GPP
    return Path(path)


# ── an unstructured (lndgrid) history ────────────────────────────────────────
# Six cells on ('lndgrid',), as ELM writes a non-rectangular domain. The cell
# index plays the role of i (with j = 0) in the formulas above.
NL = 6
LND_LAT = 46.40 + 0.010 * np.arange(NL)
LND_LON = 238.40 + 0.013 * np.arange(NL)
LND_AREA = 0.50 + 0.010 * np.arange(NL)


def write_lnd(path, y0, tb, labels, variables, landfrac=None):
    import netCDF4
    with netCDF4.Dataset(path, "w", format="NETCDF3_64BIT_OFFSET") as d:
        d.createDimension("time", None)
        d.createDimension("lndgrid", NL)
        d.createDimension("levgrnd", NLEV)
        d.createDimension("hist_interval", 2)
        t = d.createVariable("time", "f8", ("time",))
        t.units = f"days since {y0:04d}-01-01 00:00:00"
        t.calendar = "noleap"
        t.bounds = "time_bounds"
        b = d.createVariable("time_bounds", "f8", ("time", "hist_interval"))
        lf = np.ones(NL) if landfrac is None else np.asarray(landfrac, float)
        for name, data, units in (("lat", LND_LAT, "degrees_north"),
                                  ("lon", LND_LON, "degrees_east"),
                                  ("area", LND_AREA, "km^2"),
                                  ("landfrac", lf, None)):
            v = d.createVariable(name, "f4", ("lndgrid",), fill_value=FILL)
            v.missing_value = FILL
            if units:
                v.units = units
            v[:] = data
        b[:] = np.asarray(tb, float)
        t[:] = np.asarray(tb, float)[:, 1]
        ns = d.createVariable("nstep", "i4", ("time",))
        ns[:] = np.asarray([int(round(e / DT)) for _, e in tb], dtype=np.int32)
        for var in variables:
            lay = var == "SOILLIQ"
            dims = ("time", "levgrnd", "lndgrid") if lay else ("time", "lndgrid")
            v = d.createVariable(var, "f4", dims, fill_value=FILL)
            v.missing_value = FILL
            v.units = UNITS[var]
            a = np.zeros((len(tb), NLEV, NL) if lay else (len(tb), NL), np.float32)
            for r, lab in enumerate(labels):
                for c in range(NL):
                    if lab is None:
                        a[r, ..., c] = -999.0
                        continue
                    yy, doy, _ = lab
                    if lay:
                        for k in range(NLEV):
                            a[r, k, c] = value(var, yy, doy, 0, c, y0, NL, 0, k)
                    elif is_fill(var, yy, doy, 0, c, y0):
                        a[r, c] = FILL
                    else:
                        a[r, c] = value(var, yy, doy, 0, c, y0, NL)
            v[:] = a
    return Path(path)


def write_daily_lnd(dirpath, years, case="u", variables=("GPP", "TSA", "SOILLIQ")):
    """Daily lndgrid history in the same file layout as write_daily."""
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    y0 = years[0]
    files = []
    for fy in list(years) + [years[-1] + 1]:
        off = 365.0 * (fy - y0)
        tb, labels = [], []
        if fy == y0:
            tb.append((-DT, 0.0))
            labels.append(None)
        else:
            tb.append((off - 1.0, off))
            labels.append((fy - 1, 365, 0))
        if fy in years:
            for k in range(364):
                tb.append((off + k, off + k + 1))
                labels.append((fy, k + 1, 0))
        files.append(write_lnd(dirpath / f"{case}.elm.h1.{fy}-01-01-00000.nc", y0, tb,
                               labels, variables))
    return files
