"""A small whole-domain ELM case on disk, for the input-set and run-gate tests.

Built on elm_gridded_fixtures (its grid, formulas and history writers):

  write_surface   ELM surface data: LONGXY/LATIXY (0-360), PCT_NAT_PFT,
                  PCT_NATVEG, PCT_CROP, PCT_LAKE, PCT_WETLAND, PCT_GLACIER,
                  PCT_URBAN, URBAN_REGION_ID, soil fractions, TOPO,
                  PFTDATA_MASK, LANDFRAC_PFT
  write_restart   a saved state laid out as ELM lays out that surface
                  (domain_inputs.elm_layout: natural vegetation with 1 column
                  and 17 PFTs in every active cell; three urban land units of
                  5 one-PFT columns where URBAN_REGION_ID is not 0; one land
                  unit each for lake, wetland and glacier where present), with
                  the surface's weights
  write_case      a case folder (CaseDocs/lnd_in, timing/) and run folder
                  (daily, monthly and 3-hourly history, next-year restarts,
                  job-tagged lnd/cpl logs)

Vegetated cells are every cell but (0, 0), which carries urban land, and
(3, 1), which carries a lake.
"""
import gzip
from pathlib import Path

import numpy as np

import elm_gridded_fixtures as fx

NPFT, NURB, NSOIL = 17, 3, 4
URBAN_CELL, LAKE_CELL = (0, 0), (3, 1)
URBAN_SHARES = (10.0, 6.0, 4.0)                    # tall building, high, medium density
URBAN_COLUMN_WEIGHTS = (0.4, 0.2, 0.2, 0.15, 0.05)  # roof, walls, roads: sum to 1

DAILY_FIELDS = ("DOYCODE", "CELLID", "GPP", "QVEGT", "QVEGE", "QSOIL", "H2OSNO",
                "FAREA_BURNED", "TSA", "SOILLIQ", "TLAI", "RAIN", "SNOW", "QRUNOFF",
                "TWS", "TOTECOSYSC", "TOTSOMC", "TOTVEGC", "LEAFC", "BTRAN")


def surface_fields(g, edit=None):
    """The surface's fields; edit(fields) may change them in place."""
    nlat, nlon = g["shape"]
    natveg = np.full((nlat, nlon), 100.0)
    lake = np.zeros((nlat, nlon))
    urban = np.zeros((NURB, nlat, nlon))
    if URBAN_CELL[0] < nlat and URBAN_CELL[1] < nlon:
        natveg[URBAN_CELL] = 100.0 - sum(URBAN_SHARES)
        urban[:, URBAN_CELL[0], URBAN_CELL[1]] = URBAN_SHARES
    if LAKE_CELL[0] < nlat and LAKE_CELL[1] < nlon:
        natveg[LAKE_CELL] = 90.0
        lake[LAKE_CELL] = 10.0
    j, i = np.meshgrid(np.arange(nlat), np.arange(nlon), indexing="ij")
    raw = np.stack([1.0 + ((k + 1) * (j + 2) + 3 * i) % 7 for k in range(NPFT)])
    pft = 100.0 * raw / raw.sum(axis=0)          # sums to 1 within 2.3e-16 after / 100
    topo = 500.0 + 300.0 * j + 50.0 * i
    f = {"PCT_NATVEG": natveg, "PCT_LAKE": lake, "PCT_URBAN": urban, "PCT_NAT_PFT": pft,
         "PCT_WETLAND": np.zeros((nlat, nlon)), "PCT_GLACIER": np.zeros((nlat, nlon)),
         "PCT_CROP": np.zeros((nlat, nlon)), "URBAN_REGION_ID": np.ones((nlat, nlon), "i4"),
         "TOPO": topo}
    if edit:
        edit(f)
    return f


def write_surface(path, g=None, with_mask=True, negative_soil=False, edit=None, drop=()):
    """edit(fields) may change the fields before they are written; drop names
    variables to leave out."""
    import netCDF4
    g = g or fx.grid()
    nlat, nlon = g["shape"]
    f = surface_fields(g, edit)
    lon, lat = np.meshgrid(g["lon"], g["lat"])
    with netCDF4.Dataset(path, "w") as d:
        d.createDimension("lsmlat", nlat)
        d.createDimension("lsmlon", nlon)
        d.createDimension("natpft", NPFT)
        d.createDimension("numurbl", NURB)
        d.createDimension("nlevsoi", NSOIL)
        two = ("lsmlat", "lsmlon")

        def put(name, kind, dims, data):
            if name not in drop:
                d.createVariable(name, kind, dims)[:] = data
        put("LONGXY", "f8", two, lon)
        put("LATIXY", "f8", two, lat)
        put("PCT_NAT_PFT", "f8", ("natpft",) + two, f["PCT_NAT_PFT"])
        put("PCT_URBAN", "f8", ("numurbl",) + two, f["PCT_URBAN"])
        for n in ("PCT_NATVEG", "PCT_LAKE", "PCT_WETLAND", "PCT_GLACIER", "PCT_CROP"):
            put(n, "f8", two, f[n])
        put("URBAN_REGION_ID", "i4", two, f["URBAN_REGION_ID"])
        for n, base in (("PCT_SAND", 40.0), ("PCT_CLAY", 20.0), ("ORGANIC", 5.0)):
            a = np.full((NSOIL, nlat, nlon), base)
            if negative_soil and n == "PCT_SAND":
                a[:, 2, 3] = -999.0
            put(n, "f8", ("nlevsoi",) + two, a)
        put("TOPO", "f4", two, f["TOPO"])
        if with_mask:
            put("PFTDATA_MASK", "i4", two, np.ones((nlat, nlon), "i4"))
            put("LANDFRAC_PFT", "f8", two, np.ones((nlat, nlon)))
    return Path(path)


def write_restart(path, g=None, y0=2015, scale=1.0, edit=None, extra_types=(),
                  surface_edit=None, active=None):
    """A saved state matching write_surface, laid out in ELM's order.

    scale multiplies the carbon and LAI state (2.0 imitates a state the run
    did not start from); surface_edit is the edit given to write_surface, so
    the layout and weights follow that surface; active (bool, the grid's
    shape) leaves out cells a domain masks; edit(tables) may change the tables
    before they are written; extra_types adds one land unit of each listed
    type to cell (1, 1)."""
    import netCDF4
    g = g or fx.grid()
    nlat, nlon = g["shape"]
    f = surface_fields(g, surface_edit)
    lon, lat = np.meshgrid(g["lon"], g["lat"])
    active = np.ones((nlat, nlon), bool) if active is None else np.asarray(active, bool)
    t = {k: [] for k in ("grid1d_lon", "grid1d_lat", "grid1d_ixy", "grid1d_jxy",
                         "land1d_ityplun", "land1d_gridcell_index", "land1d_wtxy",
                         "cols1d_ityplun", "cols1d_gridcell_index",
                         "pfts1d_ityplun", "pfts1d_gridcell_index", "pfts1d_itypveg",
                         "pfts1d_wtlnd", "totsomc", "totvegc", "leafc", "tlai")}

    def col(typ, gc, somc=0.0):
        t["cols1d_ityplun"].append(typ)
        t["cols1d_gridcell_index"].append(gc)
        t["totsomc"].append(somc)

    def pft(typ, gc, veg, w, vegc=0.0, leafc=0.0, lai=0.0):
        t["pfts1d_ityplun"].append(typ)
        t["pfts1d_gridcell_index"].append(gc)
        t["pfts1d_itypveg"].append(veg)
        t["pfts1d_wtlnd"].append(w)
        t["totvegc"].append(vegc)
        t["leafc"].append(leafc)
        t["tlai"].append(lai)

    gc = 0
    for j in range(nlat):
        for i in range(nlon):
            if not active[j, i]:
                continue
            gc += 1
            t["grid1d_lon"].append(lon[j, i])
            t["grid1d_lat"].append(lat[j, i])
            t["grid1d_ixy"].append(i + 1)
            t["grid1d_jxy"].append(j + 1)
            units = [(1, f["PCT_NATVEG"][j, i])]
            if f["URBAN_REGION_ID"][j, i] != 0:
                units += [(7 + k, f["PCT_URBAN"][k, j, i]) for k in range(NURB)]
            for typ, name in ((5, "PCT_LAKE"), (6, "PCT_WETLAND"), (3, "PCT_GLACIER")):
                if f[name][j, i] > 0:
                    units.append((typ, f[name][j, i]))
            if (j, i) == (1, 1):
                units += [(typ, 0.0) for typ in extra_types]
            for typ, pct in units:
                t["land1d_ityplun"].append(typ)
                t["land1d_gridcell_index"].append(gc)
                t["land1d_wtxy"].append(pct / 100.0)
                if typ == 1:
                    col(1, gc, scale * fx.value("TOTSOMC", y0, 1, j, i, y0, nlon))
                    for k in range(NPFT):
                        pft(1, gc, k, f["PCT_NAT_PFT"][k, j, i] / 100.0,
                            scale * fx.value("TOTVEGC", y0, 1, j, i, y0, nlon),
                            scale * fx.value("LEAFC", y0, 1, j, i, y0, nlon),
                            scale * fx.value("TLAI", y0, 1, j, i, y0, nlon))
                elif 7 <= typ <= 9:
                    for w in URBAN_COLUMN_WEIGHTS:
                        col(typ, gc)
                        pft(typ, gc, 0, w)
                else:
                    col(typ, gc)
                    pft(typ, gc, 0, 1.0)
    t = {k: np.asarray(v) for k, v in t.items()}
    if edit:
        edit(t)
    with netCDF4.Dataset(path, "w") as d:
        for dim, n in (("gridcell", len(t["grid1d_lon"])), ("landunit", len(t["land1d_wtxy"])),
                       ("column", len(t["cols1d_ityplun"])), ("pft", len(t["pfts1d_wtlnd"]))):
            d.createDimension(dim, n)
        for k, v in t.items():
            dim = {"grid1d": "gridcell", "land1d": "landunit", "cols1d": "column",
                   "pfts1d": "pft"}.get(k.split("_")[0])
            dim = dim or ("column" if k == "totsomc" else "pft")
            kind = "i4" if np.asarray(v).dtype.kind in "iu" else "f8"
            d.createVariable(k, kind, (dim,))[:] = v
    return Path(path)


LND_IN = """&elm_inparm
 dtime = 1800
 finidat = '{restart}'
 fsurdat = '{surface}'   ! the surface data
 hist_fincl1 = 'GPP','TSA','CELLID'
 hist_fincl2 = 'GPP','TLAI','QVEGT','QVEGE','QSOIL','RAIN','SNOW',
         'QRUNOFF','TWS','H2OSNO','TOTECOSYSC','TOTSOMC','TOTVEGC','LEAFC'
 hist_fincl3 = 'TSA','CELLID'
 hist_mfilt = 12,365,2920
 hist_nhtfrq = 0,-24,-3
 use_cn = .true.
/
&ndepdyn_nml
 stream_year_first_ndep = 2000
/
"""


def write_logs(run_dir, case_dir, case, job, lid="261014-112716", ok=True,
               lnd_lines=(), throughput=30.18, gz=True, start=20150101, end=20170101,
               days=730.0, e3sm_lines=()):
    """A job's lnd, cpl (and, given e3sm_lines, e3sm) logs and timing file, in
    CIME's layout. The coupler log names the model dates the job ran: start
    when it begins, end at a successful termination."""
    run_dir, case_dir = Path(run_dir), Path(case_dir)
    lnd = ["(elm_init) land initialisation", *lnd_lines, "(elm_run) done"]
    cpl = ["(seq_timemgr_clockInit) start_ymd      =   20150101",
           f" memory_write: model date =   {start:8d}       0 memory =    5660.84 MB "
           f"(highwater)       1072.82 MB (usage)",
           f" tStamp_write: model date =   {end if ok else start:8d}       0 wall clock = "
           f"2026-10-14 13:10:19 avg dt =     7.76 dt =    35.16"]
    if ok:
        cpl += ["(seq_mct_drv): ===============          SUCCESSFUL TERMINATION OF "
                "CPL7-e3sm ===============",
                f"(seq_mct_drv): ===============        at YMD,TOD =   {end:8d}       0"
                f"       ===============",
                f"(seq_mct_drv): ===============  # simulated days (this run) =      "
                f"{days:.3f}  ==============="]
    logs = [("lnd.log", lnd), ("cpl.log", cpl)]
    if e3sm_lines:
        logs.append(("e3sm.log", list(e3sm_lines)))
    for name, lines in logs:
        p = run_dir / (f"{name}.{job}.{lid}" + (".gz" if gz else ""))
        text = "\n".join(lines) + "\n"
        if gz:
            with gzip.open(p, "wt") as fh:
                fh.write(text)
        else:
            p.write_text(text)
    (case_dir / "timing").mkdir(parents=True, exist_ok=True)
    (case_dir / "timing" / f"e3sm_timing.{case}.{job}.{lid}").write_text(
        f"    Model Throughput:        {throughput:.2f}   simulated_years/day \n\n"
        "    Init Time   :     418.208 seconds \n"
        "    Run Time    :    2862.626 seconds        7.843 seconds/day \n"
        "    Estimated Ocn Init Run Time   :       0.000 seconds \n")


def write_case(root, years=(2015, 2016), case="synth", job="111", restarts=True):
    """case and run folders for `years`, logs for `job`. Returns their paths."""
    root = Path(root)
    case_dir, run_dir = root / case, root / "scratch" / case / "run"
    (case_dir / "CaseDocs").mkdir(parents=True)
    run_dir.mkdir(parents=True)
    g = fx.grid()
    surf = write_surface(root / "surfdata.nc", g)
    rest = write_restart(root / "finidat.nc", g)
    (case_dir / "CaseDocs" / "lnd_in").write_text(LND_IN.format(restart=rest, surface=surf))
    fx.write_daily(run_dir, list(years), case=case, variables=DAILY_FIELDS)
    fx.write_daily(run_dir, list(years), case=case, stream="h2", steps_per_day=8)
    fx.write_monthly(run_dir, list(years), case=case)
    if restarts:
        for y in years:
            (run_dir / f"{case}.elm.r.{y + 1}-01-01-00000.nc").write_bytes(b"restart")
    write_logs(run_dir, case_dir, case, job, start=years[0] * 10000 + 101,
               end=(years[-1] + 1) * 10000 + 101, days=365.0 * len(years))
    return {"case_dir": case_dir, "run_dir": run_dir, "surface": surf, "restart": rest,
            "grid": g, "case": case, "job": job}
