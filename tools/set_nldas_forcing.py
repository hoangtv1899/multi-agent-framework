#!/usr/bin/env python3
"""
Repoint a built ELM case's DATM streams from Qian T62 (~1.9°, ~200 km) to
NLDAS-2 (0.125°, ~12 km) — a config-level forcing swap: same CLMNCEP datmode,
same CLM variable names (PRECTmms/FSDS/TBOT/WIND/QBOT/PSRF), only the stream
file paths + domain change. Rewrites the three
run/datm.streams.txt.CLM_QIAN.{Precip,Solar,TPQW} files in place, so each column
gets its NEAREST 12 km NLDAS-2 cell (i.e. forcing that varies with location /
elevation, unlike Qian's 2 giant cells).

    python3 tools/set_nldas_forcing.py <case_dir> [--yr-start 1995 --yr-end 1995]
"""
import argparse
from pathlib import Path

# NOTE (Compy): the NLDAS2 tree exists under /compyfs/inputdata but only
# Solar/ is populated (1980-01..1981-11); Precip/ and TPQWL/ are empty.
NLDAS = "/compyfs/inputdata/atm/datm7/atm_forcing.datm7.NLDAS2.0.125d.v1"
DOMAIN_DIR = "/compyfs/inputdata/share/domains/domain.clm"
DOMAIN = "domain.lnd.nldas2_0224x0464_c110415.nc"
# stream label -> (NLDAS2 subdir, filename token, field variableNames block)
STREAMS = {
    "Precip": ("Precip", "Prec", "PRECTmms precn"),
    "Solar":  ("Solar",  "Solr", "FSDS swdn"),
    "TPQW":   ("TPQWL",  "TPQWL",
               "TBOT     tbot\n        WIND     wind\n"
               "        QBOT     shum\n        PSRF     pbot"),
}

TEMPLATE = """<?xml version="1.0"?>
<file id="stream" version="1.0">
<dataSource>
   GENERIC
</dataSource>
<domainInfo>
  <variableNames>
     time    time
        xc      lon
        yc      lat
        area    area
        mask    mask
  </variableNames>
  <filePath>
     {domain_dir}
  </filePath>
  <fileNames>
     {domain}
  </fileNames>
</domainInfo>
<fieldInfo>
   <variableNames>
     {fields}
   </variableNames>
   <filePath>
     {field_path}
   </filePath>
   <fileNames>
{files}
   </fileNames>
</fieldInfo>
</file>
"""


def build(subdir, token, fields, y0, y1):
    files = "\n".join(f"    ctsmforc.NLDAS2.0.125d.v1.{token}.{y}-{m:02d}.nc"
                      for y in range(y0, y1 + 1) for m in range(1, 13))
    return TEMPLATE.format(domain_dir=DOMAIN_DIR, domain=DOMAIN, fields=fields,
                           field_path=f"{NLDAS}/{subdir}", files=files)


def apply_nldas(case_dir, y0=1995, y1=1995):
    run = Path(case_dir) / "run"
    for label, (subdir, token, fields) in STREAMS.items():
        (run / f"datm.streams.txt.CLM_QIAN.{label}").write_text(
            build(subdir, token, fields, y0, y1))
    return len(STREAMS)


def main():
    ap = argparse.ArgumentParser(description="Repoint a case's DATM to NLDAS-2")
    ap.add_argument("case_dir")
    ap.add_argument("--yr-start", type=int, default=1995)
    ap.add_argument("--yr-end", type=int, default=1995)
    args = ap.parse_args()
    n = apply_nldas(args.case_dir, args.yr_start, args.yr_end)
    print(f"repointed {n} streams -> NLDAS-2 (12 km) for {Path(args.case_dir).name}")


if __name__ == "__main__":
    main()
