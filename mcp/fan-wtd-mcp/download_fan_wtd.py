#!/usr/bin/env python3
"""
Helper to provision / verify the Fan 2013 water-table-depth dataset for the
fan_wtd MCP server.

The actual download URL for Fan 2013 tiles changes between archives, so this
script takes the URL explicitly (or from FAN_WTD_URL) rather than hard-coding a
possibly-stale link. See README.md for where to obtain the file.

Usage:
    # verify an already-downloaded / FAN_WTD_NC-pointed file opens correctly
    python download_fan_wtd.py --check

    # download from a URL you provide, into the default data dir
    python download_fan_wtd.py --url https://<host>/<fan2013_tile>.nc
    FAN_WTD_URL=https://<host>/<tile>.nc python download_fan_wtd.py
"""
import argparse
import os
import sys
import urllib.request
from pathlib import Path

DEFAULT_DEST = (Path(__file__).resolve().parents[2]
                / "data" / "fan_wtd" / "fan2013_wtd.nc")


def _target_path():
    return Path(os.path.expanduser(os.environ.get("FAN_WTD_NC", str(DEFAULT_DEST))))


def check():
    path = _target_path()
    if not path.exists():
        print(f"NOT FOUND: {path}\n  Set FAN_WTD_NC or download with --url. See README.md.")
        return 1
    try:
        import xarray as xr
        ds = xr.open_dataset(path)
        print(f"OK: {path}")
        print(f"  variables: {list(ds.data_vars)}")
        print(f"  coords:    {list(ds.coords)}")
        print(f"  dims:      {dict(ds.sizes)}")
        return 0
    except Exception as e:
        print(f"FOUND but failed to open: {path}\n  {e}")
        return 2


def download(url):
    dest = _target_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading\n  {url}\n-> {dest}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        print(f"Download failed: {e}")
        return 1
    print("Download complete. Verifying...")
    return check()


def main():
    ap = argparse.ArgumentParser(description="Provision/verify the Fan 2013 WTD dataset")
    ap.add_argument("--url", default=os.environ.get("FAN_WTD_URL", ""),
                    help="URL of a Fan 2013 NetCDF tile (or set FAN_WTD_URL)")
    ap.add_argument("--check", action="store_true",
                    help="Only verify the existing/FAN_WTD_NC-pointed file")
    args = ap.parse_args()

    if args.check or not args.url:
        sys.exit(check())
    sys.exit(download(args.url))


if __name__ == "__main__":
    main()
