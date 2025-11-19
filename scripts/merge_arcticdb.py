#!/usr/bin/env python3
import os
import argparse
import logging
from glob import glob
import arcticdb as adb
from datetime import datetime

def merge_arcticdb(source_root: str, dest_path: str):
    src_dirs = glob(os.path.join(source_root, "arctic-db-*"))
    if not src_dirs:
        print(f"❌ No source directories found at {source_root}/arctic-db-*")
        return

    # Sort to ensure consistent merge order (partition 0 first → benchmark safe)
    src_dirs = sorted(src_dirs)
    print(f"Starting merge of {len(src_dirs)} ArcticDB shards → {dest_path}")
    print(f"Shards: {', '.join([os.path.basename(p) for p in src_dirs])}")

    dest_uri = f"lmdb://{dest_path}"
    os.makedirs(dest_path, exist_ok=True)

    arctic = adb.Arctic(dest_uri)
    if not arctic.has_library("prices"):
        print("Creating new 'prices' library in merged DB")
        arctic.create_library("prices")
    dest_lib = arctic.get_library("prices")

    total_merged = 0
    all_symbols = set()
    duplicates_skipped = 0
    failed_symbols = []
    benchmark_found = False

    for src in src_dirs:
        shard_merged = 0
        try:
            src_uri = f"lmdb://{src}"
            src_arctic = adb.Arctic(src_uri)

            if not src_arctic.has_library("prices"):
                print(f"Skipping {src}: no 'prices' library")
                continue

            src_lib = src_arctic.get_library("prices")
            symbols = src_lib.list_symbols()
            print(f"Processing {src} → {len(symbols)} symbols")

            for symbol in symbols:
                try:
                    data_obj = src_lib.read(symbol)
                    df = data_obj.data

                    if not {"close", "datetime"}.issubset(df.columns):
                        print(f"  Skipping {symbol}: missing 'close' or 'datetime'")
                        failed_symbols.append((src, symbol, "Missing columns"))
                        continue

                    # CRITICAL: If symbol already exists, keep the one with MORE data points
                    # This handles cases where a ticker was refetched in a later partition
                    if symbol in all_symbols:
                        existing = dest_lib.read(symbol).data
                        if len(df) > len(existing):
                            dest_lib.delete(symbol)  # remove old version
                            print(f"  Upgrading {symbol}: {len(existing)} → {len(df)} rows")
                        else:
                            print(f"  Keeping existing {symbol} ({len(existing)} rows ≥ {len(df)})")
                            duplicates_skipped += 1
                            continue
                    else:
                        all_symbols.add(symbol)

                    # Write (or overwrite with better data)
                    dest_lib.write(symbol, df)
                    if symbol == "NIFTYMIDSML400.NS":
                        benchmark_found = True
                        print(f"  BENCHMARK {symbol} merged successfully ({len(df)} days)")

                    total_merged += 1
                    shard_merged += 1

                except Exception as e:
                    failed_symbols.append((src, symbol, str(e)))
                    print(f"  Failed {symbol}: {e}")

            print(f"Done with {src}: {shard_merged} symbols merged\n")

        except Exception as e:
            print(f"Failed to open shard {src}: {e}")
            failed_symbols.append((src, "N/A", str(e)))

    # Final Report
    print("="*60)
    print("INDIA RS MERGE COMPLETE")
    print("="*60)
    print(f"Total unique symbols merged : {len(all_symbols):,}")
    print(f"Total write operations      : {total_merged:,}")
    print(f"Duplicates skipped/upgraded : {duplicates_skipped}")
    print(f"Failed symbols              : {len(failed_symbols)}")
    print(f"Benchmark NIFTYMIDSML400.NS : {'FOUND & MERGED' if benchmark_found else 'MISSING!'}")
    print(f"Merged DB location          : {dest_path}")
    print("="*60)

    # Save failure log
    if failed_symbols:
        log_path = os.path.join(dest_path, "merge_failed_symbols.log")
        with open(log_path, "w") as f:
            f.write(f"# Merge failed symbols - {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}\n")
            f.write("Source,Symbol,Error\n")
            for src, sym, err in failed_symbols[:1000]:  # cap log size
                f.write(f"{src},{sym},{err}\n")
        print(f"Failed symbols logged to: {log_path}")

    if not benchmark_found:
        print("CRITICAL: NIFTYMIDSML400.NS was NOT found in any shard!")
        print("RS calculation will fail. Check your fetch partitions.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge partitioned ArcticDB shards (India RS System)")
    parser.add_argument("--source-root", required=True, help="Folder containing arctic-db-* shards")
    parser.add_argument("--dest-path", required=True, help="Destination for merged DB")

    args = parser.parse_args()
    merge_arcticdb(args.source_root, args.dest_path)
