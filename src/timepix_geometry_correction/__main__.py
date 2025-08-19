#!/usr/bin/env python

import argparse
import sys

from timepix_geometry_correction.timepix_geometry_correction import main as cli_main


def main():
    parser = argparse.ArgumentParser(description="Timepix 1&3 Chips Geometry Correction")
    parser.add_argument("config", nargs="?", help="Geometry config file")

    args = parser.parse_args()

    if not args.config:
        print("Error: Geometry config file is required")
        sys.exit(1)
    cli_main(args.config, args.log)


if __name__ == "__main__":
    sys.exit(main())
