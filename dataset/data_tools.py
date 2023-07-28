#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2023-07-27 10:55:36
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"""
Pretending zipped files are mp4 files for uploading to and sharing from aliyun webdrive.

Algorithm:

========================== before encoding ===========================
 -------------------------------------------------
|              payload                |  zip/gz   |
|             (N Bytes)               |  metadata |
 -------------------------------------------------
This is recognized as a zip file.

========================== after encoding ============================
 --------------------------------------------------------------------
| mp4 standard  |  subsequent payload |  zip/gz   |  first 32 Bytes  |
| header (32B)  |     (N-32 Bytes)    |  metadata |   of payload     |
 --------------------------------------------------------------------
This is recognized as an mp4 file.

After encoding and decoding the md5 checksum will be verified to ensure integrity.
"""

import os, sys, hashlib, io
from tqdm import tqdm

raw_file_list = [
    "preprocessed.tar.gz",
    "raw-data.z01",
    "raw-data.z02",
    "raw-data.z03",
    "raw-data.z04",
    "raw-data.z05",
    "raw-data.zip",
]
encoded_file_list = [
    "preprocessed.mp4",
    "raw-data-1.mp4",
    "raw-data-2.mp4",
    "raw-data-3.mp4",
    "raw-data-4.mp4",
    "raw-data-5.mp4",
    "raw-data-6.mp4",
]
# raw_file_list = ["preprocessed.tar.gz"]
# encoded_file_list = ["preprocessed.mp4"]
md5_path = "drin-datasets.md5"
mp4_header = b"\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2avc1mp41"
header_len = len(mp4_header)


def md5(filename):
    hash_md5 = hashlib.md5()
    file_len = os.path.getsize(filename)
    with open(filename, "rb") as f:
        for chunk in tqdm(iter(lambda: f.read(4096), b""), total=file_len // 4096):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def mimic_header(filename):
    with open(filename, "br+") as f:
        original_header = f.read(header_len)
        f.seek(0)
        f.write(mp4_header)
        f.seek(0, io.SEEK_END)
        f.write(original_header)


def recover_header(filename):
    with open(filename, "br+") as f:
        f.seek(-header_len, io.SEEK_END)
        original_header = f.read(header_len)
        f.seek(0)
        f.write(original_header)
        f.seek(-header_len, io.SEEK_END)
        f.truncate()


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--encode":
        md5_list = []
        for filename, encoded_filename in zip(raw_file_list, encoded_file_list):
            print(f"encoding {filename} -> {encoded_filename}")
            md5_list.append(md5(filename))
            os.rename(filename, encoded_filename)
            mimic_header(encoded_filename)
        with open(md5_path, "w") as f:
            f.write('\n'.join(md5_list))
        return

    with open(md5_path, "r") as f:
        md5_list = f.read().splitlines()
    for filename, encoded_filename, md5_value in zip(raw_file_list, encoded_file_list, md5_list):
        print(f"decoding {encoded_filename} -> {filename}")
        os.rename(encoded_filename, filename)
        recover_header(filename)
        if md5_value != md5(filename):
            print(
                f"ERROR: MD5 checksum verification FAILED for file {filename} ({encoded_filename}),"
                " please try re-downloading the file."
            )
        else:
            print(f"Conversion done for file {filename}. MD5 checksum verification PASSED.")


if __name__ == "__main__":
    main()
