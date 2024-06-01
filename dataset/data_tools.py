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

import os, hashlib, io
from tqdm import tqdm
from argparse import ArgumentParser

mp4_header = b"\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2avc1mp41"
header_len = len(mp4_header)


def md5(filename):
    hash_md5 = hashlib.md5()
    file_len = os.path.getsize(filename)
    with open(filename, "rb") as f:
        for chunk in tqdm(iter(lambda: f.read(4096), b""), total=(file_len + 4095) // 4096):
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


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dir", default=".")
    parser.add_argument("--encode", action="store_true")
    parser.add_argument("--raw_files", nargs="+", default=[])
    parser.add_argument("--encoded_files", nargs="+", default=[])
    return parser.parse_args()


def encode(args):
    md5_list = []
    for raw_name, encoded_name in zip(args.raw_files, args.encoded_files):
        raw_path = os.path.join(args.dir, raw_name)
        encoded_path = os.path.join(args.dir, encoded_name)
        if not os.path.exists(raw_path):
            print(f"{raw_path} not found, skipping...")
            continue
        print(f"generating md5 checksum for {raw_path}...")
        md5_list.append(md5(raw_path))
        os.rename(raw_path, encoded_path)
        mimic_header(encoded_path)
        print(f"encode {raw_path} -> {encoded_path} successfully.")
    with open(os.path.join(args.dir, "md5"), "w") as f:
        f.write("\n".join(md5_list))


def decode(args):
    with open(os.path.join(args.dir, "md5"), "r") as f:
        md5_list = f.read().splitlines()
    for raw_name, encoded_name, md5_value in zip(args.raw_files, args.encoded_files, md5_list):
        raw_path = os.path.join(args.dir, raw_name)
        encoded_path = os.path.join(args.dir, encoded_name)
        if not os.path.exists(encoded_path):
            print(f"{encoded_path} not found, skipping...")
            continue
        os.rename(encoded_path, raw_path)
        recover_header(raw_path)
        print(f"decode {encoded_path} -> {raw_path} successfully, verifying md5 checksum...")
        if md5_value != md5(raw_path):
            print(
                f"MD5 checksum verification FAILED for file {raw_path} ({encoded_path}),"
                " please try re-downloading the file."
            )
        else:
            print(f"Conversion done for file {raw_path}. MD5 checksum verification PASSED.")


def main():
    args = parse_args()
    if args.encode:
        encode(args)
    else:
        decode(args)


if __name__ == "__main__":
    main()
