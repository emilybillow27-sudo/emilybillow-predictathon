# src/vcf_utils.py

import pandas as pd
import numpy as np

def parse_vcf_to_dosage(vcf_path, out_path, chunk_size=50000):
    """
    Stream a VCF file and convert genotypes to dosage matrix.
    """

    print(f"Reading VCF header from: {vcf_path}")

    # Extract sample names
    with open(vcf_path, "r") as f:
        for line in f:
            if line.startswith("#CHROM"):
                header = line.strip().split("\t")
                samples = header[9:]
                break

    print(f"Detected {len(samples)} samples")

    # Write header
    with open(out_path, "w") as out:
        out.write("marker," + ",".join(samples) + "\n")

    print("Beginning streaming genotype conversion...")

    # Stream VCF body
    buffer = []
    row_count = 0

    with open(vcf_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue

            parts = line.strip().split("\t")
            marker_id = parts[2]
            genotypes = parts[9:]

            # Genotype → dosage
            dosage = []
            for gt in genotypes:
                call = gt.split(":")[0]
                if call in ["0/0", "0|0"]:
                    dosage.append("0")
                elif call in ["0/1", "1/0", "0|1", "1|0"]:
                    dosage.append("1")
                elif call in ["1/1", "1|1"]:
                    dosage.append("2")
                else:
                    dosage.append("")

            buffer.append(marker_id + "," + ",".join(dosage))
            row_count += 1

            # Chunked write
            if row_count % chunk_size == 0:
                with open(out_path, "a") as out:
                    out.write("\n".join(buffer) + "\n")
                buffer = []
                print(f"Processed {row_count} markers...")

    # Final flush
    if buffer:
        with open(out_path, "a") as out:
            out.write("\n".join(buffer) + "\n")

    print(f"Finished. Total markers processed: {row_count}")
    print(f"Dosage matrix written to {out_path}")