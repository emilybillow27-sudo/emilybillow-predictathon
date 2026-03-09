#!/usr/bin/env python3

import os
import subprocess

TRIAL_VCFS = {
    "2025_AYT_Aurora": "GBS_SDSU_2025.vcf",
    "24Crk_AY2-3": "GBS_UMN_2020.vcf",
    "25_Big6_SVREC_SVREC": "ThermoFisher_AgriSeq_4K.fixed.vcf",
    "AWY1_DVPWA_2024": "GBS_WSU_2023.vcf",
    "CornellMaster_2025_McGowan": "GBS_Cornell_2024.vcf",
    "OHRWW_2025_SPO": "Wheat3K.fixed.vcf",
    "STP1_2025_MCG": "GBS_TAMU_MCG25.vcf",
    "TCAP_2025_MANKS": "Allegro_V2.vcf",
    "YT_Urb_25": "GBS_UIUC_2024.fixed.vcf",
}

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = f"{root}/results/samples"
    os.makedirs(outdir, exist_ok=True)

    for trial, vcfname in TRIAL_VCFS.items():
        vcf_path = f"{root}/data/predictathon/{trial}/genotypes/{vcfname}"
        outpath = f"{outdir}/{trial}.txt"

        # Extract sample names using bcftools
        cmd = ["bcftools", "query", "-l", vcf_path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[extract_new_samples] ERROR reading {vcf_path}")
            print(e.stderr)
            continue

        samples = result.stdout.strip().split("\n")
        samples = sorted(set(s.strip() for s in samples if s.strip()))

        with open(outpath, "w") as f:
            for s in samples:
                f.write(s + "\n")

        print(f"[extract_new_samples] {trial}: {len(samples)} samples → {outpath}")

if __name__ == "__main__":
    main()