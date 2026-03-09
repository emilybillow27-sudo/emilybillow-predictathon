import subprocess

predictathon_trials = [
    "AWY1_DVPWA_2024",
    "TCAP_2025_MANKS",
    "25_Big6_SVREC_SVREC",
    "OHRWW_2025_SPO",
    "CornellMaster_2025_McGowan",
    "24Crk_AY2-3",
    "2025_AYT_Aurora",
    "YT_Urb_25",
    "STP1_2025_MCG"
]

print("\n==============================")
print("   Running CV0 + CV00 Batch")
print("==============================\n")

for trial in predictathon_trials:
    print(f"--- CV0: {trial} ---")
    subprocess.run(["python", "src/cv0_predict.py", trial], check=True)

    print(f"--- CV00: {trial} ---")
    subprocess.run(["python", "src/cv00_predict.py", trial], check=True)

print("\n✓ All Predictathon CV0 and CV00 predictions complete.\n")