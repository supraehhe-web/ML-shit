"""Print the learning-curve table from artifacts/logs/learning_curve/*.json."""
import json
from pathlib import Path

rows = []
for f in sorted(Path("artifacts/logs/learning_curve").glob("*.json")):
    j = json.loads(f.read_text())
    e = j["one_step_eps_mae"]
    cu = {r["t"]: r for r in j["condition_usage"]}
    rows.append((f.stem, e["200"], e["1000"], e["1500"], e["1900"], cu[1500]["zeroed_over_kept_other"],
                 j["partial_t0_600"]["gain_db"], j["pure_noise_2000"]["psnr"], j["pure_noise_2000"]["floor"]["psnr"],
                 j["pure_noise_2000"]["gain_db"]))
print(f"{'name':16s} {'mae200':>7s} {'mae1000':>8s} {'mae1500':>8s} {'mae1900':>8s} {'z/k1500':>8s} {'part+dB':>8s} {'pure':>7s} {'floor':>6s} {'gain':>6s}")
for r in rows:
    print(f"{r[0]:16s} " + " ".join(f"{v:8.4f}" if i < 4 else f"{v:8.2f}" for i, v in enumerate(r[1:])))
