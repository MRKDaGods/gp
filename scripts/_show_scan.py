"""Display scan results from 10c v12.

NOTE: MOTA/HOTA were from broken COMBINED_SEQ (always -2.116/0.0).
Only IDF1 (mtmc_idf1 from global accumulator) is valid for ranking.
"""
import json

data = json.load(open("data/outputs/nb10c_v12_single/scan_results.json"))
results = data["results"]
results.sort(key=lambda x: x.get("IDF1", 0), reverse=True)

print(f"Total combos: {len(results)}")
print("NOTE: MOTA/HOTA from COMBINED_SEQ bug -- disregard\n")

header = f"{'sim':<6} {'algo':<5} {'app_w':<7} {'bridge':<8} {'st_w':<7}   {'IDF1':>7}"
print(header)
print("-" * 48)
for r in results:
    algo = "CD" if r.get("algorithm", "") == "community_detection" else "CC"
    print(
        f"{r['sim_thresh']:<6} {algo:<5} {r['appearance_w']:<7} "
        f"{r['bridge_prune']:<8} {r.get('st_w',0):<7.3f}   "
        f"{r['IDF1']:>7.4f}"
    )

print()
best = results[0]
algo_b = "CD" if best.get("algorithm", "") == "community_detection" else "CC"
print(f"BEST: sim={best['sim_thresh']} algo={algo_b} app={best['appearance_w']} "
      f"bridge={best['bridge_prune']} -> IDF1={best['IDF1']:.4f}")

idf1s = [r["IDF1"] for r in results]
print(f"IDF1 range: {min(idf1s):.4f} - {max(idf1s):.4f} (spread={max(idf1s)-min(idf1s):.4f})")
