#!/usr/bin/env python3
"""Compare independently measured, same-protocol TF/initial/corrected AUROCs."""
import argparse
import csv
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    for key in ("tensorflow","initial","corrected"):
        p.add_argument("--"+key,required=True,help="metrics.json from same test split and score protocol")
    p.add_argument("--output",required=True)
    a=p.parse_args()
    rows=[]
    for label,src in (("tensorflow",a.tensorflow),("initial_pytorch",a.initial),
                      ("corrected_pytorch",a.corrected)):
        path=Path(src)
        data=json.loads(path.read_text())
        for metric in ("image_auroc","pixel_auroc","samples"):
            if metric not in data:
                raise ValueError(f"{src} missing {metric}")
        rows.append({"model":label,"category":data.get("category"),
                     "samples":data["samples"],"image_auroc":data["image_auroc"],
                     "pixel_auroc":data["pixel_auroc"],"source":str(path.resolve()),
                     "checkpoint":data.get("checkpoint","unrecorded")})
    if len({x["category"] for x in rows})!=1 or len({x["samples"] for x in rows})!=1:
        raise ValueError("Different categories or sample counts; comparison invalid")
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/"comparison.json").write_text(json.dumps({"metrics":rows,
        "caution":"Caller must verify same dataset files, preprocessing, scoring and evaluation protocol."},indent=2))
    with (out/"comparison.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    for r in rows:
        print(r["model"],"image AUROC:",r["image_auroc"],"pixel AUROC:",r["pixel_auroc"])
    print("Saved:",out)
if __name__=="__main__":
    main()
