"""Download the UCI Adult (Census Income) dataset and merge it into data/adult-all.csv.

The UCI archive ships the data as two files (adult.data = 32,561 rows,
adult.test = 16,281 rows). The project uses all 48,842 records and makes its own
stratified 70/15/15 split, so the two files are concatenated here.
"""
import os
import urllib.request

BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def fetch(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(BASE + name, path)
    return path


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    rows = []
    for name in ["adult.data", "adult.test"]:
        with open(fetch(name)) as f:
            for line in f:
                line = line.strip()
                # skip blank lines and the "|1x3 Cross validator" header in adult.test
                if not line or line.startswith("|"):
                    continue
                # adult.test labels end with a dot (">50K."), remove it
                rows.append(line.rstrip("."))
    out = os.path.join(DATA_DIR, "adult-all.csv")
    with open(out, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"Wrote {len(rows):,} records to {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
