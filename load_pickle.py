import pickle
from collections import defaultdict

with open("test_torch_cov.pkl", "rb") as f:
    C = defaultdict(int)
    while True:
        try:
            r = pickle.load(f)
            print(r)
            C[r[0]] += 1
        except EOFError:
            break
    for k, v in sorted(C.items(), key=lambda item: -item[1])[:50]:
        print(f"{k} {v}")
