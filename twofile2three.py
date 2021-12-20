import sys

pred = open(sys.argv[1]).readlines()
pred = [s.strip() for s in pred]
golds = open(sys.argv[2]).readlines()
golds = [s.strip() for s in golds]

with open(sys.argv[3] + "test.pred", "w") as fpred, open(sys.argv[3] + "test.key", "w") as fkey, open(
        sys.argv[3] + "test.gold", "w") as fgold:
    for s in pred:
        ss = s.split("\t")
        if len(ss) == 2:
            fkey.write(ss[0] + "\n")
            fpred.write(ss[1] + "\n")
    for s in golds:
        ss = s.split("\t")
        if len(ss) == 2:
            fgold.write(ss[1] + "\n")

