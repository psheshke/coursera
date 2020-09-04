import sys

n = int(sys.argv[1])

k = 1
for i in range(n):
    mstr = " "*(n - k) + "#"*k
    k += 1
    print(mstr)