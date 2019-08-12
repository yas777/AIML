import sys
f = open(sys.argv[1])

ns = int(f.readline().split()[1])
na = int(f.readline().split()[1])
st = int(f.readline().split()[1])

sc = 0
pol = [0 for i in range(ns)]

eds = [False for i in range(ns)]
temp = f.readline().split()
if(int(temp[1]) == -1):
    eds = None
else:
    for i in range(len(temp[1:])):
        eds[int(temp[i+1])] = True
        sc = sc + 1
        pol[int(temp[i+1])] = -1 

# print(eds)
ts = [[[0.0 for i in range(ns)] for i in range(na)] for i in range(ns)]
rw = [[[0.0 for i in range(ns)] for i in range(na)] for i in range(ns)]
while(1):
    temp = f.readline()
    arr = temp.split()
    if(arr[0] != "discount"):
        st0 = int(arr[1])
        act = int(arr[2])
        st1 = int(arr[3])
        r = float(arr[4])
        t = float(arr[5])
        ts[st0][act][st1] = t
        rw[st0][act][st1] = r
    else:
        disc = float(arr[1])
        break
prevl = [0.0 for i in range(ns)]
newvl = [0.0 for i in range(ns)]

stop = [False for i in range(ns)]

# print(ts[6])
it = 0
while(sc!=ns):
    # print("-----")
    # print(it)
    # print(prevl[0])
    # print(prevl[1])
    # print(prevl[2])
    it = it+1
    for i in range(ns):
        temp = float("-inf")
        if(eds is None or eds[i] == False):
            for j in range(na):
                sig = 0.0
                flag = 0
                for k in range(ns):
                    if(ts[i][j][k] != 0):
                        flag = 1 
                    sig+=((ts[i][j][k])*(rw[i][j][k] + disc*(prevl[k])))
                if(flag == 1 and sig > temp):
                    temp = sig
                    pol[i] = j
            newvl[i] = temp     
            if(stop[i] == False and abs(newvl[i] - prevl[i]) <= 10**-8):
                sc+=1
                stop[i] = True
            prevl[i] = newvl[i]
for i in range(ns):
    print(str(newvl[i]) + " " + str(pol[i]))
print("iterations " + str(it))
