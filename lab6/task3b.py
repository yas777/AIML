import sys
import random
import numpy as np
import math

random.seed(0)
f = open(sys.argv[1])
ns = 0
grid = []
stno = []
st = 0
eds = 0
len1 = 0
rw = 0
sti = 0
stj = 0
while(1):
    temp = f.readline()
    if (temp == ""): break
    else: temp = temp.split()
    li0 = []
    li1 = []
    c = 0
    for i in temp:
        li1.append(ns)
        if (int(i) == 2):
            st = ns
            sti = rw
            stj = c
        elif(int(i) == 3):eds = ns
        if(int(i) != 1):ns = ns + 1
        li0.append(int(i))
        c = c + 1
    len1 = len(li0)
    grid.append(li0)
    stno.append(li1)
    rw = rw + 1
f.close()
f = open(sys.argv[2])
# print(ns)
pol = [0.0 for k in range(ns)]
for k in range(ns):
    pol[k] = int(f.readline().split()[1])
# print(pol)
i = sti 
j = stj
cs = st
pr = float(sys.argv[3])
prs = ""
while(cs != eds):
    # print("Cs , i, j -- " + str(cs) + " " + str(i) + " " + str(j))
    temp = random.random()
    if (temp <= pr):
        # print("lss pr")
        if(pol[cs] == 0):
            prs = prs + "E "
            j = j + 1
        if(pol[cs] == 1):
            prs = prs + "W "
            j = j - 1
        if(pol[cs] == 2):
            prs = prs + "N "
            i = i - 1
        if(pol[cs] == 3):
            prs = prs + "S "
            i = i + 1
    else:
        # print("grr pr")
        es = 1
        ws = 1
        n = 1
        s = 1
        if (j < len1 - 1):es = grid[i][j+1]
        if (j > 0): ws = grid[i][j-1]
        if (i > 0): n = grid[i-1][j]
        if (i < len1 -1 ):s = grid[i+1][j]
        c = 0
        if (es != 1):c = c + 1
        if (ws != 1):c = c + 1
        if (n != 1): c = c + 1
        if (s != 1): c = c + 1
        
        div = (1-pr)/c

        cn=1
        cnt = math.ceil((1-temp)/div)
        if (es != 1):
            if(cn == cnt):
                prs+="E "
                j = j + 1
                cn = cn + 1
            else:
                cn = cn+1
            
        if (ws != 1):
            if(cn == cnt):
                prs+="W "
                j = j - 1
                cn = cn + 1
            else:
                cn = cn+1
            
        if (n != 1):
            if(cn == cnt):
                prs+="N "
                i = i - 1
                cn = cn + 1
            else:
                cn = cn+1
            
        if (s != 1):
            if(cn == cnt):
                prs+="S "
                i = i + 1
                cn = cn + 1
            else:
                cn = cn+1
    cs = stno[i][j]
print(prs)

    # es = 1
    # ws = 1
    # n = 1
    # s = 1
    # if (j < len1-1): es = grid[i][j+1]
    # if (j > 0): ws = grid[i][j-1]
    # if (i > 0): n = grid[i-1][j]
    # if (j < len1-1):s = grid[i+1][j]

    # c = 0
    # if (es != 1): c = c + 1
    # if (ws != 1): c = c + 1
    # if (n != 1): c = c + 1
    # if (s != 1): c = c + 1

    # pr1 = (1-pr)/c
    # pr2 = 0
    # pr3 = 0
    # pr4 = 0
    # if (pol[cs] == 0):
    #     if (ws != 1): pr2 = pr1
    #     if (n != 1): pr3 = pr1
    #     if (s != 1): pr4 = pr1
    #     temp = int(np.random.choice(4, 1, [pr+pr1, pr2, pr3, pr4]))
    # if (pol[cs] == 1):
    #     if (es != 1): pr2 = pr1
    #     if (n != 1): pr3 = pr1
    #     if (s != 1): pr4 = pr1
    #     temp = int(np.random.choice(4, 1, [pr2, pr + pr1, pr3, pr4]))
    # if (pol[cs] == 2):
    #     if (es != 1): pr2 = pr1
    #     if (ws != 1): pr3 = pr1
    #     if (s != 1): pr4 = pr1
    #     temp = int(np.random.choice(4, 1, [pr2, pr3, pr + pr1, pr4]))
    # if (pol[cs] == 3):
    #     if (es != 1): pr2 = pr1
    #     if (ws != 1): pr3 = pr1
    #     if (n != 1): pr4 = pr1
    #     temp = int(np.random.choice(4, 1, [pr2, pr3, pr4, pr + pr1]))
    # if (temp == 0):
    #     prs = prs + " E"
    #     print("")
    #     j = j + 1
    # if (temp == 1):
    #     prs = prs + " W"
    #     print("W")
    #     j = j - 1
    # if (temp == 2):
    #     prs = prs + " N"
    #     print("N")
    #     i = i - 1
    # if (temp == 3):
    #     prs = prs + "S"
    #     print("S")
    #     i = i + 1