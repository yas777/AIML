import sys

f = open(sys.argv[1])
pr = float(sys.argv[2])
rw1 = -0.1
rw2 = 10
ns = 0
grid = []
stno = []
st = 0
eds = 0
len1 = 0
while(1):
    temp = f.readline()
    if (temp == ""): break
    else: temp = temp.split()
    li0 = []
    li1 = []
    for i in temp:
        li1.append(ns)
        if (int(i) == 2): st = ns
        elif(int(i) == 3):eds = ns
        if(int(i) != 1):ns = ns + 1
        li0.append(int(i))
    # print(li1)
    len1 = len(li1)
    grid.append(li0)
    stno.append(li1)

f.close()
def helper(i, j, a, o1, o2, r, pr):
    print("transition " + str(stno[i][j]) + " " + str(a) + " " + str(stno[i+o1][j+o2]) + " " + str(r) + " " + str(pr))
# print(stno)
# print(len(grid))
print("numStates " + str(ns))
print("numActions 4")
print("start " + str(st))
print("end " + str(eds))
for i in range(len(grid)):
    for j in range(len(grid[i])):
        cst = grid[i][j]
        # print(cst)
        if(cst == 0 or cst == 2):
            # print("-"*10)
            # print(str(i) + ", " + str(j))
            if (j < len1 - 1):es = grid[i][j+1]
            else : es = 1
            if (j> 0) :ws = grid[i][j-1]
            else : ws = 1
            if (i > 0): n = grid[i-1][j]
            else : n = 1
            if (i < len1 - 1):s = grid[i+1][j]
            else : s= 1
            c = 0
            if (es != 1):c = c + 1
            if (ws != 1):c = c + 1
            if (n != 1): c = c + 1
            if (s != 1): c = c + 1
            pr1 = (1-pr)/c
            pr2 = pr + pr1
            if(es == 0 or es==2 or es == 3):
                if(es != 3):            helper(i, j, 0, 0, 1, rw1, pr2)
                if(es == 3):            helper(i, j, 0, 0, 1, rw2, pr2)
                # if(pr1 != 0):
                if(ws == 0 or ws == 2): helper(i, j, 0, 0, -1, rw1, pr1)
                if(ws == 3):            helper(i, j, 0, 0, -1, rw2, pr1)
                if(n == 0 or n==2):     helper(i, j, 0, -1, 0, rw1, pr1)
                if(n==3):               helper(i, j, 0, -1, 0, rw2, pr1)
                if(s==0 or s==2):       helper(i, j, 0, 1, 0, rw1, pr1)
                if(s == 3):             helper(i, j, 0, 1, 0, rw2, pr1)
            
            if(ws == 0 or ws == 2 or ws == 3):
                if(ws != 3):            helper(i, j, 1, 0, -1, rw1, pr2)
                if(ws == 3):            helper(i, j, 1, 0, -1, rw2, pr2)
                # if(pr1 != 0):
                if(es == 0 or es == 2): helper(i, j, 1, 0, 1, rw1, pr1)
                if(es == 3):            helper(i, j, 1, 0, 1, rw2, pr1)
                if(n == 0 or n == 2):   helper(i, j, 1, -1, 0, rw1, pr1)
                if(n == 3):             helper(i, j, 1, -1, 0, rw2, pr1)
                if(s == 0 or s == 2):   helper(i, j, 1, 1, 0, rw1, pr1)
                if(s == 3):             helper(i, j, 1, 1, 0, rw2, pr1)
            
            if(n == 0 or n == 2 or n == 3):
                if(n != 3):             helper(i, j, 2, -1, 0, rw1, pr2)
                if(n == 3):             helper(i, j, 2, -1, 0, rw2, pr2)
                # if(pr1 != 0):
                if(es == 0 or es == 2): helper(i, j, 2, 0, 1, rw1, pr1)
                if(es == 3):            helper(i, j, 2, 0, 1, rw2, pr1)
                if(ws == 0 or ws == 2): helper(i, j, 2, 0, -1, rw1, pr1)
                if(ws == 3):            helper(i, j, 2, 0, -1, rw2, pr1)
                if(s == 0 or s == 2):   helper(i, j, 2, 1, 0, rw1, pr1)
                if(s == 3):             helper(i, j, 2, 1, 0, rw2, pr1)

            if(s == 0 or s == 2 or s == 3):
                if(s != 3):             helper(i, j, 3, 1, 0, rw1, pr2)
                if(s == 3):             helper(i, j, 3, 1, 0, rw2, pr2)
                # if(pr1 != 0):
                if(es == 0 or es == 2): helper(i, j, 3, 0, 1, rw1, pr1)
                if(es == 3):            helper(i, j, 3, 0, 1, rw2, pr1)
                if(ws == 0 or ws == 2): helper(i, j, 3, 0, -1, rw1, pr1)
                if(ws == 3):            helper(i, j, 3, 0, -1, rw2, pr1)
                if(n == 0 or n == 2):   helper(i, j, 3, -1, 0, rw1, pr1)
                if(n == 3):             helper(i, j, 3, -1, 0, rw2, pr1)
print("discount 0.9")




