gt = open("../gt.txt", "w")

for line in open("../gt_ori.txt"):
    time = line.split(' ', 1)[0]
    sec = time[:-9]
    nsec = time[-9:]
    okvis_time = sec + '.' + nsec
    print(time)
    print(okvis_time)

    out = okvis_time + ' ' + line.split(' ', 1)[1]
    print(out)
    gt.write(out)

gt.close()