gt = open("OKVIS", "w")
gt.write("#timestamp [ns], x [m], y [m], z [m], q.x, q.y, q.z, q.w\n")
for line in open("okvis2-slam_trajectory.csv"):
    data_list = line.split(', ')
    # data_list[7] = data_list[7][:-1]
    print(data_list)
    if data_list[0] == "timestamp":
        continue

    sec = data_list[0][:-9]
    nsec = data_list[0][-9:]
    okvis_time = sec + '.' + nsec
    print(data_list[0])
    print(okvis_time)

    out = okvis_time + ' ' + data_list[1] + ' ' + data_list[2] + ' ' \
        + data_list[3] + ' ' + data_list[4] + ' ' + data_list[5] + ' '\
        + data_list[6] + ' ' + data_list[7] + '\n'
    print(out)
    gt.write(out)
gt.close()
