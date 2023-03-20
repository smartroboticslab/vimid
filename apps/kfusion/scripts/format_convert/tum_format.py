gt = open("../gt.txt", "w")
gt.write("#timestamp [ns], x [m], y [m], z [m], q.x, q.y, q.z, q.w")
for line in open("../vicon_data.csv"):
    data_list = line.split(',', 7)
    data_list[7] = data_list[7][:-1]
    print(data_list)

    sec = data_list[0][:-9]
    nsec = data_list[0][-9:]
    okvis_time = sec + '.' + nsec
    print(data_list[0])
    print(okvis_time)

    out = okvis_time + ' ' + data_list[1] + ' ' + data_list[2] + ' ' \
        + data_list[3] + ' ' + data_list[5] + ' ' + data_list[6] + ' '\
        + data_list[7] + ' ' + data_list[4] + '\n'
    print(out)
    gt.write(out)
gt.close()
