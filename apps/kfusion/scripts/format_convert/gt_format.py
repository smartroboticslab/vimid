gt = open("gt.txt", "w")
gt.write("#timestamp [ns], x [m], y [m], z [m], q.x, q.y, q.z, q.w\n")
for line in open("camera_groundtruth.csv"):
    data_list = line.split(',')
    # data_list[7] = data_list[7][:-1]
    print(data_list)
    if data_list[0] == "timestamp":
        continue

    out = data_list[0] + ' ' + data_list[1] + ' ' + data_list[2] + ' ' \
        + data_list[3] + ' ' + data_list[4] + ' ' + data_list[5] + ' '\
        + data_list[6] + ' ' + data_list[7]
    print(out)
    gt.write(out)
gt.close()
