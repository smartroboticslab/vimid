import os

path = "/home/ryf/slam/dataset/room_2/depth0/data/"
f_rgb = open("rgb.txt", "r")
f_depth = open("depth.txt", "r")

while True:
    line_rgb = f_rgb.readline()
    line_depth = f_depth.readline()
    if not line_rgb:
        break
    rgb = line_rgb.split(",", 1)[0] + ".png"
    depth = line_depth.split(",", 1)[0] + ".png"
    os.rename(path + depth, path + rgb)
    print("First: ", rgb)
    print("Second: ", depth)
