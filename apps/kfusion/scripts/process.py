depth = open("depth.txt", "w+")
count = 0
for line in open("data.csv"):
    if not (count % 2):
        depth.write(line)
    count += 1

depth.close()
