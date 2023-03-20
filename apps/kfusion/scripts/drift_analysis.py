import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


drift = pd.read_csv("drift.csv")
sns.boxplot(x="Frame", y="Position error [m]", hue="Method", data=drift)
plt.show()



# mf_1 = open("Mid-Fusion-1", "w")
# mf_1.write("#timestamp [ns], x [m], y [m], z [m], q.x, q.y, q.z, q.w")
# mf_2 = open("Mid-Fusion-2", "w")
# mf_2.write("#timestamp [ns], x [m], y [m], z [m], q.x, q.y, q.z, q.w")
# mf_3 = open("Mid-Fusion-3", "w")
# mf_3.write("#timestamp [ns], x [m], y [m], z [m], q.x, q.y, q.z, q.w")
# mf_4 = open("Mid-Fusion-4", "w")
# mf_4.write("#timestamp [ns], x [m], y [m], z [m], q.x, q.y, q.z, q.w")
# mf_5 = open("Mid-Fusion-5", "w")
# mf_5.write("#timestamp [ns], x [m], y [m], z [m], q.x, q.y, q.z, q.w")
#
# vm_1 = open("VI-Mid-1", "w")
# vm_1.write("#timestamp [ns], x [m], y [m], z [m], q.x, q.y, q.z, q.w")
# vm_2 = open("VI-Mid-2", "w")
# vm_2.write("#timestamp [ns], x [m], y [m], z [m], q.x, q.y, q.z, q.w")
# vm_3 = open("VI-Mid-3", "w")
# vm_3.write("#timestamp [ns], x [m], y [m], z [m], q.x, q.y, q.z, q.w")
# vm_4 = open("VI-Mid-4", "w")
# vm_4.write("#timestamp [ns], x [m], y [m], z [m], q.x, q.y, q.z, q.w")
# vm_5 = open("VI-Mid-5", "w")
# vm_5.write("#timestamp [ns], x [m], y [m], z [m], q.x, q.y, q.z, q.w")
# count = 0
# for line in open("Mid-Fusion.txt"):
#     if count < 50:
#         mf_1.write(line)
#         mf_2.write(line)
#         mf_3.write(line)
#         mf_4.write(line)
#         mf_5.write(line)
#     elif 50 <= count < 100:
#         mf_2.write(line)
#         mf_3.write(line)
#         mf_4.write(line)
#         mf_5.write(line)
#     elif 100 <= count < 150:
#         mf_3.write(line)
#         mf_4.write(line)
#         mf_5.write(line)
#     elif 150 <= count < 200:
#         mf_4.write(line)
#         mf_5.write(line)
#     elif 200 <= count < 250:
#         mf_5.write(line)
#     count += 1
# count = 0
# for line in open("VI-Mid.txt"):
#     if count < 50:
#         vm_1.write(line)
#         vm_2.write(line)
#         vm_3.write(line)
#         vm_4.write(line)
#         vm_5.write(line)
#     elif 50 <= count < 100:
#         vm_2.write(line)
#         vm_3.write(line)
#         vm_4.write(line)
#         vm_5.write(line)
#     elif 100 <= count < 150:
#         vm_3.write(line)
#         vm_4.write(line)
#         vm_5.write(line)
#     elif 150 <= count < 200:
#         vm_4.write(line)
#         vm_5.write(line)
#     elif 200 <= count < 250:
#         vm_5.write(line)
#     count += 1
# mf_1.close()
# mf_2.close()
# mf_3.close()
# mf_4.close()
# mf_5.close()
# vm_1.close()
# vm_2.close()
# vm_3.close()
# vm_4.close()
# vm_5.close()
