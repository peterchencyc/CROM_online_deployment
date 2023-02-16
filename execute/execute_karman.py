import os
import argparse


folderu = r"weights\Karman_Fulid\vel_u"
folderv = r"weights\Karman_Fulid\vel_v"

nameu = "epoch=5999-step=5999"
namev = "epoch=5999-step=5999"

datau = r"data\Karman_Fulid\vel_u\h5_f_0000000000.h5"
datav = r"data\Karman_Fulid\vel_v\h5_f_0000000000.h5"

resolution = 200
total_steps = 1000
save_name = "Karman"

umax = 1.
umin = -1.
vmax = 1.
vmin = -1.

run_format = 'python online/online_karman.py -folderu {} -folderv {} -nameu {} -namev {} -datau {} -datav {} -resolution {} -total_steps {} -save_name {} -umax {} -umin {} -vmax {} -vmin {}'

command = run_format.format(folderu, folderv, nameu, namev, datau, datav, resolution, total_steps, save_name, umax, umin, vmax, vmin)

os.system(command)

command = 'python common/draw_karman.py -folder output\Karman\curl_Karman -fps 30'