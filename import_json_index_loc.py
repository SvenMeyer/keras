import os
import time
import json
import pandas as pd
import numpy as np

home = os.path.expanduser("~")
dir  = "/media/sumeyer/SSD_2/ML_DATA/programmatic-dataprovider/data/de/training-datasets/v4/features.out.json/"
dir  = "/ML_DATA/programmatic-dataprovider/data/de/training-datasets/v4/features.out.json/"
filename = "part-r-00000-93628840-fd71-4a78-8bdb-6cafdf2b2738.json"
# filename = "test_sample.json"
datafile = home + dir + filename

print("open file : ", datafile)

X = np.zeros((4096,32768), dtype='float32')


df = pd.DataFrame(columns=['hhid','uid','cookieid']) #, index=['hhid','uid','cookieid'])
df = df.set_index(['hhid','uid','cookieid'])
i = 0

print("start model.fit ..."),
start = time.time()

with open(datafile) as f:
    lines = f.readlines()
    for line in lines:
        i += 1
        d = json.loads(line)
        # print(i, d['hhid'],d['uid'],d['cookieid'],d['featurekey'],d['featurevalue'])
        # make each line a real name-value dictionary
        dict_nv = {'hhid':d['hhid'] , 'uid':d['uid'] , 'cookieid':d['cookieid'] , d['featurekey']:d['featurevalue']}
        # print(dict_nv)
        # df.loc()

        # print(df.loc['792611', '01', '12b31fa586482f2a9ca83b7c26b2ba8b'])

print(df.info())
print(df)

time_fit = (time.time() - start)
print(i, "lines processed")
print("DONE in ", time_fit, "sec")

# DONE in  17.91254711151123 sec
# run 2
# 4456158 lines processed
# DONE in  17.209401607513428 sec