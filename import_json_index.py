import os
import json
import pandas as pd

home = os.path.expanduser("~")
dir  = "/media/sumeyer/SSD_2/ML_DATA/programmatic-dataprovider/data/de/training-datasets/v4/features.out.json/"
filename = "part-r-00000-93628840-fd71-4a78-8bdb-6cafdf2b2738.json"
filename = "test_sample.json"
datafile = dir + filename

print("open file : ", datafile)

df = pd.DataFrame(columns=['hhid','uid','cookieid']) #, index=['hhid','uid','cookieid'])
df = df.set_index(['hhid','uid','cookieid'])
i = 0
with open(datafile) as f:
    lines = f.readlines()
    for line in lines:
        i += 1
        d = json.loads(line)
        # print(i, d['hhid'],d['uid'],d['cookieid'],d['featurekey'],d['featurevalue'])
        # make each line a real name-value dictionary
        dict_nv = {'hhid':d['hhid'] , 'uid':d['uid'] , 'cookieid':d['cookieid'] , d['featurekey']:d['featurevalue']}
        # df_nv = pd.DataFrame.from_dict(dict_nv)
        df_nv = pd.DataFrame(columns=['hhid', 'uid', 'cookieid'])  # , index=['hhid','uid','cookieid'])
        df_nv = df_nv.set_index(['hhid', 'uid', 'cookieid'])
        df_nv = df_nv.append(dict_nv, ignore_index = True, verify_integrity = False)
        # print(i, dict_nv)
        print(df)
        print(df_nv)
        print("-------")
        df=pd.concat([df,df_nv], ignore_index=False, verify_integrity=False)

print(df.info())
print(df)