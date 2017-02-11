import os
import json
import parquet
import pandas as pd

home = os.path.expanduser("~")
dir  = "/media/sumeyer/SSD_2/ML_DATA/"
filename = "part-r-00000-67ebd6f0-bfb4-42e0-b516-d7aaa77cbcb8.snappy.parquet"
datafile = dir + filename

print("open file : ", datafile)


## assuming parquet file with two rows and three columns:
## foo bar baz
## 1   2   3
## 4   5   6

with open(datafile) as fo:
   # prints:
   # {"foo": 1, "bar": 2}
   # {"foo": 4, "bar": 5}
   for row in parquet.DictReader(fo):
       print(json.dumps(row))


with open(datafile) as fo:
   # prints:
   # 1,2
   # 4,5
   for row in parquet.reader(fo):
       print(",".join([str(r) for r in row]))

print(df.info())
print(df)