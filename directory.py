dir_path = os.path.dirname(os.path.realpath(__file__))
print("dir_path = ", dir_path)
cwd = os.getcwd()
print("cwd = ", cwd)

print("Path at terminal when executing this file")
print(os.getcwd() + "\n")

print("This file path, relative to os.getcwd()")
print(__file__ + "\n")

print("This file full path (following symlinks)")
full_path = os.path.realpath(__file__)
print(full_path + "\n")

print("This file directory and name")
path, filename = os.path.split(full_path)
print(path + ' --> ' + filename + "\n")

print("This file directory only")
print(os.path.dirname(full_path))

# read file

home = os.path.expanduser("~")
dir  ="/ML_DATA/"
filename = "test_sample.json"
datafile = home + dir + filename
print("open file : ", datafile)

with open(datafile) as f:
    lines = f.readlines()

# write pandas.DataFrame
df.to_pickle(home + out_dir + filename + '_' + time.strftime("%Y-%m-%d_%H-%M-%S") + '.pkl')
# df = pd.read_pickle(file_name)

df.to_csv(home + out_dir + filename + '_' + time.strftime("%Y-%m-%d_%H-%M-%S") + '.csv', sep='\t')

# store = pd.HDFStore(filename_out + '.h5')
# store['filename'] = df  # save it
# store['df']  # load it