import os
#os.chdir('/Users/WD/source/repos/B10802101/py_tut')    #change directory
print(os.getcwd())  #get current directory
#print(os.path.exists('/Users/WD/source/repos/B10802101/py_tut/file.py'))    #check if a file exist
#os.path.isdir & os.path.isfile
#print(os.path.splitext('/Users/WD/source/repos/B10802101/py_tut/file.py'))    #split file name and extension into a tuple
#os.listdir: list all the files in a directory into a tuple
#print(os.listdir())

for f in os.listdir():
    f_name, f_ext = os.path.splitext(f)
    save1, de1 = f_name.split('- -')
    # print(save1)
    new_f_name = "{} - {}".format(save1, save2)
    os.rename(f, new_f_name)
