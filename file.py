import os
#os.chdir('/Users/WD/source/repos/B10802101/py_tut')    #change directory
#print(os.getcwd())  #get current directory
#print(os.path.exists('/Users/WD/source/repos/B10802101/py_tut/file.py'))    #check if a file exist
#os.path.isdir & os.path.isfile
#print(os.path.splitext('/Users/WD/source/repos/B10802101/py_tut/file.py'))    #split file name and extension into a tuple
#os.listdir: list all the files in a directory into a tuple
#print(os.listdir())

os.chdir('/Users/WD/source/repos/B10802101/py_tut')
# print(os.listdir(os.getcwd()))
for f in os.listdir():
    if f == ".git" or f == ".vs":
        continue
    f_name, f_ext = os.path.splitext(f)
    print(f_name)
    # new_f_name1 = f_name.split(".")
    # print(new_f_name1)
    
    # new_f_name = "{}{}".format(new_f_name1, '.py')
    # os.rename(f, new_f_name)