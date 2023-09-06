import os
os.chdir('/Users/WD/source/repos/B10802101/py_tut')    #change directory
print(os.getcwd())  #get current directory
print(os.path.exists('/Users/WD/source/repos/B10802101/py_tut/file.py'))    #check if a file exist
#os.path.isdir & os.path.isfile
print(os.path.splitext('/Users/WD/source/repos/B10802101/py_tut/file.py'))    #check if a file exist
