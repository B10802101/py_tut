import csv
import os
#import requests
################################################################Commands
# os.chdir('/Users/WD/source/repos/B10802101/py_tut')                          #change directory
# print(os.getcwd())                                                           #get current directory
# print(os.path.exists('/Users/WD/source/repos/B10802101/py_tut/file.py'))     #check if a file exist
# os.path.isdir & os.path.isfile                                               #the path is a directory or a file
# print(os.path.splitext('/Users/WD/source/repos/B10802101/py_tut/file.py'))   #split file name and extension into a tuple
# print(os.listdir())                                                          #list all the files in a directory into a tuple
################################################################

# os.chdir('./csv.py')
# with open('example.csv', 'r') as file_name:                                                 #file_name is a variable of the file example.csv
#     csv_reader = csv.DictReader(file_name)                                                  #csv_reader is a register for reading file in Dict type
#     with open('example_copy', 'w') as copy_file:                                            #copy_file is a variable of the file example_copy
#         field_name = ["QuotaAmount", "StartDate", "OwnerName", "Username"]                  #accessing the class in origin csv file             
#         csv_writer = csv.DictWriter(copy_file, fieldnames = field_name, delimiter = '\t')   #csv_writer is a register for writing file in Dict type
#         for line in csv_reader:
#             del line["QuotaAmount"]                                                         #delete "QuotaAmount" class in csv_reader register
#             csv_writer.writerow(line)                                                       #take out contents in csv_writer and write rows into copy file
#             print(line)

################################################################################################

# with open('example.csv', 'w') as f:
#     file_name, file_ext = os.path.splitext(".\csv.py\example.csv")
#     new_file_name = file_name
#     os.rename('.\csv.py\example.csv', new_file_name)

################################################################################################

# #seize image URL and copy onto a new file
# r = requests.get(input())                          #https://images.nationalgeographic.org/image/upload/t_edhub_resource_key_image/v1652341068/EducationHub/photos/ocean-waves.jpg
# while True:
#     if r.ok == True:                          #if URL is exist
#         print('Downloading...')
#         break
#     else:
#         print('Invalid URL')
# with open('imagefile.png', 'wb') as f:        #wb is write in bytes
#     f.write(r.content)
# print('Download success!')

###########context manager######################################################################

# with open("example.csv",'r') as f:
#     pass
# print(f.closed)

# f = open('example.csv')
# print(f.name)
# print(f.closed)
# f.close()
# print(f.closed)

###########Read a file######################################################################

# os.chdir('./csv.py')
# print(os.getcwd())
# with open("read_a_file.txt",'r') as f:
#     f_content = f.read(10)
#     while len(f_content) > 0:
#         print(f_content, end = '')
#         f_content = f.read(10)
#         #f.seek(0)

###########Copy a image file######################################################################
# os.chdir('./csv.py')
# with open("image.jpg",'rb') as fr:
#     with open("image_copy.jpg",'wb') as fw:
#         chunksize = 300
#         fr_chunkcontent = fr.read(chunksize)
#         while len(fr_chunkcontent) > 0:
#             fw.write(fr_chunkcontent)                   #seize fr_chunkcontent to write into fw
#             fr_chunkcontent = fr.read(chunksize)
