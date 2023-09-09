#Create a file and rename it
import os
with open(input("Create a file: "),"w") as f1:
    print("Create successfully")
while True:
    try:
        with open (input("Open a file: "), "r") as f2:
            f_reader = f2.read()
        break
                                                                #only deal with the first meet error
        #a = 5 / 0
    except( FileNotFoundError, ZeroDivisionError) as er1:
        print(er1)
        print("Try again...")
    except ZeroDivisionError as er2:
        print(er2)
    else:
        print("opened successfully, continue...")
    finally:
        print("/=============================")

new_file_name = "example - {}".format(f1.name)
os.rename(f1.name, new_file_name)