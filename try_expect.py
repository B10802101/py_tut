#Create a file and rename it
import os
# with open(input("Create a file: "),"w") as f1:
#     print("Create successfully")
while True:
    try:
        # with open (input("Open a file: "), "r") as f2:
        #     f_reader = f2.read()
        # ans = 5 / 0
        print("Input a division:")
        division = int(input())
        a = 5 / division                               #only deal with the first meet error      
        break                                 
    except Exception as er1:
        print(er1)
        print("Try again...")
    except ZeroDivisionError as er2:
        print(er2)
    # else:
    #     print("opened successfully, continue...")
    finally:
        print("/=============================")
print('Ans = {}'.format(a))
# new_file_name = "example - {}".format(f1.name)
# os.rename(f1.name, new_file_name)