#people is the default value of name
# def hello(greeting, name = "people"):
#     print("Hi," "{}, {}".format(greeting, name))
# hello("123")

# #this function can receive values and keywords
# def student(*args, **kwargs):    #arguments in function is changable
#     print(args)
#     print(kwargs)
# student(1, 2, 3, age = 10, name = "John")

# * can unpack the list, ** can unpack the dictionary
def student(*args, **kwargs):
    print(args)
    print(kwargs)
student(*([1,2,3]),**( {"name" : "andy", "age" : 10}))

def addition (x, y):
    return x + y