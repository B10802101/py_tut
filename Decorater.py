def printfunc(func):
    def wrapper1():
        print('Now is running {}'.format(func.__name__))    #the modified content
        func()                                              #the origin function
    return wrapper1                                          #returns the wrapped function
def Bark(func):
    def wrapper2():                                          
        print('Now is running {}'.format(func.__name__))                        #the modified content
        func()                                              #the origin function
    return wrapper2

@Bark
@printfunc
def printfunc2():                                           #origin function
    print('Now is running printfunc3')

printfunc2()