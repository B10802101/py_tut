#local global enclosing built-in
#local variable only accessed in a function
#glocal variable accessed global, careful of overwriting
#built in: packed function in files
#nest function priority: local->enclosing->global->builting->error 

# x = 'global x'
# def my_func():
#     global x
#     x = "local x"
# my_func()                   #x overwritten
# print(x)

# def outter():
#     x = 'outter x'
#     def inner():
#         #x = 'inner x'         #when no x defined in inner(), then seek outside function's x
#         print(x)
#     inner()
#     print(x)
# outter()