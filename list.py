fruits = ['egg', 'Banana', 'Apple']
numbers = [1, 3, 2, 5, 9]
# numbers[-1] is the last element
# extend(), append(), insert(), pop(), reverse(), sort(): no return value
# print(fruits[0:1]) # print(fruits[:2]) # print(fruits[0:])
#min max sum
#index()

###########################sort vs sorted##############################

# fruits.sort(key = lambda x: len(x))
# print(fruits)
# newfruits = fruits.sort(key = lambda x: len(x))
# print(newfruits)

#######################################################################

# print(max(numbers))
# print(max(fruits))

#######################################################################

# for fruit in fruits:            #print elements
#     print(fruit)
# print(fruits)                   # print list

#######################################################################

# for num,fruit in enumerate(fruits, start=122):  #num is index, fruit is content
#     print(num, fruit)

############Tuple######################################################

##list is mutable tuple is unmutable    ##Only for access values and cant be changed
fruitstuple = ('Apple', 'Banana')
#fruits[0] = 'Grape'
print(fruitstuple[0])

#############Set#######################################################

##Sets  #intersection(Num), difference(Num), Union(Num)
# Num1 = {'1','2','3','4'}
# Num2 = {'1','4'}
# print(Num2.intersection(Num1))