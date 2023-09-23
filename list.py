fruits = ['egg', 'Banana', 'Apple']
numbers = [1, 3, 2, 5, 9]

# print(numbers[-1]) 
# extend(), append(), insert(), pop(), reverse(), sort(): no return value
# fruits.extend(numbers)
# print(fruits)
# print(fruits[0:1]) # print(fruits[ :2]) # print(fruits[0: ])
# min max sum
# index()

###########################sort vs sorted##############################

# fruits.sort(key = lambda x: len(x))
# print(fruits)
# newfruits = sorted(fruits, key = lambda x: len(x), reverse = True)
# print(newfruits)

#######################################################################

# print(max(numbers))
# print(max(fruits))

#######################################################################

# for fruit in fruits:            #print elements
#     print(fruit)
# print(fruits)                   # print list

#######################################################################

# for num,fruit in enumerate(fruits):  #num is index, fruit is content
#     print('index :{}'.format(num), 'element is: {}'.format(fruit))

############Tuple######################################################

#list is mutable tuple is unmutable    ##Only for access values and cant be changed
# fruitstuple = ('Apple', 'Banana')
# print(fruitstuple[0])
# print(fruitstuple[1])

#############Set#######################################################

##Sets  #intersection(Num), difference(Num), Union(Num)
Num1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
Num2 = {1, 4}
diff = Num1.difference(Num2)
print(Num1)