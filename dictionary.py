student1 = {'name': 'Andy', 'age': 13, 'courses': ['math', 'english'], 'gender': 'female'}
student2 = {'name': 'Johnny', 'age': 13, 'courses': ['math', 'physics']}
student3 = {'name': 'Cindy', 'age': 11, 'courses': ['math', 'english']}
long_dict = {}
for i in range(1,101):
    key = 'key_{}'.format(i)        #key_1, key_2, key_3...
    value = i
    long_dict[key] = value
# print(long_dict)
##################Combine two dict############################################################

# student1.update(student2)                   # combine two dictionaries, if the key already exists, the value will be overwritten by student2
# for key, value in student1.items():
#     print(key, value)

##################Delete key and value in a dict##############################################

# storage1 = student1.pop('name', 'age')
# print(storage1)
# print(student1)

# student1['name'] = 'Andy'                   # create a new key and value
# print(student1)

##################print the key and value in a dict###########################################

# for a,b in student1.items():             # a = key, b = value
#     print(a,b)

##################sorting a dict##############################################################

stlist = [student1, student2, student3]
print(stlist)
stlist.sort(key = lambda x: len(x['name']), reverse = True)
print(stlist)
# newstlist = sorted(stlist, key = lambda x: len(x['name']), reverse = True)
# print(newstlist)

##################Commands####################################################################
# get(): input key and output value
# pop(): pop out the key and value to a variable or not
# keys(), values(): return all the keys and values in a dict


