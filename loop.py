#break : directly break out of the loop
#continnue : directly skip to the next iteration

# for i in range(1,11): #not include 11
#     print(i)

x = 0
while x<5:
    print(x)
    if x == 2:
        break   #jump out of the loop immediately
    x += 1