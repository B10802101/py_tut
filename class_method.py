class Car:
    door = 4                            #Attributes
    def __init__(self, seat, color):
        self.seat = seat
        self.color = color

    def thecolor(self):                    #Instance method
        print("the color of the car is {}'.format(self.color)")
    
    @classmethod
    def van(cls, vanseat, vancolor):
        return cls(vanseat, vancolor)
    
    @staticmethod
    def thisis():
        print('this is a car')
################################################################
# sportcar = Car(4, "black")    #Object, instance
# normalcar = Car(4, "Red")   #Object

# Car.thisis()
# sportcar.thisis()


# van1 = Car.van(7, "white")
# print(van1.seat)
# print(van1.color)


