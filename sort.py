class student():
    def __init__(self, name, age):                              #constructor
        self.name = name
        self.age = age
    def __repr__(self):
        return "({},{} yr)".format(self.name, self.age)         #print(st1) = print(st1.__repr__())
    
def stsort(st):
    return st.age

st1 = student('Bob', 13)
st2 = student('Andy', 12)
stli = [st1, st2]
new_stli = sorted(stli, key = lambda st: st.age, reverse=True)             #lambda: use to built a function easily
print(new_stli)
print(stli.sort(key = lambda st: st.age, reverse=True))