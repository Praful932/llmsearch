import b
from a import A

def foo_monkey_patch():
    print("I'm in monkey patch - c.py")
b.foo = foo_monkey_patch


ob = A()
ob.getval()

