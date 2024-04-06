d = [
    {
        'name' : 'abc',
        'school' : "duke law",
    },
    {
        'name' : "xyz",
        'school' : "dy patil",
    }
]

s = "her name is {name}, She studies at {school}"

print(s)

input_cols = ['name', 'school']

for i in d:
    print(s.format(**i))