import dill

with open('data.pkl', 'rb') as file_:
    profdata = dill.load(file_)


print(profdata.data)
print(profdata.params)