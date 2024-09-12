

tudo=''
with open("6-separando_ComBasenosNomes/features.csv", "r", encoding='utf8') as dados:
    lista = dados.read()

    for x in lista:
        if x == ';': tudo += f'{x} \n'
        else: tudo += x

print(tudo)