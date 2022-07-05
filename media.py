with open("teste.txt") as file:
    lines = file.readlines()
    soma = 0
    for line in lines:
        soma += float(line)
    print(soma/10);