import random

with open("dataset256.txt", "w+") as file:
    for i in range(256):
        file.write(str(random.randint(-1000, 1000)) + " ")
