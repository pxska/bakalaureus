with open("data.txt", "r", encoding="UTF-8") as f:
    txt = f.readlines()

time = []
for x in txt:
    if "Total seconds required for training" in x:
        time.append(float(x.split(":")[1].strip("\n").lstrip()))

print(sum(time))