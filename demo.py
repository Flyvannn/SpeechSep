import math
with open("test.txt",encoding='utf-8') as f:
    lines = f.readlines()
nums = lines[0].split()
new_nums = []
max = 0
for num in nums:
    s = float(num)
    new_nums.append(s)
    if math.fabs(s) > max:
        max = math.fabs(s)
print(new_nums)
print(max)
last_nums = [i/2**15 for i in new_nums]
print(last_nums[-10:])
