import re

pattern = "(\d\d)-(\d\d\d)-(\d\d\d\d)"

pattern2 = r"\1\2\3"

number = input()

number2 = re.sub(pattern, pattern2, number)

print(number2)
