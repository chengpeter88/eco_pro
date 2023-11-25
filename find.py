def find_max(num_list):
    max = 0 
    for number in num_list:
        if number > max:
            max = number
    print(max)