def coefficient_rate(input_array1, input_array2):
    array1_and_array2_num = 0
    array1_num = 0
    array2_num = 0
    width, length, height = input_array1.shape
    for i in range(width):
        for j in range(length):
            for k in range(height):
                if input_array1[i, j, k] and input_array2[i, j, k]:
                    ++array1_and_array2_num
                if input_array1[i, j, k]:
                    ++array1_num
                if input_array2[i, j, k]:
                    ++array2_num
    return 2 * array1_and_array2_num / (array1_num + array2_num)
