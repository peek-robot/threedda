def normalize(input, min, max):
    return 2 * (input - min) / (max - min) - 1

def denormalize(input, min, max):
    return (input + 1) * (max - min) / 2 + min