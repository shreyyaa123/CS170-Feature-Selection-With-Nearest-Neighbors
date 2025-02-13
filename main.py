
import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

if __name__ == "__main__":
    print("Welcome to my Feature Selection Algorithm!")
    filename = input("Type the name of the file to test: ")
    print("Which feature selection method would you like to use?")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    
    while True:
        choice = input("Enter '1' for Forward Selection or '2' for Backward Elimination: ").strip()
        
        if choice == '1':
            print("\nYou selected Forward Selection. Starting the process...\n")
            break
        elif choice == '2':
            print("\nYou selected Backward Elimination. Starting the process...\n")
            break
        else:
            print("Invalid input. Please enter '1' or '2'.")