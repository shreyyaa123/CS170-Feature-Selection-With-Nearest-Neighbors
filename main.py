import numpy as np
from math import sqrt, inf


def get_user_choice():
    choice = input(
        "Enter '1' for Forward Selection or '2' for Backward Elimination: "
    ).strip()

    if choice == '1':
        print("\nYou selected Forward Selection. Starting the process...\n")
        return choice
    elif choice == '2':
        print("\nYou selected Backward Elimination. Starting the process...\n")
        return choice
    else:
        print("Invalid input. Please enter '1' or '2'.")
        return get_user_choice()

def forward_selection(data):
    current_set_of_features = []  #Initalize empty feature set
    best_overall_accuracy = 0
    return current_set_of_features

def backward_elimination(data):
    current_set_of_features = list(range(data.shape[1] - 1))
    return current_set_of_features

def euclidian_distance(x1, x2):
    distance_sqaured = 0
    for i in range(len(x1)):
        distance_sqaured += (x1[i]-x2[i])**2
    distance = sqrt(distance_sqaured)
    return distance
    
def leave_one_out_cross_validation(data, current_set, features_to_add):
    n = data.shape[0] #data size
    number_correctly_classified = 0 
    for i in range(n):
        object_to_classify = data[i, 1:]
        classfication_label = data[i,0]

        nearest_neighbor_distance = float(inf)
        nearest_neighbor_label = None

        for k in range(n):
            if k == i:
                continue
            distance = euclidian_distance(data[k,1:], object_to_classify)
            if distance < nearest_neighbor_distance:
                nearest_neighbor_distance = distance
                nearest_neighbor_label = data[k, 0]
        if classfication_label == nearest_neighbor_label:
            number_correctly_classified += 1
    accuracy = number_correctly_classified/n
    return accuracy
                
if __name__ == "__main__":
    print("Welcome to my Feature Selection Algorithm!")
    print()
    filename = input("Type the name of the file to test: ")
    
    try:
        data = np.loadtxt(filename, delimiter=None)  
        print("File loaded successfully.\n")

        # Ensure class labels are in the first column and features are the rest
        labels = data[:, 0]             # First column: Class labels
        features = data[:, 1:]          # Remaining columns: Feature values

    except Exception as e:
        print(f"Error loading file: {e}")
        exit()

    print()
    print("Which feature selection method would you like to use?")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    print()

    user_choice = get_user_choice()

    if user_choice == '1':
        selected_features = feature_search_demo(data)
    elif user_choice == '2':
        selected_features = backward_elimination(data)

