import numpy as np
import time
from math import sqrt

def euclidian_distance(x1, x2):
    distance_sqaured = 0
    for i in range(len(x1)):
        distance_sqaured += (x1[i]-x2[i])**2
    distance = sqrt(distance_sqaured)
    return distance

def leave_one_out_cross_validation(data):
    n = data.shape[0] 
    number_correctly_classified = 0 
    for i in range(n):
        object_to_classify = data[i, 1:]
        classfication_label = data[i,0]

        nearest_neighbor_distance = float('inf')
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
    accuracy = (number_correctly_classified/n)*100
    return accuracy

def forward_selection(data):
    start_time = time.time()
    n_features = data.shape[1] - 1  # Gets the number of features (excluding the class label in column 0)
    current_set_of_features = []  # Initialize the current empty set
    best_accuracy_so_far = 0  
    best_feature_set = None

    print("Running nearest neighbor with no features, using “leaving-one-out” evaluation, I get an accuracy of")
    best_accuracy_so_far = leave_one_out_cross_validation(data[:, [0]]) 
    print(f"{best_accuracy_so_far:.1f}%")

    print()
    print("Beginning search.")
    for i in range(n_features): 
        best_accuracy_for_current_level = 0  
        feature_to_add = None  

        # Evaluate each feature
        for feature in range(1, n_features + 1):  
            if feature not in current_set_of_features:  # Only evaluate features not yet selected

                temp_feature_set = [0] + current_set_of_features + [feature]  # Create a temporary set of features to evaluate

                selected_data = data[:, temp_feature_set]  # Select data corresponding to the selected features
                accuracy = leave_one_out_cross_validation(selected_data)  # Evaluate model with these features
                print(f"Using feature(s) {current_set_of_features + [feature]} accuracy is: {accuracy:.1f}%", end=" ")
                print()
                if accuracy > best_accuracy_for_current_level:  # Updating best_accuracy at the current level
                    best_accuracy_for_current_level = accuracy
                    feature_to_add = feature

        # Select the best feature 
        if feature_to_add is not None:
            current_set_of_features.append(feature_to_add)  # Add the best feature to the selected set
            print()
            print(f"Feature set {current_set_of_features} was best, {best_accuracy_for_current_level:.1f}%")
            print()

        if best_accuracy_for_current_level > best_accuracy_so_far:
            best_accuracy_so_far = best_accuracy_for_current_level
            best_feature_set = current_set_of_features.copy()

    print(f'Finished search!! The best feature subset is {best_feature_set} which has an accuracy of {best_accuracy_so_far:.1f}%')

    end_time = time.time()  
    print(f"Time taken for Forward Selection: {end_time - start_time:.2f} seconds\n")
    return current_set_of_features


def backward_elimination(data):
    start_time = time.time()
    n_features = data.shape[1] - 1
    current_set_of_features = list(range(1, n_features + 1)) #full feature set
    best_accuracy_so_far = 0
    best_feature_set = current_set_of_features.copy()
    
    print("Running nearest neighbor with all 6 features, using “leaving-one-out” evaluation, I get an accuracy of")
    best_accuracy_so_far = leave_one_out_cross_validation(data[:, [0] + current_set_of_features]) #calculates accuracy of full feature set
    print(f"{best_accuracy_so_far:.1f}%")
    print()
    print("Beginning Search.")

    for i in range(n_features):
        best_accuracy_for_current_level = 0
        feature_to_remove = None
        for feature in current_set_of_features:
            temp_feature_set = current_set_of_features.copy() # created temp set to remove each feature
            temp_feature_set.remove(feature)
            
            selected_data = data[:, [0] + temp_feature_set]  # Select data corresponding to the selected features
            accuracy = leave_one_out_cross_validation(selected_data)  # Evaluate model with these features
            print(f"Using feature(s) {temp_feature_set} accuracy is: {accuracy:.1f}%", end=" ")
            print()
            
            if accuracy > best_accuracy_for_current_level:  # Updating best_accuracy at the current level
                best_accuracy_for_current_level = accuracy
                feature_to_remove = feature  
        
        if feature_to_remove is not None:
            current_set_of_features.remove(feature_to_remove)  # Remove the feature
            print()
            print(f"Feature set {current_set_of_features} was best, {best_accuracy_for_current_level:.1f}%")
            print()

        if best_accuracy_for_current_level > best_accuracy_so_far: #Updates the best feature set to get the feature set with the highest accuracy
            best_accuracy_so_far = best_accuracy_for_current_level
            best_feature_set = current_set_of_features.copy()
    
    print(f'Finished search!! The best feature subset is {best_feature_set} which has an accuracy of {best_accuracy_so_far:.1f}%')

    end_time = time.time() 
    print(f"Time taken for Backward Elimination: {end_time - start_time:.2f} seconds\n")
    return best_feature_set
    
def get_user_choice():
    choice = input(
        "Enter '1' for Forward Selection or '2' for Backward Elimination: "
    ).strip()

    if choice == '1':
        return choice
    elif choice == '2':
        return choice
    else:
        print("Invalid input. Please enter '1' or '2'.")
        return get_user_choice()


if __name__ == "__main__":
    print("Welcome to Shreya Mohan's Feature Selection Algorithm")
    print()
    filename = input("Type in the name of the file to test : ")
    print()
    data = np.loadtxt(filename)
    
    num_rows = data.shape[0] 
    num_features = data.shape[1] - 1 
    print(f"This file has {num_features} features (not including the class attribute), with {num_rows} instances.")

    print()
    print("Which feature selection method would you like to use?")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    print()
    user_choice = get_user_choice()

    if user_choice == '1':
        selected_features = forward_selection(data)
        
    elif user_choice == '2':
        selected_features = backward_elimination(data)
 

    