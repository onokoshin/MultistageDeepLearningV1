'''
INSTRUCTION:
Before you run the script, please follow the steps below before starting.
    1. Please set the correct local csv file path below
        The two csv files needed here are product of process_data.py

    2. Please adjust threshold in constant below

'''

import pandas as pd
from azure.cognitiveservices.vision.customvision.training import training_api

# Provide appropriate information
project_id = "8d05ada4-02ee-42f6-9784-9c6aa3457f9d"
training_key = "2fb2e54d14fc4c4e8fb264338277c250"
trainer = training_api.TrainingApi(training_key)

# Set the local csv file location here
testing_car_df_path = r"/Users/Koshin/PycharmProjects\deep_learning\testing_car.csv"
prediction_result_df_path = r"/Users/Koshin/PycharmProjects\deep_learning\prediction_result.csv"

# Decide the threshold percentage - default 75%
CONST_THRESHOLD = 75

### In case you are analyzing custom model, please provide appropriate path to custom_model_result.csv
custom_model_result_df_location = r"/Users/Koshin/PycharmProjects\deep_learning\bumper_damage_result.csv"


def csv_to_dataFrame(local_csv_location):

    with open(local_csv_location) as file:

        #reads the csv file
        df = pd.read_csv(file)

    return df


def analyze_two_excel_sheets():

    testing_car_df = csv_to_dataFrame(testing_car_df_path)
    prediction_result_df = csv_to_dataFrame(prediction_result_df_path)

    classifier_list = list(testing_car_df.columns.values)[3:16]

    glass_damage = classifier_list[0]
    water_damage = classifier_list[1]
    bumper_damage = classifier_list[2]
    rolled_over = classifier_list[3]
    side_door_damage = classifier_list[4]
    slid_off_road = classifier_list[5]
    location = classifier_list[6]
    time = classifier_list[7]
    weather = classifier_list[8]
    road_condition = classifier_list[9]
    bumps_and_dings = classifier_list[10]
    engine_damage = classifier_list[11]
    severity = classifier_list[12]

    # debug lines
    # print(classifier_list)

    # print(testing_car_df[severity])

    # apply true if correct prediction selected, false if not selected
    for i in range(len(testing_car_df)):

        # glass damage feature true/false update
        change_df_values(testing_car_df, prediction_result_df, glass_damage, CONST_THRESHOLD, i)
        # water damage feature true/false update
        change_df_values(testing_car_df, prediction_result_df, water_damage, CONST_THRESHOLD, i)
        # bumper damage feature true/false update
        change_df_values(testing_car_df, prediction_result_df, bumper_damage, CONST_THRESHOLD, i)
        # rolled_over feature true/false update
        change_df_values(testing_car_df, prediction_result_df, rolled_over, CONST_THRESHOLD, i)
        # side_door_damage feature true/false update
        change_df_values(testing_car_df, prediction_result_df, side_door_damage, CONST_THRESHOLD, i)
        # slid_off_road feature true/false update
        change_df_values(testing_car_df, prediction_result_df, slid_off_road, CONST_THRESHOLD, i)
        # location feature true/false update
        three_selection_classifier(testing_car_df, prediction_result_df, location, 'UrbanLocation', 'SuburbanLocation',
                                   'RuralLocation', 'urban', 'suburban', 'rural', i)
        # time feature true/false update
        two_selection_classifier(testing_car_df, prediction_result_df, time, 'DayTime', 'NightTime', 'day', 'night', i)
        # weather feature true/false update
        three_selection_classifier(testing_car_df, prediction_result_df, weather, 'ClearWeather', 'RainyWeather',
                                   'SnowyWeather', 'clear', 'rainy', 'snowy', i)
        # road_condition true/false update
        three_selection_classifier(testing_car_df, prediction_result_df, road_condition, 'DryRoadCondition',
                                   'WetRoadCondition', 'SnowyRoadCondition', 'dry', 'wet', 'snowy', i)
        # bumps_dings true/false update
        change_df_values(testing_car_df, prediction_result_df, bumps_and_dings, CONST_THRESHOLD, i)
        # engine_damage true/false update
        change_df_values(testing_car_df, prediction_result_df, engine_damage, CONST_THRESHOLD, i)
        # severity true/false update
        five_selection_classifier(testing_car_df, prediction_result_df, severity, 'ExtremeDamageSeverity',
                                  'HighDamageSeverity', 'ModerateDamageSeverity', 'LowDamageSeverity',
                                  'NoDamageSeverity', 4, 3, 2, 1, 0, i)

    # print(testing_car_df[glass_damage], testing_car_df[water_damage])

    # drop unnecessary columns
    testing_car_df = testing_car_df.drop(columns=list(testing_car_df.columns.values)[1:3])

    # to keep record of all correct records
    classifier_TP_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # get accuracy for each column
    for i in range(len(testing_car_df)):

        if testing_car_df.iloc[i][glass_damage] == True:
            classifier_TP_list[0] += 1

        if testing_car_df.iloc[i][water_damage] == True:
            classifier_TP_list[1] += 1

        if testing_car_df.iloc[i][bumper_damage] == True:
            classifier_TP_list[2] += 1

        if testing_car_df.iloc[i][rolled_over] == True:
            classifier_TP_list[3] += 1

        if testing_car_df.iloc[i][side_door_damage] == True:
            classifier_TP_list[4] += 1

        if testing_car_df.iloc[i][slid_off_road] == True:
            classifier_TP_list[5] += 1

        if testing_car_df.iloc[i][location] == True:
            classifier_TP_list[6] += 1

        if testing_car_df.iloc[i][time] == True:
            classifier_TP_list[7] += 1

        if testing_car_df.iloc[i][weather] == True:
            classifier_TP_list[8] += 1

        if testing_car_df.iloc[i][road_condition] == True:
            classifier_TP_list[9] += 1

        if testing_car_df.iloc[i][bumps_and_dings] == True:
            classifier_TP_list[10] += 1

        if testing_car_df.iloc[i][engine_damage] == True:
            classifier_TP_list[11] += 1

        if testing_car_df.iloc[i][severity] == True:
            classifier_TP_list[12] += 1


    accuracy_list = list()

    for x in classifier_TP_list:
        accuracy = x/len(testing_car_df)
        accuracy_list.append(accuracy)

    accuracy_list = ['ACCURACY_RESULT'] + accuracy_list + ['']

    column_list = ['ImageFilename'] + classifier_list + ['Notes']

    # Convert the list of tag info to one row of dataFrame
    one_row_df = pd.DataFrame(data=[accuracy_list], columns=column_list)

    testing_car_df = testing_car_df.append(one_row_df, ignore_index=True)


    # creating two lists - precision and recall list with percentage values
    prediction_result_column_list = list(prediction_result_df.columns.values)[1:]


    precision_list, recall_list = get_performance(prediction_result_column_list)

    performance_column_list = ['Result'] + prediction_result_column_list
    precision_list = ['Precision'] + precision_list
    recall_list = ['Recall'] + recall_list

    performance_df = pd.DataFrame(data=[precision_list, recall_list], columns=performance_column_list)

    while True:
        try:
            testing_car_df.to_csv('result_df.csv', sep=',', index=False)
            print('Successfully Written to a CSV file!')
            break

        except PermissionError:
            print("Encountered Permission Error: please take an appropriate action.")
            print("\nEnter 'yes' to proceed or enter anything else to exit the program")
            answer = input()
            answer = answer.lower()

            if answer[0] != 'y':
                print('Ok, goodbye')
                exit()

    while True:
        try:
            performance_df.to_csv('performance_df.csv', sep=',', index=False)
            print('Successfully Written to a CSV file!')
            break

        except PermissionError:
            print("Encountered Permission Error: please take an appropriate action.")
            print("\nEnter 'yes' to proceed or enter anything else to exit the program")
            answer = input()
            answer = answer.lower()

            if answer[0] != 'y':
                print('Ok, goodbye')
                exit()


def change_df_values(testing_car_df, prediction_result_df, classifier, threshold, i):

    if prediction_result_df.iloc[i][classifier] >= threshold and \
            testing_car_df.iloc[i][classifier] == 'yes'\
            or prediction_result_df.iloc[i][classifier] < threshold and \
            testing_car_df.iloc[i][classifier] == 'no':
        testing_car_df.iloc[i, testing_car_df.columns.get_loc(classifier)] = True
    else:
        testing_car_df.iloc[i, testing_car_df.columns.get_loc(classifier)] = False


def two_selection_classifier(testing_car_df, prediction_result_df, classifier, pred_feature_one, pred_feature_two,
                             ans_one, ans_two, i):

    pred_feature_one_percentage = prediction_result_df.iloc[i][pred_feature_one]
    pred_feature_two_percentage = prediction_result_df.iloc[i][pred_feature_two]

    max_percentage = max(pred_feature_one_percentage, pred_feature_two_percentage)

    if max_percentage == pred_feature_one_percentage and testing_car_df.iloc[i][classifier] == ans_one or \
        max_percentage == pred_feature_two_percentage and testing_car_df.iloc[i][classifier] == ans_two:

        testing_car_df.iloc[i, testing_car_df.columns.get_loc(classifier)] = True

    else:
        testing_car_df.iloc[i, testing_car_df.columns.get_loc(classifier)] = False


def three_selection_classifier(testing_car_df, prediction_result_df, classifier, pred_feature_one, pred_feature_two,
                               pred_feature_three, ans_one, ans_two, ans_three, i):

    pred_feature_one_percentage = prediction_result_df.iloc[i][pred_feature_one]
    pred_feature_two_percentage = prediction_result_df.iloc[i][pred_feature_two]
    pred_feature_three_percentage = prediction_result_df.iloc[i][pred_feature_three]

    max_percentage = max(pred_feature_one_percentage, pred_feature_two_percentage, pred_feature_three_percentage)

    if max_percentage == pred_feature_one_percentage and testing_car_df.iloc[i][classifier] == ans_one or \
        max_percentage == pred_feature_two_percentage and testing_car_df.iloc[i][classifier] == ans_two or \
        max_percentage == pred_feature_three_percentage and testing_car_df.iloc[i][classifier] == ans_three:

        testing_car_df.iloc[i, testing_car_df.columns.get_loc(classifier)] = True

    else:
        testing_car_df.iloc[i, testing_car_df.columns.get_loc(classifier)] = False


def five_selection_classifier(testing_car_df, prediction_result_df, classifier, pred_feature_one, pred_feature_two,
                                pred_feature_three, pred_feature_four, pred_feature_five, ans_one, ans_two, ans_three,
                                ans_four, ans_five, i):

    pred_feature_one_percentage = prediction_result_df.iloc[i][pred_feature_one]
    pred_feature_two_percentage = prediction_result_df.iloc[i][pred_feature_two]
    pred_feature_three_percentage = prediction_result_df.iloc[i][pred_feature_three]
    pred_feature_four_percentage = prediction_result_df.iloc[i][pred_feature_four]
    pred_feature_five_percentage = prediction_result_df.iloc[i][pred_feature_five]

    max_percentage = max(pred_feature_one_percentage, pred_feature_two_percentage, pred_feature_three_percentage,
                         pred_feature_four_percentage, pred_feature_five_percentage)

    # print('max_percentage: ', max_percentage)
    # print(testing_car_df.iloc[i][classifier], ans_one, ans_two, ans_three, ans_four, ans_five)
    # if testing_car_df.iloc[i][classifier] == ans_one:
    #     print('Severity 0 matching!!!')

    if max_percentage == pred_feature_one_percentage and testing_car_df.iloc[i][classifier] == ans_one or \
        max_percentage == pred_feature_two_percentage and testing_car_df.iloc[i][classifier] == ans_two or \
        max_percentage == pred_feature_three_percentage and testing_car_df.iloc[i][classifier] == ans_three or \
        max_percentage == pred_feature_four_percentage and testing_car_df.iloc[i][classifier] == ans_four or \
        max_percentage == pred_feature_five_percentage and testing_car_df.iloc[i][classifier] == ans_five:

        testing_car_df.iloc[i, testing_car_df.columns.get_loc(classifier)] = True

    else:
        testing_car_df.iloc[i, testing_car_df.columns.get_loc(classifier)] = False


def get_performance(classifier_list):

    iteration_list = trainer.get_iterations(project_id)

    # print(iteration_list)

    # try catch here in case there is no iteration
    try:
        iteration_performance = trainer.get_iteration_performance(project_id, iteration_list[0].id,
                                                                  threshold=CONST_THRESHOLD * 0.01)

    except IndexError:
        print('There is no iteration available. Please create an iteration before running this script')
        exit()

    precision_dict = dict()
    recall_dict = dict()

    for performance in iteration_performance.per_tag_performance:
        # print(performance)
        precision_dict[performance.name] = performance.precision
        recall_dict[performance.name] = performance.recall

    precision_list = list()
    recall_list = list()

    # add precision values and recall values into list in an appropriate classifier order
    for classifier in classifier_list:
        precision_list.append(precision_dict[classifier])
        recall_list.append(recall_dict[classifier])

    return precision_list, recall_list


def analyze_two_excel_sheets_custom_model():

    classifier = 'BumperDamage'

    testing_car_df = csv_to_dataFrame(testing_car_df_path)
    prediction_result_df = csv_to_dataFrame(custom_model_result_df_location)

    # drop unnecessary columns
    testing_car_df = testing_car_df.drop(columns=list(testing_car_df.columns.values)[1:3])

    # apply true if correct prediction selected, false if not selected
    for i in range(len(testing_car_df)):
        yes_no_classifier_editor(testing_car_df, prediction_result_df, classifier, i)

    classifier_TP_score = 0

    # get accuracy
    for i in range(len(prediction_result_df)):

        if prediction_result_df.iloc[i][classifier] is True:
            classifier_TP_score += 1

    accuracy = classifier_TP_score/len(testing_car_df)

    accuracy_list = ['ACCURACY_RESULT', accuracy]

    column_list = ['ImageFilename', classifier]

    # Convert the list of tag info to one row of dataFrame
    one_row_df = pd.DataFrame(data=[accuracy_list], columns=column_list)

    prediction_result_df = prediction_result_df.append(one_row_df, ignore_index=True)

    while True:
        try:
            prediction_result_df.to_csv('custom_model_bumper_damage_result.csv', sep=',', index=False)
            print('Successfully Written to a CSV file!')
            break

        except PermissionError:
            print("Encountered Permission Error: please take an appropriate action.")
            print("\nEnter 'yes' to proceed or enter anything else to exit the program")
            answer = input()
            answer = answer.lower()

            if answer[0] != 'y':
                print('Ok, goodbye')
                exit()


def yes_no_classifier_editor(testing_car_df, prediction_result_df, classifier, i):

    if prediction_result_df.iloc[i][classifier] == 'yes' and \
            testing_car_df.iloc[i][classifier] == 'yes'\
            or prediction_result_df.iloc[i][classifier] == 'no' and \
            testing_car_df.iloc[i][classifier] == 'no':
        prediction_result_df.iloc[i, prediction_result_df.columns.get_loc(classifier)] = True
    else:
        prediction_result_df.iloc[i, prediction_result_df.columns.get_loc(classifier)] = False



def main():
    print('Please type \'custom_vision\' or \'custom_model\' to output result.')

    user_response = input()

    while user_response != 'custom_vision' and user_response != 'custom_model':
        print('Please type \'custom_vision\' or \'custom_model\' to output result.')
        user_response = input()

    if user_response == 'custom_vision':
        ### This is to analyze Custom Vision Services
        analyze_two_excel_sheets()
    else:
        ### To analyze custom model
        analyze_two_excel_sheets_custom_model()


if __name__ == "__main__":
    main()