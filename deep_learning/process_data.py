'''
INSTRUCTION:
Before you run the script, please follow the steps below before starting.
    1. Please verify whether you have the correct project_id, training_key, and prediction_id.
        If not, please obtain them from setting in your custom vision project.
        One you obtain go to setting, copy paste the keys into appropriate global variables below.

    2. Please specify the local csv file where you have stored all the information related each
        each image. Even though this script contains data cleaning method, it does not cover
        every edge case.

    3. If you have added or removed classifier/feature, you may need to modify the code.

    4. If the api error occurs in prediction, there is a chance that your image is larger than 4 MB.
        Prediction images must be smaller than 4 MB, so please double-check your image size.
        Using Paint and its resize feature is a easy way to resize images.

'''

import pandas as pd
import sklearn.utils as sku
from azure.cognitiveservices.vision.customvision.training import training_api
from azure.cognitiveservices.vision.customvision.training.models import ImageUrlCreateEntry
import time as sleep_timer
from azure.cognitiveservices.vision.customvision.prediction import prediction_endpoint
from azure.cognitiveservices.vision.customvision.prediction.prediction_endpoint import models




### Global Constants: project related information -- obtain them from custom vision
project_id = "8d05ada4-02ee-42f6-9784-9c6aa3457f9d"
training_key = "2fb2e54d14fc4c4e8fb264338277c250"
trainer = training_api.TrainingApi(training_key)
prediction_key = "8df190e421c741a8b79ca00b56a3f23e"

# base_image_url is the base url from blob storage
base_image_url = "https://oltivainsurance8754.blob.core.windows.net/car-dataset/"

# set the local csv file location here
local_csv_location = r"/Users/Koshin/Google Drive/avanade_ai/CarDamageLabels_bumper_front_rear_angle_closeup.csv"


### Global Constants: classifier answer selection options
CONST_YES = 'yes'
CONST_NO = 'no'
CONST_URBAN = 'urban'
CONST_SUBURBAN = 'suburban'
CONST_RURAL = 'rural'
CONST_DAY = 'day'
CONST_NIGHT = 'night'
CONST_CLEAR = 'clear'
CONST_RAINY = 'rainy'
CONST_SNOWY = 'snowy'
CONST_DRY = 'dry'
CONST_WET = 'wet'
CONST_ZERO = 0
CONST_ONE = 1
CONST_TWO = 2
CONST_THREE = 3
CONST_FOUR = 4


# reads data from csv file and convert it to pandas dataFrame
def csv_to_dataFrame():

    with open(local_csv_location) as file:

        #reads the csv file
        df = pd.read_csv(file)

        df = df_data_cleaning(df)

    return df


# Cleans up data by making sure there is no empty space and character casing has been all up-to-date
# @param df: takes in a data frame and cleans it up
def df_data_cleaning(df):

    # drop all NaN values if exists
    df = df.dropna(thresh=2)

    # assigns the first column name to a variable
    fileName = df.columns.values[0]

    # naming validation
    if fileName != 'ImageFilename':
        # rename the first columns since it contains glitchy characters
        df.columns.values[0] = 'ImageFilename'

    classifier_list = list(df.columns.values)[3:16]

    # ensure that every attribute is in lower-case and there is no space after
    for row_index, row in df.iterrows():

        for classifier in classifier_list:

            # lower-casing and data cleaning apply to string input e.g. 'rural', 'yes', and etc
            if type(row[classifier]) == str:
                lower_case_value = row[classifier].lower()
                df.at[row_index, classifier] = lower_case_value

                if row[classifier][0] == 'y' and row[classifier][1] == 'e':
                    df.at[row_index, classifier] = CONST_YES
                elif row[classifier][0] == 'n' and row[classifier][1] == 'o':
                    df.at[row_index, classifier] = CONST_NO
                elif row[classifier][0] == 'u' and row[classifier][1] == 'r':
                    df.at[row_index, classifier] = CONST_URBAN
                elif row[classifier][0] == 's' and row[classifier][1] == 'u':
                    df.at[row_index, classifier] = CONST_SUBURBAN
                elif row[classifier][0] == 'r' and row[classifier][1] == 'u':
                    df.at[row_index, classifier] = CONST_RURAL
                elif row[classifier][0] == 'd' and row[classifier][1] == 'a':
                    df.at[row_index, classifier] = CONST_DAY
                elif row[classifier][0] == 'n' and row[classifier][1] == 'i':
                    df.at[row_index, classifier] = CONST_NIGHT
                elif row[classifier][0] == 'c' and row[classifier][1] == 'l':
                    df.at[row_index, classifier] = CONST_CLEAR
                elif row[classifier][0] == 'r' and row[classifier][1] == 'a':
                    df.at[row_index, classifier] = CONST_RAINY
                elif row[classifier][0] == 's' and row[classifier][1] == 'n':
                    df.at[row_index, classifier] = CONST_SNOWY
                elif row[classifier][0] == 'd' and row[classifier][1] == 'r':
                    df.at[row_index, classifier] = CONST_DRY
                elif row[classifier][0] == 'w' and row[classifier][1] == 'e':
                    df.at[row_index, classifier] = CONST_WET


    return df


# shuffles dataFrame
def df_shuffle(car_df):

    car_df = sku.shuffle(car_df)
    return car_df


# splits dataset to 80% training and 20% training dataset
def df_split(car_df):

    training_image_count = int(len(car_df)*0.8)

    training_car_df = car_df[0:training_image_count]
    testing_car_df = car_df[training_image_count:]

    return training_car_df, testing_car_df


# upload all the images to Custom Vision to create multi classification model
def process_training_car_df(training_car_df):


    # gets all the tags in the project
    tag_list = trainer.get_tags(project_id)

    img_name_list = list()
    img_dict = dict()

    # instantiate a dictionary and fill up keys
    for row_index, row in training_car_df.iterrows():
        image_name = row['ImageFilename']
        img_dict[image_name] = list()
        img_name_list.append(image_name)



    # Divide images to appropriate tags <-- create df for each tag ***strategize this part***
    classifier_list = list(training_car_df.columns.values)[3:16] # This gives us all the feature names

    #use this for loop to simply create tags and store them in img_dict
    for classifier in classifier_list:

        if classifier == 'Location':
            img_dict = three_answer_classifier(classifier, CONST_URBAN, CONST_SUBURBAN, CONST_RURAL, training_car_df,
                                               img_dict, tag_list)

        elif classifier == 'Time':
            img_dict = two_answer_classifier(classifier, CONST_DAY, CONST_NIGHT, training_car_df, img_dict, tag_list)

        elif classifier == 'Weather':
            img_dict = three_answer_classifier(classifier, CONST_CLEAR, CONST_RAINY, CONST_SNOWY, training_car_df,
                                               img_dict, tag_list)

        elif classifier == 'RoadCondition':
            img_dict = three_answer_classifier(classifier, CONST_DRY, CONST_WET, CONST_SNOWY, training_car_df,
                                               img_dict, tag_list)

        elif classifier == 'Severity':
            img_dict = five_answer_classifier(classifier, CONST_ZERO, CONST_ONE, CONST_TWO, CONST_THREE, CONST_FOUR,
                                              training_car_df, img_dict, tag_list)

        # Any classifier that has yes/no answer goes to else
        else:
            img_dict = yes_no_classifier(classifier, training_car_df, img_dict, tag_list)


    print("\nAdding images...")
    # this is for YesDamage
    for img_name in img_name_list:
        image_url = base_image_url + img_name
        tag_list = img_dict[img_name]

        trainer.create_images_from_urls(project_id,
                                        [ImageUrlCreateEntry(url=image_url, tag_ids=tag_list)])

        # print('uploading {} \t Tags: {}' .format(image_url, tag_list))
        print('uploading {} '.format(image_url))

'''
this function append the appropriate tags to classifiers that only have yes/no answers
@param classifier - indicates which classifier this is
@param training_car_df - training_car data frame
@param tag_list - all tags obtained from trainer object
'''
def yes_no_classifier(classifier, training_car_df, img_dict, tag_list):

    selection = 'yes'

    # delete the tag if it exists
    for tag in tag_list:
        if tag.name == classifier:
            trainer.delete_tag(project_id, tag.id)

    # yes tag created
    tag = trainer.create_tag(project_id, classifier)

    # rows with yes goes to df_yes
    df_yes = training_car_df[training_car_df[classifier] == selection]

    # Using dict to store image name as key and each classifier tag as value
    for row_index, row in df_yes.iterrows():
        image_name = row['ImageFilename']
        img_dict[image_name].append(tag.id)

    return img_dict


'''
@param classifier:  this is a feature name. e.g. GlassDamage, EngineDamage and etc
@param answer_option_one: answer selection 1. e.g. yes
@param answer_option_two: answer selection 2. e.g. no
@param training_car_df: training dataFrame (80%)
@param img_dict: a dictionary that contains  image file name as key and each feature selection as value
@param trainer: trainer object
@param project_id: this is the custom vision project id 
'''
def two_answer_classifier(classifier, answer_option_one, answer_option_two, training_car_df, img_dict, tag_list):

    tag_name_one = answer_option_one[0].upper() + answer_option_one[1:] + classifier
    tag_name_two = answer_option_two[0].upper() + answer_option_two[1:] + classifier

    val_one = answer_option_one
    val_two = answer_option_two

    # delete tags that are already available so that it won't cause errors
    for tag in tag_list:
        if tag.name == tag_name_one:
            trainer.delete_tag(project_id, tag.id)
        elif tag.name == tag_name_two:
            trainer.delete_tag(project_id, tag.id)

    # two selection tags created -- e.g. day/night
    tag_one = trainer.create_tag(project_id, tag_name_one)
    tag_two = trainer.create_tag(project_id, tag_name_two)

    # rows with yes goes to dfYes and rows with no goes to dfNo
    df_one = training_car_df[training_car_df[classifier] == val_one]
    df_two = training_car_df[training_car_df[classifier] == val_two]

    # Using dict to store image name as key and each classifier tag as value
    for row_index, row in df_one.iterrows():
        image_name = row['ImageFilename']
        img_dict[image_name].append(tag_one.id)

    for row_index, row in df_two.iterrows():
        image_name = row['ImageFilename']
        img_dict[image_name].append(tag_two.id)

    return img_dict


'''
@param classifier:  this is a feature name. e.g. location, weather and etc
@param answer_option_one: answer selection 1. e.g. urban
@param answer_option_two: answer selection 2. e.g. suburban
@param answer_option_two: answer selection 3. e.g. rural
@param training_car_df: training dataFrame (80%)
@param img_dict: a dictionary that contains  image file name as key and each feature selection as value
@param trainer: trainer object
@param project_id: this is the custom vision project id 
'''
def three_answer_classifier(classifier, answer_option_one, answer_option_two, answer_option_three,
                            training_car_df, img_dict, tag_list):

    tag_name_one = answer_option_one[0].upper() + answer_option_one[1:] + classifier
    tag_name_two = answer_option_two[0].upper() + answer_option_two[1:] + classifier
    tag_name_three = answer_option_three[0].upper() + answer_option_three[1:] + classifier

    val_one = answer_option_one
    val_two = answer_option_two
    val_three = answer_option_three

    # delete tags that are already available so that it won't cause errors
    for tag in tag_list:
        if tag.name == tag_name_one:
            trainer.delete_tag(project_id, tag.id)
        elif tag.name == tag_name_two:
            trainer.delete_tag(project_id, tag.id)
        elif tag.name == tag_name_three:
            trainer.delete_tag(project_id, tag.id)

    # tags being created
    tag_one = trainer.create_tag(project_id, tag_name_one)
    tag_two = trainer.create_tag(project_id, tag_name_two)
    tag_three = trainer.create_tag(project_id, tag_name_three)

    # rows with yes goes to dfYes and rows with no goes to dfNo
    df_one = training_car_df[training_car_df[classifier] == val_one]
    df_two = training_car_df[training_car_df[classifier] == val_two]
    df_three = training_car_df[training_car_df[classifier] == val_three]

    # Using dict to store image name as key and each classifier tag as value
    for row_index, row in df_one.iterrows():
        image_name = row['ImageFilename']
        img_dict[image_name].append(tag_one.id)

    for row_index, row in df_two.iterrows():
        image_name = row['ImageFilename']
        img_dict[image_name].append(tag_two.id)

    for row_index, row in df_three.iterrows():
        image_name = row['ImageFilename']
        img_dict[image_name].append(tag_three.id)

    return img_dict


'''
@param classifier:  this is a feature name. e.g. Severity and etc
@param answer_option_one: answer selection 1. e.g. 0
@param answer_option_two: answer selection 2. e.g. 1
@param answer_option_two: answer selection 3. e.g. 2
@param answer_option_two: answer selection 4. e.g. 3
@param answer_option_two: answer selection 5. e.g. 4
@param training_car_df: training dataFrame (80%)
@param img_dict: a dictionary that contains  image file name as key and each feature selection as value
@param trainer: trainer object
@param project_id: this is the custom vision project id 
'''
def five_answer_classifier(classifier, answer_option_one, answer_option_two, answer_option_three,
                            answer_option_four, answer_option_five, training_car_df, img_dict, tag_list):

    val_one = answer_option_one
    val_two = answer_option_two
    val_three = answer_option_three
    val_four = answer_option_four
    val_five = answer_option_five

    # create string
    if type(answer_option_one) == int and answer_option_one == 0:

        answer_option_one = 'NoDamage'
        answer_option_two = 'LowDamage'
        answer_option_three = 'ModerateDamage'
        answer_option_four = 'HighDamage'
        answer_option_five = 'ExtremeDamage'

    tag_name_one = answer_option_one[0].upper() + answer_option_one[1:] + classifier
    tag_name_two = answer_option_two[0].upper() + answer_option_two[1:] + classifier
    tag_name_three = answer_option_three[0].upper() + answer_option_three[1:] + classifier
    tag_name_four = answer_option_four[0].upper() + answer_option_four[1:] + classifier
    tag_name_five = answer_option_five[0].upper() + answer_option_five[1:] + classifier

    # delete tags that are already available so that it won't cause errors
    for tag in tag_list:
        if tag.name == tag_name_one:
            trainer.delete_tag(project_id, tag.id)
        elif tag.name == tag_name_two:
            trainer.delete_tag(project_id, tag.id)
        elif tag.name == tag_name_three:
            trainer.delete_tag(project_id, tag.id)
        elif tag.name == tag_name_four:
            trainer.delete_tag(project_id, tag.id)
        elif tag.name == tag_name_five:
            trainer.delete_tag(project_id, tag.id)

    # tags being created
    tag_one = trainer.create_tag(project_id, tag_name_one)
    tag_two = trainer.create_tag(project_id, tag_name_two)
    tag_three = trainer.create_tag(project_id, tag_name_three)
    tag_four = trainer.create_tag(project_id, tag_name_four)
    tag_five = trainer.create_tag(project_id, tag_name_five)

    # rows with yes goes to dfYes and rows with no goes to dfNo
    df_one = training_car_df[training_car_df[classifier] == val_one]
    df_two = training_car_df[training_car_df[classifier] == val_two]
    df_three = training_car_df[training_car_df[classifier] == val_three]
    df_four = training_car_df[training_car_df[classifier] == val_four]
    df_five = training_car_df[training_car_df[classifier] == val_five]

    # Using dict to store image name as key and each classifier tag as value
    for row_index, row in df_one.iterrows():
        image_name = row['ImageFilename']
        img_dict[image_name].append(tag_one.id)

    for row_index, row in df_two.iterrows():
        image_name = row['ImageFilename']
        img_dict[image_name].append(tag_two.id)

    for row_index, row in df_three.iterrows():
        image_name = row['ImageFilename']
        img_dict[image_name].append(tag_three.id)

    for row_index, row in df_four.iterrows():
        image_name = row['ImageFilename']
        img_dict[image_name].append(tag_four.id)

    for row_index, row in df_five.iterrows():
        image_name = row['ImageFilename']
        img_dict[image_name].append(tag_five.id)

    return img_dict


# Conducts training process upon uploading all the images with appropriate tags
def training():

    print("Training...")
    iteration = trainer.train_project(project_id)

    while (iteration.status != "Completed"):
        iteration = trainer.get_iteration(project_id, iteration.id)
        print("Training status: " + iteration.status)
        sleep_timer.sleep(1)

    # The iteration is now trained. Make it the default project endpoint
    trainer.update_iteration(project_id, iteration.id, is_default=True)
    print("Done!")

    return iteration


# all prediction related work is done in this function
def prediction(testing_car_df, iteration):

    # Now there is a trained endpoint that can be used to make a prediction
    predictor = prediction_endpoint.PredictionEndpoint(prediction_key)

    img_name_list = list()

    # instantiate a dictionary and fill up keys
    for row_index, row in testing_car_df.iterrows():
        image_name = row['ImageFilename']
        # img_dict[image_name] = list()
        img_name_list.append(image_name)

    column_list = ['ImageFilename', 'GlassDamage', 'WaterDamage', 'BumperDamage', 'RolledOver', 'SideDoorDamage',
                   'SlidOffRoad', 'UrbanLocation', 'SuburbanLocation', 'RuralLocation', 'DayTime', 'NightTime',
                   'ClearWeather', 'RainyWeather', 'SnowyWeather',
                   'DryRoadCondition', 'WetRoadCondition', 'SnowyRoadCondition', 'Bumps&Dings', 'EngineDamage',
                   'ExtremeDamageSeverity', 'HighDamageSeverity', 'ModerateDamageSeverity', 'LowDamageSeverity',
                   'NoDamageSeverity']

    result_df = pd.DataFrame(columns=column_list)
    df_list = list()

    for img_name in img_name_list:
        test_img_url = base_image_url + img_name
        results = predictor.predict_image_url(project_id, iteration.id, url=test_img_url)

        # Alternatively, if the images were on disk in a folder called Images alongside the sample.py, then
        # they can be added by using the following.
        #
        # Open the sample image and get back the prediction results.
        # with open("Images\\test\\test_image.jpg", mode="rb") as test_data:
        #     results = predictor.predict_image(project.id, test_data, iteration.id)


        # create a dictionary to store all the tag information related to one image
        tag_dict = dict(zip(column_list, column_list))

        # insert tag name into the dictionary value
        tag_dict['ImageFilename'] = img_name

        # Display the results.
        print('\n' + 'Uploading Prediction Image ' + img_name + ':')
        for prediction in results.predictions:
            print("\t" + prediction.tag_name + ": {0:.2f}%".format(prediction.probability * 100))
            tag_dict[prediction.tag_name] = prediction.probability * 100

        # convert all the dict values aka tag values into a list
        tag_result_list = list(tag_dict.values())

        # Convert the list of tag info to one row of dataFrame
        one_row_df = pd.DataFrame(data=[tag_result_list], columns=column_list)

        # Append the one row dataFrame to result_df, which combines all results
        result_df = result_df.append(one_row_df, ignore_index=True)
        # print(result_df)

    csv_output(result_df, 'prediction_result.csv')


# creates a output csv file based on an input dataframe
# @param df - A csv file gets created based on this dataFrame
# @param csv_output_name - Simply, a name of this csv file
def csv_output(df, csv_output_name):

    print('\n{}.csv file is being created...'.format(csv_output_name))
    while True:
        try:
            # compare the reult_df with testing_car_df
            df.to_csv(csv_output_name, sep=',', index=False)
            print('Successfully created {} file!'.format(csv_output_name))
            break

        except PermissionError:
            print("Encountered Permission Error: please take an appropriate action!")
            print("\nEnter 'yes' to proceed or enter anything else to exit the program")
            answer = input()
            answer = answer.lower()

            if answer[0] != 'y':
                print('Ok, goodbye')
                exit()


# main function where the whole program gets executed
def main():
    # reads the data from a csv file
    car_df = csv_to_dataFrame()

    # shuffles the data to randomize them
    car_df = df_shuffle(car_df)

    # split dataset to 80% training and 20% training dataset
    training_car_df, testing_car_df = df_split(car_df)
    print('Testing data size:', len(testing_car_df), 'images')
    print('Training data size:', len(training_car_df), 'images')

    # write out training_car_df and teseting_car_df to a csv file
    # The results will be used for deep learning custom model as well
    csv_output(training_car_df, 'training_car.csv')
    csv_output(testing_car_df, 'testing_car.csv')

    iteration_list = trainer.get_iterations(project_id)

    if len(iteration_list) == 0:
        process_training_car_df(training_car_df)

        iteration = training()
    else:
        iteration = iteration_list[0]

    prediction(testing_car_df, iteration)
    print("\nFinish!!")


if __name__ == "__main__":
    main()









