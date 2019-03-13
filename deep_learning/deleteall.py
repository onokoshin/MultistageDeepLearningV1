from azure.cognitiveservices.vision.customvision.training import training_api

# project_id = "ac40ffed-201d-4040-b25d-a543801f7634"
# training_key = "b1beee07349649138fa1512fc196593a"
project_id = "8d05ada4-02ee-42f6-9784-9c6aa3457f9d"
training_key = "2fb2e54d14fc4c4e8fb264338277c250"
trainer = training_api.TrainingApi(training_key)


def delete_all_tags():

    tag_list = trainer.get_tags(project_id)

    for tag in tag_list:
        trainer.delete_tag(project_id, tag.id)


def delete_all_images():

    while True:
        #get all untagged images
        image_list = trainer.get_untagged_images(project_id)

        image_id_list = list()

        #append each image id into a list
        for image in image_list:
            image_id_list.append(image.id)

        print(len(image_list))

        #delete all untagged images in one batch(50) at a time -- this is the api's limit
        trainer.delete_images(project_id, image_id_list)

        #get updated image_list after deletion to check its length/count
        image_list = trainer.get_untagged_images(project_id)

        if len(image_list) == 0:
            break


def delete_iterations():

    iteration_list = trainer.get_iterations(project_id)

    for iteration in iteration_list:
        print(iteration)
        print(iteration.id)
        trainer.delete_iteration(project_id, iteration.id)

def delete_all():

    print("Are you sure you want to delete every tag and image? Yes/No")
    answer = input()
    answer = answer.lower()

    if answer[0] == 'y':
        delete_all_tags()
        delete_all_images()
        print("Bada bing bada boom! It's all gone :)")
        print("\nDo you need to delete iterations as well?")
        answer = input()
        answer = answer.lower()
        if answer[0] == 'y':
            delete_iterations()
            print("Tada!!! They are all gone!")

    else:
        print("Okie dokie, bye :)")


if __name__ == "__main__":
    delete_all()