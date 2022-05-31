import json
import pandas as pd
import math
import os

CLASSES = ['cow', 'sheep', 'dog', 'horse', 'cat']


def roundup(x):
    return int(math.ceil(x / 100.0)) * 100


def images_with_only_class(className, categories, annotations):
    # Computing the categories not include the category in question
    categoriesNoClass = categories[~(categories['name'] == className)]

    # Adding a boolean column that indicates whether the annotation is the class in queston
    annotations['HasNotClass'] = annotations['category_id'].isin(categoriesNoClass['id'])

    # Computing the sum of the above boolean column
    imageHasClass = annotations[['image_id', 'HasNotClass']].groupby('image_id').sum().reset_index()

    # Determining the images that only have that class
    imagesOnlyWithClass = imageHasClass[imageHasClass['HasNotClass'] == 0]['image_id'].drop_duplicates()

    return imagesOnlyWithClass

def main(type):
    f = open(f"/home/aidan/fiftyone/coco-2017-2/{type}/labels.json")

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    # Closing file
    f.close()

    # Creating dataframes
    images = pd.DataFrame(data['images'])
    annotations = pd.DataFrame(data['annotations'])
    categories = pd.DataFrame(data['categories'])

    # Filtering accordingly for the classes we want
    filteredCategories = categories[categories['name'].isin(CLASSES)]
    filteredAnnotations = annotations[annotations['category_id'].isin(filteredCategories['id'])]
    filteredImages = images[images['id'].isin(filteredAnnotations['image_id'])]

    # Computing Object Counts Before
    df = pd.merge(filteredCategories, filteredAnnotations, left_on='id', right_on='category_id')
    counts = df.groupby('name').size()
    print('Initial Counts:', counts)

    # # Getting images with only single object type
    # imagesOnlyWithPeople = images_with_only_class(className='person', categories=filteredCategories, annotations=filteredAnnotations)
    # imagesOnlyWithCows = images_with_only_class(className='cow', categories=filteredCategories, annotations=filteredAnnotations)
    # imagesOnlyWithsheep = images_with_only_class(className='sheep', categories=filteredCategories, annotations=filteredAnnotations)
    # imagesOnlyWithDogs = images_with_only_class(className='dog', categories=filteredCategories, annotations=filteredAnnotations)
    # imagesOnlyWithHorses = images_with_only_class(className='horse', categories=filteredCategories, annotations=filteredAnnotations)
    # imagesOnlyWithCats = images_with_only_class(className='cat', categories=filteredCategories, annotations=filteredAnnotations)
    #
    # # Computing samples with only single object type
    # imagesWithOnlyPeopleSample = df[df['image_id'].isin(imagesOnlyWithPeople.sample(frac=0.01, random_state=1))]
    # imagesWithOnlyCowsSample = df[df['image_id'].isin(imagesOnlyWithCows.sample(frac=0.25, random_state=1))]
    # imagesWithOnlySheepSample = df[df['image_id'].isin(imagesOnlyWithsheep.sample(frac=0.25, random_state=1))]
    # imagesWithOnlyDogsSample = df[df['image_id'].isin(imagesOnlyWithDogs.sample(frac=0.25, random_state=1))]
    # imagesWithOnlyHorsesSample = df[df['image_id'].isin(imagesOnlyWithHorses.sample(frac=0.25, random_state=1))]
    # imagesWithOnlyCatsSample = df[df['image_id'].isin(imagesOnlyWithCats.sample(frac=0.25, random_state=1))]
    #
    # # Computing basesample
    # baseSample = df[~df['image_id'].isin(imagesOnlyWithPeople)]
    # df = pd.concat(
    #     [baseSample,
    #      imagesWithOnlyPeopleSample,
    #      imagesWithOnlyCowsSample,
    #      imagesWithOnlySheepSample,
    #      imagesWithOnlyDogsSample,
    #      imagesWithOnlyHorsesSample,
    #      imagesWithOnlyCatsSample
    #      ], axis=0)
    # counts = df.groupby('name').size()
    # print('Counts New Counts:', counts)
    #
    # filteredAnnotations = filteredAnnotations[~filteredAnnotations['image_id'].isin(df['image_id'])]
    # filteredImages = filteredImages[~filteredImages['id'].isin(df['image_id'])]
    #
    data['annotations'] = filteredAnnotations.to_dict('records')
    data['categories'] = filteredCategories.to_dict('records')
    data['images'] = filteredImages.to_dict('records')

    # Getting the images we want to keep
    imageFileNames = [image['file_name'] for image in data['images']]

    # Removing images we don't want
    for image in os.listdir(f"/home/aidan/fiftyone/coco-2017-2/{type}/data/"):
        if image not in imageFileNames:
            os.remove(f"/home/aidan/fiftyone/coco-2017-2/{type}/data/{image}")



    # Dumping the new label
    jsonObject = json.dumps(data)

    with open(f"/home/aidan/fiftyone/coco-2017-2/{type}/labels.json", "w") as outfile:
        outfile.write(jsonObject)


if __name__ == '__main__':
    # main('train')
    main('validation')
