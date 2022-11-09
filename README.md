[//]: <> (title)
# **Cleaning NIH Chest X-ray Dataset Using A Trained Image Classifier**




Negative  | Positive
------------- | -------------
![fig 1](https://github.com/Patrick-Frisella/Patrick-Frisella/blob/main/2.png)  | ![fig caption](https://github.com/Patrick-Frisella/Patrick-Frisella/blob/main/4.png)









## **Synopsis**
---

Data integrity is the foundation of a reliable prediction model. The phrase "garbage in, garbage out" best illustrates the performance of a model when using poor quality data.

Chest x-rays are a ubiquitous, low cost radiological modality used in hospitals, urgent care clinics,and other medical facility for initial examination of common acute and chronic medical complaints. The NIH Chest X-ray dataset consists of 112,120 images with 15 different classes/labels. The image labels were extracted from the original medical reports using natural language processing (NLP). This led to numerous errors in the dataset and resulted in a poor performing models. In addition to labeling errors (finding labels, age, views, etc.), the dataset has other confounding attributes that make training a Convolutional Neural Network (CNN) model difficult.

In addition to labeling errors (finding labels, age, views, etc.), the dataset has other confounding attributes that
make training a Convolutional Neural Network (CNN) model difficult.
These include:

Aspects of the chest x-ray image that can confound the Convolutional Neural Network learning include:
- Surgical clips, cosmetic piercings, leads, contrast dyes, lines, shields, and catheters
- Arthroplasties, other medical devices

Overall Image Quality:
- Variation in technique quality
- Portions of the image that are extrathoracic (area outside the chest)





![fig 2](https://github.com/Patrick-Frisella/Patrick-Frisella/blob/main/1_u9GAvmHcjzksFS-dHxh_9Q.png)
<figcaption align = "center"><b><sub>A. Example of better quality image with no artifacts or devices B. Diminished image quality, artifact and devices C. Catheter. Images from NIH dataset</sup></b></figcaption><b> 




### Design
---
First, the dataset was explored. New image folders were created and standardized for: posteroanterior (PA) chest view, age, and finding labels. These folders were then examined and poor data integrity was confirmed by visual inspection. 

The dataset "Clean" was created using the age parameter of 20–40 years. This age group tends to have chest x-ray with typically better image quality which should make it easier for the model to learn. The images were re-interpreted into a binary classification of positive and negative pulmonary disease instead of the specific classification or diagnosis in the original dataset. Several folders were created with finding labels: "Infiltrate", "Pneumonia", and "Consolidation"; multiple categories were needed due to the lack of true positive images in the overall dataset. These folders were explored and examined for integrity of findings (meaning, did the chest x-ray image have said findings). If the images were deemed to be labeled correctly, they were placed in a new folder, "Positive". The same was done for the finding label "No Finding" and then placed in a folder, "Negative". Care was taken to appropriately include catheters, devices and shielding artifacts when available.

The Junk dataset was created from the original dataset with the same parameters as above but were not inspected for integrity and used "as is". Images with the finding labels "No Finding" were placed in a folder, "Junk_negative". Another set with the "Infiltrate" label was placed in "Junk_positive".

These datasets were used to train a simple CNN model for binary classification of chest x-ray images using the TensorFlow library.





### Data Preprocessing & Augmentation

---
``` Python

dgen_train = ImageDataGenerator(rescale= 1./255,
                                validation_split= 0.2,
                                zoom_range= 0.2,
                                horizontal_flip =False)

dgen_validation = ImageDataGenerator(rescale= 1./255 )
dgen_test = ImageDataGenerator(rescale= 1./255)



train_generator= dgen_train.flow_from_directory(train_dir,
                                                target_size= (256, 256),
                                                color_mode= 'rgb',
                                                subset= "training",
                                                batch_size= 32,
                                                class_mode= "binary")

validation_generator= dgen_train.flow_from_directory(train_dir,
                                                target_size= (256, 256),
                                                color_mode= 'rgb',
                                                subset= "validation",
                                                batch_size= 32,
                                                class_mode= "binary")

test_generator= dgen_test.flow_from_directory(test_dir,
                                              target_size= (256, 256),
                                              color_mode= 'rgb',
                                              batch_size= 32,
                                              class_mode= "binary")

```

### Convolutional Neural Network Model:

---
``` Python
model= tf.keras.models.Sequential([

tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(256, 256, 3)),
tf.keras.layers.MaxPooling2D(2, 2),

tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),

tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),

tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),

tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),

tf.keras.layers.Flatten(),

tf.keras.layers.Dense(512, activation='relu'),

tf.keras.layers.Dense(1, activation='sigmoid')
])
```
### Performance

---
#### "Junk" Dataset:

The Junk dataset performed poorly with validation accuracy less than 59%:

```Python
loss, acc = Junk.model.evaluate(validation_generator, verbose=2)
print("Model, accuracy: {:5.2f}%".format(100 * acc))

Output:
4/4 - 12s - loss: 0.6813 - accuracy: 0.5847 - 12s/epoch - 3s/step
Model, accuracy: 58.47%
```


This result mirrored those by other investigators.

#### "Clean" Dataset:
The new dataset was used to train the same model as the "Junk" data. It did train better although there was over-fitting due to the dataset size.

```Python
loss, acc = Clean.model.evaluate(validation_generator, verbose=2)
print("Model, accuracy: {:5.2f}%".format(100 * acc))

Output:
3/3 - 7s - loss: 0.4810 - accuracy: 0.8576 - 7s/epoch - 2s/step
Restored model, accuracy: 85.76%
```


#### "Filtered" Dataset:
CXR with the labels of "Infiltrate" and "No Findings" (a file of 10,000 randomly selected images from the 39,000 images
with this label and PA view) were run through the classifier and sorted in either a folder for "Positive" findings or
"Negative" findings. This new dataset was used to train the same model as both "Clean" and "Junk".
Even though the dataset is still not perfect, the model outperformed the other dataset trained models.

``` Python
loss, acc = Filtered.model.evaluate(validation_generator, verbose=2)
print("Model, accuracy: {:5.2f}%".format(100 * acc))

Output:
21/21 - 70s - loss: 0.2189 - accuracy: 0.9062 - 70s/epoch - 3s/step
Restored model, accuracy: 90.62%
```



### Conclusion

---
This is a simple approach to cleaning a poor integrity image dataset. The performance of the "Filtered" data model improved significantly from the "Junk" dataset. It was slightly better than the "Clean" dataset, likely due to the model training better with the larger "Filtered" dataset (15,273 images vs. 1,853 images,respectively). Upon further inspection, the same confounding issues (with respect to image quality) were still present. This h affected performance of the model when new unseen images (n = 10) were introduced (prediction accuracy varied when independently inspected). 

These are "real world" images with variability in overall quality.  Building a model to accurately predict the correct diagnosis with this dataset would take a multi-disciplinary team with ample domain knowledge and machine learning expertise.



### Disclaimer

---

 This is not a medical device and not to be used for diagnostic or treatment of medical conditions.

 THIS IS AN ACADEMIC EXERCISE ONLY!


### Links

---

[NIH_Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data)







