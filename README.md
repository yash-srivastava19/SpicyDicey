# SpicyDicey
A CNN for detecting the number that appears on an image of a dice.

Formerly called DiceMNISt, SpicyDicey is a CNN which predicts the number that appears on a dice. All of the work in collecting the data and editing the images has been done individually - that is why they are less in number , so feel free to add to it for a more robust model.

<h3> For creating your own dataset: </h3>
Each image of a dice is cropped,grayscaled and resized to 28x28 pixel. It is then flipped horizontally,vertically,rotated 90 clockwise,90 counter clockwise and 180 to create a large dataset. The images has a specific nomenclature in order to create datasets easily. If you add your own images, make sure to adjust the "Dataset.py" file accordingly.

<p1>Edit: For a second version of SpiceyDicey, we are able to produce a sufficient dataset to prevent overfitting. The model after this pretty strainghforward and is updated wherever deemed necessary.</p1>

<p1>For more detailed overview of the model,read the DiceMNIST paper in the repository.Also, feel free to correct or add anything. (I am very susceptible to mistakes) </p1>


<p>[TensorBoard](https://tensorboard.dev/experiment/TzCA1ZjpREaEcgx3BVZmGQ/)</p>
