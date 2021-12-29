# Image_Generator
The Image_Generator is created using the concept of Deep Convolutional GANs. The Generator is created using Transpose Convolution layers and it is the one crearting the images. The second part is the Discriminator created using Convolutional layers which discriminates the images as real or fake and feeds the output to the generator to make it better every epoch
## Input
The generator and discriminator have been trained on CIFAR-10 dataset

## Output
The generator tries to recreate the images and these images are available in the folder named <strong>results</strong>.

Real Image:
<br>
<img src="https://github.com/adupatil/Image_Generator/blob/main/results/real_sample.png" height=250 width=250>
<br>
Fake Image epoch 5:
<br>
<img src="https://github.com/adupatil/Image_Generator/blob/main/results/fake_samples_epoch_005.png" height=250 width=250>
<br>
Fake Image epoch 10:
<br>
<img src="https://github.com/adupatil/Image_Generator/blob/main/results/fake_samples_epoch_010.png" height=250 width=250>
<br>
Fake Image final epoch:
<br>
<img src="https://github.com/adupatil/Image_Generator/blob/main/results/fake_samples_epoch_019.png" height=250 width=250>
