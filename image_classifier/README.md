Training
========
The image classifier requires training images to use.
in this example flower_photos/roses is one such directory
the label given to the images is from the folder name, in this case roses

flower_photos folder is available:
http://download.tensorflow.org/example_images/flower_photos.tgz

to train.

```
cd /vagrant/image_classifier
python retrain.py \
--bottleneck_dir=/vagrant/image_classifier/tmp/bottlenecks \
--how_many_training_steps 500 \
--model_dir=/vagrant/image_classifier/tmp/inception \
--output_graph=/vagrant/image_classifier/tmp/retrained_graph.pb \
--output_labels=/vagrant/image_classifier/tmp/retrained_labels.txt \
--image_dir /vagrant/image_classifier/flower_photos
```

```
--how_many_training_steps
```
can be removed, the default is 4000

The retraining script will write out a version of the Inception v3 network with a final layer retrained to your 
categories to retrained_graph.pb and a text file containing the labels to retrained_labels.txt.

see label_image.py for how to use the graph and labels

```
cd /vagrant/image_classifier
python label_image.py /vagrant/image_classifier/flower_photos/daisy/21652746_cc379e0eea_m.jpg
```
