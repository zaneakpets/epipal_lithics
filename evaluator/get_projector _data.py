import tensorflow as tf
import numpy as np
import os
import csv
from get_valid_nearest_neighbor import get_class_names_dict_and_labels

def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    spriteimage = np.ones((img_h * n_plots, img_w * n_plots))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                j * img_w:(j + 1) * img_w] = this_img

    return spriteimage


def vector_to_matrix_mnist(mnist_digits):
    """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
    return np.reshape(mnist_digits, (-1, 28, 28))


def invert_grayscale(mnist_digits):
    """ Makes black white, and white black """
    return 1 - mnist_digits


valid_embeddings_csv = 'embeddings/efficientNetB3_try7_valid.csv'
valid_labesl_csv = 'labels/efficientNetB3_try7_valid.csv'
experient = 'efficientNetB3_try7'

cls_names, labels_list = get_class_names_dict_and_labels(valid_labesl_csv)
class_names = [ v for v in cls_names. values() ]




labels = np.asarray(labels_list,dtype=np.int)
embeddings_numpy = np.genfromtxt(valid_embeddings_csv, delimiter=',')


LABELS = os.path.join(os.getcwd(), "labels_names.tsv")
LOGDIR = 'projector/' + experient
#SPRITES = 'spritesheet.png'eet.png'eet.png'eet.png'
sess = tf.Session()


lines = 'id \t name\n'
for k in range(labels.shape[0]):
    class_ = class_names[np.int(labels[k])].replace(',', '_')
    lines = lines + '{} \t {}\n'.format(np.int(labels[k]),class_)

with open("labels_names.tsv", "w") as text_file:
    text_file.write(lines)



embedding = tf.Variable(tf.zeros([embeddings_numpy.shape[0], embeddings_numpy.shape[1]]), name="test_embedding")
assignment = embedding.assign(embeddings_numpy)
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter(LOGDIR, sess.graph)
#writer = tf.summary.FileWriter(LOGDIR + hparam)
writer.add_graph(sess.graph)

config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
embedding_config = config.embeddings.add()
embedding_config.tensor_name = embedding.name
#embedding_config.sprite.image_path = SPRITES
embedding_config.metadata_path = LABELS
# Specify the width and height of a single thumbnail.
#embedding_config.sprite.single_image_dim.extend([30, 30])
#embedding_config.sprite.single_image_dim
tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

sess.run(assignment)
saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), 0)
