import numpy as np
from sklearn.metrics import confusion_matrix
import csv
import matplotlib.pyplot as plt
from get_valid_nearest_neighbor import get_class_names_dict_and_labels
import itertools

def plot_2_conf_mat(cm1,cm2):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.imshow(cm1)
    #ax1.imshow(cm1)
    ax1.set_ylabel('True label', fontsize=18)
    ax1.set_xlabel('Predicted label', fontsize=18)
    ax1.set_title('sorted by number of images', fontsize=18)
    im = ax2.imshow(cm2)
    #im = ax2.imshow(cm2)
    ax2.set_xlabel('Predicted label', fontsize=18)
    ax2.set_title('sorted by periods', fontsize=18)
    ax1.tick_params(labelsize=14)
    ax2.tick_params(labelsize=14)


    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.2, 0.02, 0.6])
    fig.colorbar(im, cax=cbar_ax)
    #plt.tight_layout()

    plt.show()

def plot_confusion_matrix(cm, classes,experient,
                          normalize=False,
                          title='Confusion matrix, labels are sites',
                          cmap=plt.cm.Blues,):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    plt.colorbar()

    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/' + experient + '.png')
    return cm

def get_site_period_str(class_name,class_mode):
    if class_mode == 'site_period':
        period, site = class_name.split('_')
    elif class_mode == 'site_period_sorted':
        period, site = class_name.split('_')
    elif class_mode == 'period_sorted':
        period = class_name
        site = ''
    elif class_mode == 'site':
        site = class_name
        period = ''
    else:
        print('invalid class mode')
        raise

    return period,site

def get_confusion_matrix(experiment,data,class_names, num_of_relevant_neighbors = -1):


    if num_of_relevant_neighbors == -1:
        y_true = data[:,0]
        y_pred = data[:,1]
    else:
        N = data[:,0].shape[0]
        y_true = np.zeros((N*num_of_relevant_neighbors, ), dtype=np.int32)
        y_pred = np.zeros((N * num_of_relevant_neighbors,), dtype=np.int32)
        for mm in range(num_of_relevant_neighbors):
            y_true[mm*N:(mm+1)*N] = data[:,0]
            y_pred[mm*N:(mm + 1) * N] = data[:, mm + 1]
        print('aaa')
        print(y_true.shape)
    conf = confusion_matrix(y_true, y_pred)
    conf_norm = plot_confusion_matrix(conf, [], experiment, title='label = cm',
                                      normalize=True)
    plt.show()

    N = 10
    lines = ''
    for k in range(conf_norm.shape[0]):
        vec = conf_norm[k,:]
        ind_max = np.flip(np.argsort(vec), axis=0)
        ind_max = ind_max[:N]

        scores = vec[ind_max]
        scores = np.trim_zeros(scores, 'b')


        lines = lines + 'true_id, {}, true_label, {}\n'.format(k, class_names[k])

        for m in range(scores.shape[0]):
            pred_id = ind_max[m]
            temp_str = 'pred_id, {}, pred_class, {}, pred_score, {:0.3f}\n'.format(pred_id, class_names[pred_id], scores[m])
            lines = lines + temp_str

        lines = lines + '\n'

    with open('results/summary_' + experient + '.csv', "w") as text_file:
        text_file.write(lines)

    return conf_norm




experient = 'efficientNetB3_try7'
data = np.genfromtxt('conf_mat_data/efficientNetB3_try7__data.csv', delimiter=',')
labels_file = 'labels/efficientNetB3_try7_train.csv'
class_names, labls = get_class_names_dict_and_labels(labels_file)
cm = get_confusion_matrix(experient,data,class_names)


