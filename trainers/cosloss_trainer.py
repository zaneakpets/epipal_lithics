from base.base_trainer import BaseTrain
import os
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, LambdaCallback
from trainers.learning_rate_scd import step_decay_wrapper
import numpy as np
from data_loader.default_generator import get_testing_generator
from evaluator.get_valid_nearest_neighbor import eval_model
import json
import csv
from json import encoder
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import matplotlib.patches as patches




class CosLossModelTrainer(BaseTrain):
    def __init__(self, model, train_generator, valid_generator,train_generator_testing, config, class_list):
        super(CosLossModelTrainer, self).__init__(model, train_generator, config)
        self.valid_generator = valid_generator
        self.train_generator_testing = train_generator_testing
        self.callbacks = []
        self.loss = []
        self.val_loss = []
        self.val_acc = []
        self.step_decay_function = step_decay_wrapper(self.config)
        self.class_list = class_list
        encoder.FLOAT_REPR = lambda o: format(o, '.2f')

        cnt = 0

        self.init_callbacks()



    def init_callbacks(self):
        #filepath = os.path.join(self.config.callbacks.checkpoint_dir,
        #                        '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name)
        if self.config.callbacks.is_save_model:
            self.callbacks.append(
                ModelCheckpoint(
                    #self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name
                    filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_out_acc_1:.2f}.hdf5' % self.config.exp.name),
                    monitor=self.config.callbacks.checkpoint_monitor,
                    mode=self.config.callbacks.checkpoint_mode,
                    save_best_only=self.config.callbacks.checkpoint_save_best_only,
                    save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                    verbose=self.config.callbacks.checkpoint_verbose,
                )
            )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )

        if self.config.trainer.is_early_stop:
            self.callbacks.append(
                EarlyStopping(
                    monitor='val_loss', min_delta=self.config.trainer.EarlyStopping_min_delta, patience=self.config.trainer.EarlyStopping_patience, verbose=1, mode='auto')
            )

        if self.config.trainer.is_change_lr:
            if self.config.trainer.learning_rate_schedule_type == 'ReduceLROnPlateau':
                self.callbacks.append(
                    ReduceLROnPlateau(monitor='val_loss', factor=self.config.trainer.lr_decrease_factor,
                                                patience=self.config.trainer.ReduceLROnPlateau_patience, min_lr=1e-9)
                )
            elif self.config.trainer.learning_rate_schedule_type == 'LearningRateScheduler':
                self.callbacks.append(
                    LearningRateScheduler(self.step_decay_function)
            )


        if self.config.model.loss == 'triplet':
            if self.config.model.batch_type == 'hard':
                self.callbacks.append(
                    LambdaCallback(on_epoch_end=lambda epoch, logs: self.custom_epoch_end(epoch,logs,'triplet_hard'),
                        on_train_end=lambda logs: self.json_log.close())
                    )
            else:
                self.callbacks.append(
                    LambdaCallback(on_epoch_end=lambda epoch, logs: self.custom_epoch_end(epoch,logs,'triplet_all'),
                        on_train_end=lambda logs: self.json_log.close())
                    )
        elif self.config.model.loss == 'cosface':
            self.callbacks.append(
                LambdaCallback(on_epoch_end=lambda epoch, logs: self.custom_epoch_end(epoch,logs, 'cosface'),
                               on_train_end=lambda logs: self.json_log.close())
            )
        elif self.config.model.loss == 'softmax':
            self.callbacks.append(
                LambdaCallback(on_epoch_end=lambda epoch, logs: self.custom_epoch_end(epoch,logs, 'softmax'),
                               on_train_end=lambda logs: self.json_log.close())
            )


        #if hasattr(self.config,"comet_api_key"):
        #    from comet_ml import Experiment
        #    experiment = Experiment(api_key=self.config.comet_api_key, project_name=self.config.exp_name)
        #    experiment.disable_mp()
        #    experiment.log_multiple_params(self.config)
        #    self.callbacks.append(experiment.get_keras_callback())

    def train(self):
        self.json_log = open(self.config.callbacks.tensorboard_log_dir + '/loss_log.json', mode='wt', buffering=1)
        #sorted_labels = sorted(self.generator.total_labels)
        #class_weights = class_weight.compute_class_weight('balanced', np.unique(sorted_labels), sorted_labels)
        #class_weights_list = []
        #class_weights_list.append(class_weights)
        #class_weights_list.append(class_weights)
        #print(class_weights_list)


        history = self.model.fit_generator(
            self.generator,
            epochs=self.config.trainer.num_epochs,
            steps_per_epoch=len(self.generator),
            validation_data=self.valid_generator,
            validation_steps=len(self.valid_generator),
            use_multiprocessing = True,
            max_queue_size=10,
            workers=5,
            callbacks=self.callbacks,
            #class_weight=class_weights_list,
            verbose=1)

        self.loss.extend(history.history['loss'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_out_acc_1'])



    def get_accuracy(self,epoch, isSaveSTN = False):
        # get accuracy, using default generators
        self.config['data_loader']['data_dir_train'] = self.config['data_loader']['data_dir_train_test']
        self.config['data_loader']['data_dir_valid'] = self.config['data_loader']['data_dir_valid_test']
        self.train_generator = get_testing_generator(self.config, True)
        self.valid_generator = get_testing_generator(self.config, False)
        #self.train_generator =  self.train_generator_testing
        generators = [self.train_generator, self.valid_generator]
        generators_id = ['_train', '_valid']

        for m, generator in enumerate(generators):

            # print(generators_id[m])
            # for filename in generator.filenames:
            #     print(filename)

            batch_size = self.config['data_loader']['batch_size']
            num_of_images = len(generator) * (batch_size)
            labels = np.zeros((num_of_images, 1), dtype=np.int)
            predication = np.zeros((num_of_images, int(self.config.model.embedding_dim)), dtype=np.float32)

            label_map = (generator.class_indices)
            label_map = dict((v, k) for k, v in label_map.items())  # flip k,v

            cur_ind = 0
            for k in range(len(generator)):
                if (k % 10) == 0:
                    print(k)

                x, y_true_ = generator.__getitem__(k)
                y_true = [label_map[x] for x in y_true_]
                y_true = [self.class_list.index(x) for x in y_true]
                y_pred = self.model.predict(x)

                if isSaveSTN:
                    self.save_STN_images(x,generators_id[m],cur_ind,y_pred[2], y_pred[3], y_pred[4], y_pred[5])

                if self.config.model.num_of_outputs > 1:
                    y_pred = y_pred[0]

                num_of_items = y_pred.shape[0]
                predication[cur_ind: cur_ind + num_of_items, :] = y_pred
                labels[cur_ind: cur_ind + num_of_items, :] = np.expand_dims(y_true, axis=1)
                cur_ind = cur_ind + num_of_items


            predication = predication[:cur_ind, :]
            labels = labels[:cur_ind, :]



            if m == 0:
                train_labels = labels
                train_prediction = predication
            else:
                valid_labels = labels

                valid_prediction = predication

            if self.config.callbacks.is_save_embeddings:
                lines = ''
                for k in range(labels.shape[0]):
                    lines += str(labels[k,0]) + ',' + self.class_list[labels[k,0]] + ',' + generator.filenames[k] + '\n'
                with open('evaluator/labels/' + self.config.exp.name + generators_id[m] + '.csv', "w") as text_file:
                    text_file.write(lines)
                #np.savetxt('evaluator/labels/' + self.config.exp.name + generators_id[m] + str(epoch) +  '.tsv', labels, delimiter=',')
                np.savetxt('evaluator/embeddings/' + self.config.exp.name + generators_id[m] + '.csv', predication,
                       delimiter=',')

        accuracy = eval_model(train_prediction, valid_prediction, train_labels, valid_labels, self.config.exp.name,
                              is_save_files=False)
        print('accuracy_class = {0:.3f}'.format(accuracy))
        return accuracy


    def custom_epoch_end(self,epoch,logs,type):
        if self.config.callbacks.is_run_custom_embbedings_accuracy:
            acc = self.get_accuracy(epoch)
        else:
            acc = -1

        if type == 'cosface':
            self.json_log.write(
                json.dumps(
                    {'epoch': epoch, 'loss': logs['embeddings_loss'], 'val_loss': logs['val_loss'],
                     'acc': acc,'lr': logs['lr'].astype('float64')}) + '\n')
        elif type == 'softmax':
            self.json_log.write(
                json.dumps(
                    {'epoch': epoch, 'loss': logs['out_loss'], 'val_loss': logs['val_loss'],
                     'acc_out': logs['out_acc_1'],'acc_out_val': logs['val_out_acc_1'],
                     'acc': acc ,'lr': logs['lr'].astype('float64')}) + '\n')
        elif type == 'triplet_all':
            self.json_log.write(
                json.dumps(
                    {'epoch': epoch, 'loss': logs['loss'], 'val_loss': logs['val_loss'], 'acc': acc, 'positive_fraction': logs['embeddings_positive_fraction'],
                     'val_positive_fraction': logs['val_embeddings_positive_fraction']}) + '\n')
        elif type == 'triplet_hard':
            self.json_log.write(
                json.dumps(
                    {'epoch': epoch, 'loss': logs['embeddings_loss'], 'val_loss': logs['val_embeddings_loss'], 'acc': acc, 'hard_pos_dist': logs['embeddings_hardest_pos_dist'], 'hard_neg_dist': logs['embeddings_hardest_neg_dist']}) + '\n'),


    def save_STN_images(self, orig_images,generator_type, generator_index, transform_mat,transform_mat2, stn1, stn2):
        out_folder = ''
        if generator_type == '_train':
            out_folder = 'experiments/2019-09-17/efficientNetB0_optim_stn_no_overlap_medium_mul/STN/train'
            generartor = self.train_generator
        else:
            out_folder = 'experiments/2019-09-17/efficientNetB0_optim_stn_no_overlap_medium_mul/STN/valid'
            generartor = self.valid_generator

        print(generator_index)
        for k in range(orig_images.shape[0]):
            transform = transform_mat[k]
            transform2 = transform_mat2[k]
            stn_image1 = (stn1[k] +1) * 0.5
            stn_image2 = (stn2[k] + 1) * 0.5
            file_name = generartor.filenames[generator_index + k]
            orig_image = (orig_images[k] +1)*0.5
            folder, file = file_name.split(('/'))
            print(file_name)
            #print(out_folder + '/' + folder + '/' + file + '.png')
            #print(os.path.isfile(out_folder + '/' + folder + '/' + file + '.png'))
            #if os.path.isfile(out_folder + '/' + folder + '/' + file + '.png'):
            #    continue

            f, (ax1,ax2,ax3) = plt.subplots(1, 3)
            ax1.get_yaxis().set_visible(False)
            ax1.get_xaxis().set_visible(False)
            #ax2.get_xaxis().set_visible(False)
            #ax2.get_yaxis().set_visible(False)

            ax1.imshow(orig_image)
            ax1.set_title('input')
            ax2.imshow(stn_image1)
            ax3.imshow(stn_image2)
            ax2.set_title('stn1')
            ax3.set_title('stn2')

            print(transform)
            print(transform2)
            x_offset = int(0.5*self.config.model.img_width*(transform[2]+0.5))
            y_offset = int(0.5*self.config.model.img_height*(transform[5]+0.5))
            rect = patches.Rectangle((x_offset, y_offset ), 224, 224, linewidth=1, edgecolor='r', facecolor='none')
            x_offset2 = int(0.5*self.config.model.img_width*(transform2[2]+0.5))
            y_offset2 = int(0.5*self.config.model.img_height*(transform2[5]+0.5))
            rect2 = patches.Rectangle((x_offset2, y_offset2 ), 224, 224, linewidth=1, edgecolor='g', facecolor='none')
            ax1.add_patch(rect)
            ax1.add_patch(rect2)


            if not os.path.isdir(out_folder + '/' + folder):
                os.mkdir(out_folder + '/' + folder)
            plt.savefig(out_folder + '/' + folder + '/' + file + '.png')

            #plt.savefig('.png')

            #plt.show()
            #print(file_name)
            #print(stn_image.shape)

            #np.testing.assert_equal(0,1)