import numpy as np
import os,cv2,math,random,skimage,logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf
import tensorflow.keras
import subprocess as sp
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap
from tensorflow.keras.preprocessing import image_dataset_from_directory
from skimage.metrics import structural_similarity as ssim
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class Data2():
    def flip_up_down(img):
        return tf.image.flip_up_down(img)

    def flip_left_right(img):
        return tf.image.flip_left_right(img)

    def rotate(img, angle):
        return tf.image.rot90(img, k=int(angle / 0.2) % 4)
   
    def apply_random_transformation(img):
        choice = tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)
        if choice == 0:
            return Data2.rotate(Data2.flip_up_down(Data2.flip_left_right(img)), -0.2)
        elif choice == 1:
            return Data2.rotate(img, -0.2)
        else:
            return Data2.rotate(img, 0.2)
    def load_data(train_dir=None,test_dir=None,gtruth_dir=None,augmentation_target='medium',dataset='screw',classes=None,
                  target_size = (128, 128),batch_size = 32):
        if train_dir is not None and test_dir is not None:
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            if augmentation_target=='minimal':
                train_datagen = ImageDataGenerator(rescale=1./255,
                                                   preprocessing_function=Data2.apply_random_transformation)
            if augmentation_target=='custom':
                if dataset=='screw':
                    train_datagen = ImageDataGenerator(rotation_range=40,
                                                   width_shift_range=0.05,
                                                   height_shift_range=0.05,
                                                   brightness_range=(0.8,1.2),
                                                   shear_range=0.05,
                                                   zoom_range=0.05,
                                                   fill_mode='nearest',
                                                   # horizontal_flip=True,
                                                   # vertical_flip=True,
                                                   rescale=1./255,
                                                   preprocessing_function=Data2.apply_random_transformation
                                                  )
                elif dataset=='hazelnut':
                    train_datagen = ImageDataGenerator(rotation_range=40,
                                                   width_shift_range=0.05,
                                                   height_shift_range=0.05,
                                                   brightness_range=(0.8,1.2),
                                                   shear_range=0.05,
                                                   zoom_range=0.05,
                                                   fill_mode='nearest',
                                                   horizontal_flip=True,
                                                   vertical_flip=True,
                                                   rescale=1./255,
                                                   preprocessing_function=Data2.apply_random_transformation
                                                  )
            elif augmentation_target=='full':
                train_datagen=ImageDataGenerator(featurewise_center=True,
                                                 samplewise_center=True,
                                                 featurewise_std_normalization=True,
                                                 samplewise_std_normalization=True,
                                                 zca_whitening=True,
                                                 zca_epsilon=1e-06,
                                                 rotation_range=40,
                                                 width_shift_range=0.2,
                                                 height_shift_range=0.2,
                                                 brightness_range=(0.8,1.2),
                                                 shear_range=0.2,
                                                 zoom_range=0.2,
                                                 channel_shift_range=0.0,
                                                 fill_mode='nearest',
                                                 cval=0.0,
                                                 horizontal_flip=True,
                                                 vertical_flip=True,
                                                 rescale=1./255,
                                                 preprocessing_function=Data2.apply_random_transformation,
                                                 data_format=None,
                                                 validation_split=0.1,
                                                 interpolation_order=1,
                                                 dtype=None
                                                )
            test_datagen = ImageDataGenerator(rescale=1./255)
            train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=target_size,
                batch_size=batch_size,
                class_mode=None,
                shuffle=True,
                seed=42) 
            test_generator = test_datagen.flow_from_directory(
                test_dir,
                color_mode="rgb",
                target_size=target_size,
                batch_size=batch_size,
                class_mode='categorical',
                classes=classes,
                shuffle=False,
                seed=42)
            gtruth_generator = test_datagen.flow_from_directory(
                gtruth_dir,
                color_mode="rgb",
                target_size=target_size,
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=False,    
                seed=42) 
            return train_generator,test_generator,gtruth_generator
        else:
            return None,None
class Data():
    def path_verifier(path=None):
        if path is not None:
            if not os.path.exists(path):
                os.makedirs(path)
                print(f'{path} created')
            else:
                print(f'{path} already')
            
    def get_anomaly_types(test_dir=None,gt_dir=None,dataset=None):
        if dataset is not None:
            anomaly_types = sorted(os.listdir(test_dir))
            anomaly_types_gt = sorted(os.listdir(gt_dir))
            print('*'*120)
            print(f'Classes in Test:\t{anomaly_types}\nClasses in GT:\t\t{anomaly_types_gt}')
            print('*'*120)
            anomaly_types = sorted(anomaly_types, key=Data.custom_sort)
            anomaly_types_gt = sorted(anomaly_types_gt, key=Data.custom_sort)
            print(f'Classes in Test:\t{anomaly_types}')
            return anomaly_types,anomaly_types_gt
    def custom_sort(item):
        return 0 if item == 'good' else 1
    def visualise_augmentation(generator=None,
                               batch_size=8,
                               images_per_row=4,
                               num_augmented_images_to_display=8,
                               original_image_index = 5,
                               augmentation_target=None,
                               results_path=None,
                               save_plots=False,
                               destroy_fig=False,
                               dpi=150,
                              ):
        if generator is not None:
            num_rows = int(np.ceil(batch_size / images_per_row))

            fig, axs = plt.subplots(num_rows, images_per_row, figsize=(16, num_rows * 4))
            if num_rows > 1:
                axs = axs.flatten()
            for i in range(batch_size):
                augmented_images = generator[0][original_image_index]
                row_index = i // images_per_row
                col_index = i % images_per_row
                axs[i].imshow(augmented_images)
                axs[i].axis('off')
                axs[i].set_title(f'Augmented {i+1}')
            for i in range(batch_size, num_rows * images_per_row):
                fig.delaxes(axs[i])

            plt.tight_layout(pad=0.2)
            if save_plots:
                if results_path is not None:
                    plt.savefig(f'{results_path}/Augmentation_{augmentation_target}.png',dpi=dpi)
            if destroy_fig:
                plt.close(fig)
            plt.show()
    def convert_to_float(image):
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image
    def get_subdata(DS_path=None,DS_name=None,):
        DS_name = 'screw'
        train_dir = os.path.join(DS_path,'train')
        gtruth_dir = os.path.join(DS_path,'ground_truth')
        test_dir = os.path.join(DS_path,'test')
        return os.path.abspath(train_dir),os.path.abspath(gtruth_dir),os.path.abspath(test_dir)

    def trans1(img):
        return tfa.image.rotate(tf.image.flip_left_right(tf.image.flip_up_down(img)),-.2,fill_mode="reflect",interpolation="bilinear")

    def trans2(img):
        return tfa.image.rotate(img,-.2,fill_mode="reflect",interpolation="bilinear")

    def trans3(img):
        return tfa.image.rotate(img,.2,fill_mode="reflect",interpolation="bilinear")
    
    def load_data_normal(DS_path=None,DS_name='screw',prefetch=False,augment=True,image_size = [128, 128], batch_size= None):
        tf.autograph.set_verbosity(0)
#         train_dir,validation_dir,test_dir = Data.get_subdata(DS_path=DS_path,DS_name=DS_name)

        ds = image_dataset_from_directory(
            DS_path,
            labels=None,
#             label_mode = None,
            image_size=image_size,
            interpolation='nearest',
            batch_size=batch_size,
            shuffle=True,
        )
        if augment:
            ds1,ds2,ds3,ds4 = ds,ds.map(Data.trans1),ds.map(Data.trans2),ds.map(Data.trans3)
            ds = ds1.concatenate(ds2).concatenate(ds3).concatenate(ds4)
        if prefetch:
            AUTOTUNE = tf.data.experimental.AUTOTUNE
            ds = (
                ds
                .map(Data.convert_to_float)
                .cache()
                .prefetch(buffer_size=AUTOTUNE)
            )
        return ds

    def load_data_abnormal(DS_path=None,DS_name='screw',prefetch=False,image_size=[128, 128],batch_size=None):
#         train_dir,validation_dir,test_dir = Data.get_subdata(DS_path=DS_path,DS_name=DS_name)

        ds_a = image_dataset_from_directory(
            DS_path,
#             labels=None,
            label_mode = None, # categorical
            image_size=image_size,
            interpolation='nearest',
            batch_size=batch_size,
            shuffle=False,
        )
        if prefetch:
            AUTOTUNE = tf.data.experimental.AUTOTUNE
            ds_a = (ds_a.map(Data.convert_to_float).cache().prefetch(buffer_size=AUTOTUNE))
        return ds_a
    def load_data_gtruth(DS_path=None,DS_name='screw',prefetch=False,image_size=[128, 128],batch_size=24):
        ds_gt = image_dataset_from_directory(
            DS_path,
#             labels=None,
            label_mode = None,
            color_mode = 'rgb',
            image_size=image_size,
            interpolation='nearest',
            batch_size=batch_size,
            shuffle=False,
        )
        if prefetch:
            AUTOTUNE = tf.data.experimental.AUTOTUNE
            ds_gt = (ds_gt.map(Data.convert_to_float).cache().prefetch(buffer_size=AUTOTUNE))
        return ds_gt
    def get_img_gt_by_index_class(anomaly_type=None,
                                  data_train=None,
                                  data_test=None,
                                  data_gtruth=None,
                                  dataset=None,
                                  image_no1=0,
                                  image_no2=0,
                                  convert_to_float = True
                                 ):  
        if anomaly_type is not None:
            if dataset=='screw':
                if anomaly_type=='good':
                    I_A = np.array(list(data_test[0])[image_no1][image_no2])
                    gt=np.zeros(I_A.shape)
                elif anomaly_type=='manipulated_front':
                    I_A = np.array(list(data_test[1])[image_no1][image_no2])
                    gt  = np.array(list(data_gtruth[1])[image_no1][image_no2])
                elif anomaly_type=='scratch_head':
                    I_A = np.array(list(data_test[2])[image_no1][image_no2])
                    gt  = np.array(list(data_gtruth[2])[image_no1][image_no2])
                elif anomaly_type=='scratch_neck':
                    I_A = np.array(list(data_test[3])[image_no1][image_no2])
                    gt  = np.array(list(data_gtruth[3])[image_no1][image_no2])
                elif anomaly_type=='thread_side':
                    I_A = np.array(list(data_test[4])[image_no1][image_no2])
                    gt  = np.array(list(data_gtruth[4])[image_no1][image_no2])
                elif anomaly_type=='thread_top':
                    I_A = np.array(list(data_test[5])[image_no1][image_no2])
                    gt  = np.array(list(data_gtruth[5])[image_no1][image_no2])
                return I_A.astype(np.float32)/255.0,gt
            elif dataset=='hazelnut':
                if anomaly_type=='good':
                    I_A = np.array(list(data_test[0])[image_no1][image_no2])
                    gt=np.zeros(I_A.shape)
                elif anomaly_type=='crack':
                    I_A = np.array(list(data_test[1])[image_no1][image_no2])
                    gt  = np.array(list(data_gtruth[1])[image_no1][image_no2])
                elif anomaly_type=='cut':
                    I_A = np.array(list(data_test[2])[image_no1][image_no2])
                    gt  = np.array(list(data_gtruth[2])[image_no1][image_no2])
                elif anomaly_type=='hole':
                    I_A = np.array(list(data_test[3])[image_no1][image_no2])
                    gt  = np.array(list(data_gtruth[3])[image_no1][image_no2])
                elif anomaly_type=='print':
                    I_A = np.array(list(data_test[4])[image_no1][image_no2])
                    gt  = np.array(list(data_gtruth[4])[image_no1][image_no2])
                return I_A.astype(np.float32)/255.0,gt
        else:
            return None,None
    def draw_gt_contour(grountruth_image=None, input_image=None):
        if grountruth_image is not None:
            rgb_image = input_image
            binary_mask = grountruth_image[:,:,0].astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contoured_image = rgb_image.copy()
            cv2.drawContours(contoured_image, contours, -1, (0, 255, 255), 1)  # (0, 255, 255) corresponds to yellow, 2 is the thickness
            return contoured_image
class Explanation():
    def get_CV_beta(beta):
        return np.std(beta) / np.mean(beta)
    def get_beta_from_expl(expl):
        '''
        Function get_beta_from_expl will compute beta from explanation
        Args:
            expl: Explanation returned by Strtaified Lime Image Explainer
        Result:
            beta: Local Exp for Top Label 
        '''
        n = len(np.unique(expl.segments))
        beta = np.zeros(n)
        for i,v in expl.local_exp[ expl.top_labels[0] ]:
            beta[i] = v
        return beta

    def heatmap_from_beta(segments, beta):
        heatmap = np.zeros_like(segments, dtype=np.float32)
        for segm, importance in enumerate(beta):
            heatmap[ segments==segm ] += importance 
        return heatmap
    
    
    def plot_heatmap():
        pass
class Prediction():
    def plot_latent_space(vae, n=6, figsize=8,image_size=None,
                         latent_dim=None,epochs=None,results_path=None,
                         title=None,save_all_figs = False,
                          destroy_fig=False):
        _ ,digit_size = image_size
        scale = 100
        # display a n*n 2D manifold of digits
        figure = np.zeros((digit_size * n, digit_size * n,3))
        # linearly spaced coordinates corresponding to the 2D plot of digit classes in the latent space
        grid_x = np.linspace(-scale, scale, n)
        grid_y = np.linspace(-scale, scale, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[2*random.random()-1 for i in range(latent_dim)]])

                x_decoded = vae.decoder.predict(z_sample, verbose=False)
                digit = x_decoded[0].reshape(digit_size, digit_size,3)
                figure[
                    i * digit_size : (i + 1) * digit_size,
                    j * digit_size : (j + 1) * digit_size,
                ] = digit

        fig = plt.figure(figsize=(figsize, figsize))
        start_range = digit_size // 2
        end_range = n * digit_size + start_range
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure)
        plt.tight_layout(pad=0.05)
        plt.title(f'Latent Space $Z$ for {epochs} epochs')
        if save_all_figs:
            plt.savefig(f'{results_path}/latentspace_{title}.png',dpi=200)
        if destroy_fig:
            plt.close(fig)
        plt.show()
    def plot_data_train(model=None,data=None,image_size=[128,128],fontsize=12,epochs=8,dpi=150,
                    n=8,figsize = 5,verbose=False,data_set=None,save_plot=False,results_path=None,destroy_fig=False):
        if model is not None:
            import matplotlib.gridspec as gridspec
            # On training set
            digit_size, _ = image_size
            figure = np.zeros((digit_size*3, digit_size * n,3))
            img = list(data)[0]
            fig,axs = plt.subplots(3,n,figsize=(figsize*(figsize/3.9), figsize), sharex=True, sharey=True)
            for i in range(n):
    #             ax1 = plt.subplots(3, n, i)
                _,b_img = model(img)
                a = list(b_img)[i]
            
                axs[0][i].imshow(list(img)[i], aspect=None)
                axs[0][0].set_ylabel(f'$X$ ',fontsize=fontsize)

                axs[1][i].imshow(a, aspect=None)
                axs[1][0].set_ylabel(f"$X\'$ ",fontsize=fontsize)
                diff_img = list(img)[i]-a
                diff_img = np.linalg.norm(diff_img, axis=2)
                axs[2][i].imshow(diff_img, aspect=None, cmap='inferno')
#                 print(diff_img.shape,diff_img.dtype)
                axs[2][0].set_ylabel(f"$X-X\'$ ",fontsize=fontsize)
        fig.text(0, 0.5, f'{data_set}', va='center', rotation='vertical')
        fig.tight_layout(pad=0)
        fig.subplots_adjust(wspace=0, hspace=0)
        for ax in axs.flatten():
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if save_plot:
                if results_path is not None:
                    plt.savefig(f'{results_path}/Compared_{data_set}_{epochs}_{n}.png',dpi=dpi)
        if destroy_fig:
            plt.close(fig)   
        plt.show()
class Segmentation():

    def explore_segmentation_types(input_image=None,
                                   reconstructed_img=None,
                                   seg_type = ['slic','quickshift','felzenszwalb'],
                                  color1=(1, 0, 0)):
        if input_image is not None:
            seg_type = ['slic','quickshift','felzenszwalb']
            fig,axs = plt.subplots(len(seg_type),4, figsize=(10,10), sharex=True, sharey=True)

            for st_id,st in enumerate(seg_type):

                if st=='quickshift':
                    segmentation_fn = SegmentationAlgorithm(st, kernel_size=1, 
                                                            max_dist=2, ratio=0.1, random_seed=1234) 
                elif st=='slic':
                    segmentation_fn = SegmentationAlgorithm(st,compactness=50,max_num_iter=5, 
                                                     ratio=0.5,random_seed=1234)
                elif st=='felzenszwalb':
                    segmentation_fn = SegmentationAlgorithm(st, kernel_size=4, 
                                                            max_dist=2, ratio=0.1, random_seed=1234) 

                def segments_getter(img):
                    return segments

                axs[st_id][0].imshow(input_image)
            #     axs[st_id][0].
                segments = segmentation_fn(input_image)
                # print(len(np.unique(segments)))
                segs = np.unique(segments).shape[0]
                immgg=skimage.segmentation.mark_boundaries(input_image, segments, 
                                                           color=color1, outline_color=None, 
                                                           mode='outer', background_label=0)
                axs[st_id][1].imshow(immgg)
                axs[st_id][1].set_title(f'$I\_A$ \n$k={len(np.unique(segments))})$')

                segments = segmentation_fn(reconstructed_img)
                # print(len(np.unique(segments)))
                segs = np.unique(segments).shape[0]
                immgg=skimage.segmentation.mark_boundaries(input_image, segments, 
                                                           color=color1, outline_color=None, 
                                                           mode='outer', background_label=0)
                axs[st_id][2].imshow(immgg)
                axs[st_id][2].set_title(f'$R\_A$ \n$k={len(np.unique(segments))}$')

                segments = segmentation_fn(input_image*0.5+reconstructed_img*0.5)
                # print(len(np.unique(segments)))
                segs = np.unique(segments).shape[0]
                immgg=skimage.segmentation.mark_boundaries(input_image, segments, 
                                                           color=color1, outline_color=None, 
                                                           mode='outer', background_label=0)
                axs[st_id][3].imshow(immgg)
                axs[st_id][3].set_title(f'$I_A*0.5+R_A*0.5$ \nk=${len(np.unique(segments))}$')
            # set labels
            for st_id,st in enumerate(seg_type):
                axs[st_id][0].set_ylabel(st, fontsize = 16)

            for ax in axs:
                for aa in ax:       
                    aa.set_xticks([])
                    aa.set_yticks([])
            plt.tight_layout(pad=0.05)
    def get_segmentation(input_image=None,
                         reconstructed_img=None,
                         mask_type='blend',
                         seg_type=None):
        if seg_type=='quickshift':
            
            
#             segments = quickshift(input_image, kernel_size=2, 
#                       max_dist=8, ratio=0.99, random_seed=1234, sigma=0.25) 
            
            segmentation_fn = SegmentationAlgorithm(seg_type, kernel_size=2,
                                                    max_dist=8, ratio=0.99, random_seed=1234, sigma=0.25) 
        elif seg_type=='slic':
            segmentation_fn = SegmentationAlgorithm(seg_type,compactness=50,max_num_iter=5, 
                                                     ratio=0.5,random_seed=1234)
        elif seg_type=='felzenszwalb':
            segmentation_fn = SegmentationAlgorithm(seg_type, kernel_size=4, 
                                                            max_dist=2, ratio=0.1, random_seed=1234) 

        def segments_getter(img):
            return segments
        if mask_type=='input':
            segments = segmentation_fn(input_image)   
        elif mask_type=='reconstructed':
            segments = segmentation_fn(reconstructed_img)
        elif mask_type=='blend':
            segments = segmentation_fn(input_image*0.5+reconstructed_img*0.5)
        segs = np.unique(segments).shape[0]
#         print(f'using seg_type:\t{seg_type}, \tmask_type:\t{mask_type} \t== \t{segs}')
        return segments,segs,segments_getter

class Evaluate:
    def find_optimal_separation_threshold(anomaly_scores):
        # print(anomaly_scores)
        def score_threshold(anomaly_scores, delta):
            y_true = [ a[1] for a in anomaly_scores ]
            y_pred = [ a[0] > delta for a in anomaly_scores ]
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
            return math.sqrt( tpr * (1 - fpr) )
    
        deltas = [ a[0] for a in anomaly_scores ]
        opt_i = np.argmax([ score_threshold(anomaly_scores, delta) for delta in deltas ])
        print(opt_i, anomaly_scores[opt_i])
        return anomaly_scores[opt_i][0] + 0.00001
    
    
    def computer_anomaly_score(input_image=None,reconstructed_image=None):
        return np.linalg.norm(input_image - reconstructed_image, axis=2)

class visualize():
    def calc_IoU_curve(y_true, y_pred):
        yd = np.array(sorted(zip(y_pred, y_true), reverse=True))
        X2, IoU2, Th = [], [], []
        nT = np.sum(y_true)
        nInt = 0
        i = 0
        while i < len(y_pred):
            start = i
            while i < len(y_pred) and yd[start,0] == yd[i,0]:
                if yd[i,1]: nInt += 1
                i += 1
            
            IoU2.append(nInt / (i + nT - nInt))
            X2.append(i)
            Th.append(yd[start,0])

        IoU2 = np.array(IoU2)
        X2 = np.array(X2)
        return X2, IoU2, Th[np.argmax(IoU2)]  # X, Y for IoU curve, threshold

    def combine_groundtruth_explanation(gtruth, heatmap, threshold):
        gt = gtruth[:,:,0]>0
        ht = (heatmap >= threshold).astype(np.uint8)
        img = np.zeros(shape=list(heatmap.shape)+[3], dtype=np.uint8)
        img[:,:,0] = 255*(1-gt)
        img[:,:,1] = 255*(1-ht)
        img[:,:,2] = 255*(1-ht)
        return img

    def get_axis_limits(axs=None):
        for ax_id,ax in enumerate(axs):
            print(f'{ax_id} -> {ax.get_xlim()} , {ax.get_ylim()}')
    def get_training_curves(history_frame=None,
                            metric=None,
                            save_all_figs=False,
                           results_path=None,
                            title=None,
                            destroy_fig=False,
                           ):
        fig, ax = plt.subplots(figsize=(10,4))
        if history_frame is not None:
            history_frame.loc[:, [metric]].plot(ax=ax)
            plt.xlabel('Epochs')
            plt.tight_layout(pad=0.05)
        if save_all_figs:
            plt.savefig(f'{results_path}/traincurve_{title}_{metric}.png',dpi=200)
        
        if destroy_fig:
            plt.close(fig)  
        plt.show()
    def train_pca_on_latent_space(data=None,n_components=2):
        if data is not None:
            pca = PCA(n_components=2)
            pca.fit(data)
            data_transformed = pca.transform(data)
            return pca,data_transformed
    def plot_latent_space_normal(model=None,
                                 data=None,
                                 epochs=None,
                                 save_plot=False,
                                 results_path=None,
                                 title=None,
                                 destroy_fig=False,
                                ):   
        fig = plt.figure(figsize=(6, 4))
        z_mean,z_log_var, _ = model.predict(data, batch_size=4)
        y_train = data.labels
        pca,z_mean_transformed = visualize.train_pca_on_latent_space(data=z_mean,
                                                     n_components=2)
        fig = plt.figure(figsize=(6, 4))
        plt.scatter(z_mean_transformed[:, 0], z_mean_transformed[:, 1], c='black')
# #         plt.colorbar()
# #         plt.set_cmap('tab10')
#         plt.xlabel("z[0]")
#         plt.ylabel("z[1]")
#         plt.title('Latent Space for Train Data')
        if save_plot:
            if results_path is not None:
                plt.savefig(f'{results_path}/latentspace_train_{epochs}.png',dpi=200)
        if destroy_fig:
            plt.close(fig)
        plt.show()
        return pca
    def plot_latent_space_all(model=None,
                              data_train=None,
                              data_test=None,
                                 epochs=None,
                                 save_plot=False,
                                 results_path=None,
                              title=None,
                              destroy_fig=False,
                              batch_size=4,
                                ):
        
        y_test = data_test.labels
        z_mean_test,z_log_var, _ = model.predict(data_test, batch_size=batch_size)
        
        z_mean_train,z_log_var, _ = model.predict(data_train, batch_size=batch_size)
        
        fig = plt.figure(figsize=(6, 4))
        pca,z_mean_train = visualize.train_pca_on_latent_space(data=z_mean_train,
                                                         n_components=2)

        z_mean_test = pca.transform(z_mean_test)

        plt.scatter(z_mean_train[:, 0], z_mean_train[:, 1], c='black')
        plt.scatter(z_mean_test[:, 0], z_mean_test[:, 1], c=y_test)
        plt.colorbar()
        plt.set_cmap('tab10')
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.title('Latent Space for All Data')
        if save_plot:
            if results_path is not None:
                plt.savefig(f'{results_path}/latentspace_all_{epochs}.png',dpi=200)
        if destroy_fig:
            print('destroy_fig')
            plt.close(fig)
        plt.show()     
def compare(data=None,input_type=None,save_plot=False,results_path=None,verbose=False):
    if data is not None:
        global model
        I_A = np.array(list(data)[0][0])
        R_A = model.predict(np.array([I_A]), verbose=verbose)

        preds = model.evaluate(np.array([I_A]) )
        pred = lime_predicter_max(np.array([I_A]), verbose=verbose)
#         print(preds)
        fig,axs = plt.subplots(1,2)
        axs[0].imshow(I_A),axs[0].set_title(f'Input ({input_type})')
        axs[1].imshow(R_A[1][0]),axs[1].set_title('Reconstructed')
        fig.suptitle(f' {pred[0][0]:0.4} ')

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout(pad=0.05)
        if save_plot:
            if results_path is not None:
                plt.savefig(f'{results_path}/ComparedwithScore_{pred[0][0]:0.4}.png',dpi=200)

    if perform_single_test_compare:
        compare(data=ds, input_type='Normal', save_plot=True,
                results_path=results_path, verbose=False)
        compare(data=ds_a1,input_type='Anamolous_scratch_neck', save_plot=True,
                results_path=results_path, verbose=False)
        compare(data=ds_a2,input_type='Anamolous_manipulated_front', save_plot=True,
                results_path=results_path, verbose=False)
        compare(data=ds_a3,input_type='Anamolous_scratch_neck', save_plot=True,
                results_path=results_path, verbose=False)
        compare(data=ds_a4,input_type='Anamolous_thread_side', save_plot=True,
                results_path=results_path, verbose=False)
        compare(data=ds_a5,input_type='Anamolous_thread_top', save_plot=True,
                results_path=results_path, verbose=False)
class Resources():
    def get_gpu_memory():
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        return memory_free_values[0]
class get_losses():
        # Score Fucntions by ELVIO
    def mean_predict(model, data):
        z_mean, z_log_var, z = model.vae.encoder(data)
        reconstruction = model.vae.decoder(z_mean)
        return reconstruction
    def z_predict(model, data):
        z_mean, z_log_var, z = model.vae.encoder(data)
        reconstruction = model.vae.decoder(z)
        return reconstruction
    
    def lime_predict_2norm(data, verbose=False):
        global model
        rec = get_losses.mean_predict(model, data)
        diff = (rec - data)
        n = tf.norm(tf.norm(diff, axis=-1), axis=[-1, -2])
        return np.array([[x] for x in n])
    
    def lime_predict_latent(images, verbose=False):
        global model
        rec = model.vae.encoder(images)[2]
        n = [[np.linalg.norm(x - zRA)] for x in rec]
        return np.array(n)
        
    def lime_predict_am_sum(images, verbose=False):
        # global model
        # rec = model.predict(images, verbose=verbose)
        # diff = (rec[1] - images)
        # diffn2 = np.linalg.norm(diff, axis=3)
        # # print(diff.shape, diffn2.shape)
        # n = np.array([[np.sum(x)] for x in diffn2])
        # # n = [[np.linalg.norm(x)] for x in diff]
        # return np.array(n)
        # # global IN, RA
        return np.array([[np.linalg.norm(x)] for x in images])
    
    import tensorflow.keras as keras
    from keras import backend as K
    
    # From "General Frameworks for Anomaly Detection Explainability: A comparative study"
    def lime_predict_loss(data, verbose=False):
        global model
        batch_size = K.shape(data)[0]
        reconstruction = get_losses.mean_predict(model, data)
        MSE_loss = tf.reduce_sum(tf.square(reconstruction - data), axis=[1,2,3])
        return np.array([ [l] for l in MSE_loss ])
    
    def lime_predict_max2norm(data, verbose=False):
        global model
        rec = get_losses.mean_predict(model, data)
        diff = (rec - data)
        n = tf.norm(tf.reduce_max(diff, axis=-1), axis=[-1, -2])
        return np.array([[x] for x in n])
    
    def lime_predict_maxloss(data, verbose=False):
        global model
        batch_size = K.shape(data)[0]
        reconstruction = get_losses.mean_predict(model, data)
        MSE_maxloss = tf.reduce_sum(tf.reduce_max(tf.square(reconstruction - data), axis=[3]), axis=[1,2])
        return np.array([ [l] for l in MSE_maxloss ])
    
    def get_beta_from_expl(expl):
        n = len(np.unique(expl.segments))
        beta = np.zeros(n)
        for i,v in expl.local_exp[ expl.top_labels[0] ]:
            beta[i] = v
        return beta
    
    def heatmap_from_beta(segments, beta):
        heatmapmap = np.zeros_like(segments, dtype=np.float32)
        for segm, importance in enumerate(beta):
            heatmapmap[ segments==segm ] += importance 
        return heatmapmap
    
    import tensorflow_addons as tfa
    
    def lime_predict_maxloss_denoised(data, verbose=False):
        global model
        batch_size = K.shape(data)[0]
        reconstruction = get_losses.mean_predict(model, data)
        
        difference = tf.cast(tf.square(reconstruction - data)*256, tf.int32)
        denoised = tfa.image.gaussian_filter2d(difference, sigma=1, filter_shape=5)
        clipped = tf.cast(tf.clip_by_value(difference - denoised, 5, 10000), tf.float32)
        MSE_maxloss = tf.reduce_sum(tf.reduce_max(clipped, axis=[3]), axis=[1,2]) / 256
        return np.array([ [l] for l in MSE_maxloss ])
    
    from skimage.metrics import structural_similarity
    
    def lime_ssim_loss(data, verbose=False):
        global model
        reconstruction = get_losses.mean_predict(model, data)
        ssim_scores = []
        
        for i in range(data.shape[0]):
            ssim = structural_similarity(np.linalg.norm(data[i], axis=2), 
                                             np.linalg.norm(reconstruction[i], axis=2),
                                             data_range=1, win_size=5)#, full=True)
            # vmax = np.max(np.abs(img))
            ssim_scores.append([ssim])
            
        return np.array(ssim_scores)
    ########################## LOSSES ######################
    
    def lime_predict_maxloss_denoised(data, verbose=False):
        global model
        batch_size = K.shape(data)[0]
        # z_mean, z_log_var, z = model.vae.encoder(data)
        # reconstruction = model.vae.decoder(z)
        rec = mean_predict(model, data)
        difference = tf.cast(tf.square(rec - data)*256, tf.int32)
        denoised = tfa.image.gaussian_filter2d(difference, sigma=1, filter_shape=5)
        clipped = tf.cast(tf.clip_by_value(difference - denoised, 5, 10000), tf.float32)
        MSE_maxloss = tf.reduce_sum(tf.reduce_max(clipped, axis=[3]), axis=[1,2]) / 256
        return np.array([ [l] for l in MSE_maxloss ])  
    
    def lime_predict_2norm(data, verbose=False):
        global model
        rec = mean_predict(model, data)
        diff = (rec[1] - data)
        n = tf.norm(tf.norm(diff, axis=-1), axis=[-1, -2])
        return np.array([[x] for x in n])
    def lime_ssim_loss(data, verbose=False):
        global model
        # z_mean, z_log_var, z = model.vae.encoder(data)
        # reconstruction = model.vae.decoder(z)
        rec = mean_predict(model, data)
        anomaly_scores = []
        # fig,axs = plt.subplots(1,3, figsize=(5,10))
        for img_id in range(data.shape[0]):
            img_ = cv2.cvtColor(data[img_id], cv2.COLOR_BGR2GRAY)
            reconstruction_ = np.squeeze(rec[img_id])
            reconstruction_ = cv2.cvtColor(reconstruction_, cv2.COLOR_BGR2GRAY)
            ssim_value,sim_img = ssim(img_, reconstruction_,
                              data_range=np.max(reconstruction_) - np.min(reconstruction_),
                             full=True)
            anomaly_scores.append(ssim_value)
        return np.array(np.expand_dims(anomaly_scores, axis = 1))