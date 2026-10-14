"""MVTec loaders: evaluation and paper-compatible ImageDataGenerator."""
from pathlib import Path
import numpy as np
from PIL import Image

VALID_EXTENSIONS={".png",".jpg",".jpeg",".bmp"}
def image_files(folder):
    folder=Path(folder)
    if not folder.is_dir(): raise FileNotFoundError(folder)
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS)
def load_rgb(path,size=128):
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB").resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32)/255.0
def load_mask(path,size=128):
    with Image.open(path) as img:
        return np.asarray(img.convert("L").resize((size,size),Image.Resampling.NEAREST),dtype=np.uint8)>0
def training_paths(root,category):
    paths=image_files(Path(root)/category/"train"/"good")
    if not paths: raise ValueError("No training images")
    return paths
def evaluation_samples(root,category):
    folder=Path(root)/category/"test"
    if not folder.is_dir(): raise FileNotFoundError(folder)
    for kind in sorted(p for p in folder.iterdir() if p.is_dir()):
        for path in image_files(kind):
            defect=kind.name!="good"
            mask=Path(root)/category/"ground_truth"/kind.name/(path.stem+"_mask.png")
            if defect and not mask.exists(): raise FileNotFoundError(mask)
            yield path,int(defect),mask if defect else None

def make_train_dataset(root,category,image_size,batch_size,seed=42,shuffle=True):
    import tensorflow as tf
    ds=tf.data.Dataset.from_tensor_slices([str(p) for p in training_paths(root,category)])
    if shuffle: ds=ds.shuffle(len(training_paths(root,category)),seed=seed,reshuffle_each_iteration=True)
    def decode(path):
        x=tf.io.decode_image(tf.io.read_file(path),channels=3,expand_animations=False)
        x.set_shape([None,None,3])
        return tf.cast(tf.image.resize(x,(image_size,image_size)),tf.float32)/255.
    return ds.map(decode,num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def paper_rotation(img):
    """Exact Data2 preprocessing: random 0/1/2 -> k=3/3/1 quarter turns."""
    import tensorflow as tf
    choice=int(tf.random.uniform([],minval=0,maxval=3,dtype=tf.int32).numpy())
    if choice==0:
        return tf.image.rot90(tf.image.flip_up_down(tf.image.flip_left_right(img)),k=3).numpy()
    if choice==1:
        return tf.image.rot90(img,k=3).numpy()
    return tf.image.rot90(img,k=1).numpy()

def make_paper_train_generator(root,category,image_size=128,batch_size=32,seed=42):
    """Port Data2.load_data(..., augmentation_target='custom') from paper."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    if category not in ("screw","hazelnut"):
        raise ValueError("Paper augmentation only defined for screw and hazelnut")
    aug=dict(rotation_range=40,width_shift_range=0.05,height_shift_range=0.05,
             brightness_range=(0.8,1.2),shear_range=0.05,zoom_range=0.05,
             fill_mode="nearest",rescale=1./255,
             preprocessing_function=paper_rotation)
    if category=="hazelnut":
        aug.update(horizontal_flip=True,vertical_flip=True)
    train_dir=Path(root)/category/"train"
    if not train_dir.is_dir(): raise FileNotFoundError(train_dir)
    return ImageDataGenerator(**aug).flow_from_directory(
        str(train_dir),target_size=(image_size,image_size),
        batch_size=batch_size,class_mode=None,shuffle=True,seed=seed)
