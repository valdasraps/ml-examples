from read import ReadData, CubeLoader
from sklearn.pipeline import Pipeline
from median import Picker
from lib.cubes import Cubes
from scalers import Gradient, Standard
from cv_runner import CVRunner
from process import PrepData, PrepModel, Downsample, PCARunner, Cutter, TrainTestSelector
from classifier import Classifier, Creator, KNNClassifier, XGBoostClassifier #, TpotClassifier
from results import Collector, Writer, Pickle
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D, LeakyReLU, ELU, ReLU
import keras.optimizers as opt
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard
import os
from persist import Persistance
from model_create import ModelCreate
from model_runner import ModelRunner
from filters import SavgolFilter, NdviFilter, McariFilter, SoilFilter, AMFilter, LinearTransform
from features import FeatureSelect

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt

import numpy as np
import hashlib

from lib.pipeline2dot import pipeline2dot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Iš kur imam pradinius duomenis (meta ir bazinį neapdorotą datasetą)
folder = '/data/hypersp/datasets/Kvieciai'
result_folder = f'{folder}/RawSveikiAugalai/'
data_folder = f'{folder}'
Augalas = 'žieminiai kviečiai'

dr = ReadData(Augalas, data_folder, data_folder, metadata_filename='raw_data.csv')

# užkrauna duomenis į data masyvą (read_cube_data = False neskaito iš tikro duomenų)
data = dr.load(read_cube_data = False)

l = []
for a in np.unique(data[1].meta[:,[0,1,4]], axis = 0):
    l.append(' '.join(a[a != '']))
group_names = np.array(l)

nn_input = 192
nn_output = 2
elu_alpha = 0.2

model = Sequential()

model.add(Dense(nn_input, input_dim = nn_input, name = 'dense1', 
                kernel_regularizer = regularizers.l1_l2(l1 = 1e-5, l2 = 1e-4),
                bias_regularizer = regularizers.l2(1e-4),
                activity_regularizer = regularizers.l2(1e-5),
                kernel_initializer = 'he_normal'))
model.add(BatchNormalization())
#model.add(ELU(alpha = elu_alpha))
model.add(ReLU())
model.add(Dropout(0.5))

model.add(Dense(nn_input // 2, name = 'dense2', 
                kernel_regularizer = regularizers.l1_l2(l1 = 1e-5, l2 = 1e-4),
                bias_regularizer = regularizers.l2(1e-4),
                activity_regularizer = regularizers.l2(1e-5),
                kernel_initializer='he_normal'))
model.add(BatchNormalization())
#model.add(ELU(alpha = elu_alpha))
model.add(ReLU())
model.add(Dropout(0.5))

model.add(Dense(nn_input // 4, name = 'dense3', 
                kernel_regularizer = regularizers.l1_l2(l1 = 1e-5, l2 = 1e-4),
                bias_regularizer = regularizers.l2(1e-4),
                activity_regularizer = regularizers.l2(1e-5),
                kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(ELU(alpha = elu_alpha))
#model.add(ReLU())
model.add(Dropout(0.5))

# model.add(Dense(nn_input // 4, name = 'dense4', kernel_initializer='he_normal'))
# model.add(BatchNormalization())
# model.add(ELU(alpha = elu_alpha))
# model.add(Dropout(0.5))

# model.add(Dense(nn_input // 5, name = 'dense5', kernel_initializer='he_normal'))
# model.add(BatchNormalization())
# model.add(ELU(alpha = elu_alpha))
# model.add(Dropout(0.5))

model.add(Dense(nn_output, activation = 'softmax', name = 'softmax'))

model.summary()

epochs = 5
learning_rate = 0.001
decay_rate = 1e-6
momentum_rate = 0.9

optim = opt.SGD(lr = learning_rate, decay = decay_rate, momentum = momentum_rate, nesterov = False)
model.compile(loss = 'categorical_crossentropy',
              optimizer = optim, metrics = ['accuracy'])

es = EarlyStopping(monitor='loss', min_delta = 0, patience = 10, verbose = 0, restore_best_weights = True)
tb = TensorBoard(log_dir = 'logs', histogram_freq = 1)

# Kur deti rezultatus
result_out_folder = '/data/hypersp/datasets/Kvieciai/Rezultatai'
res_folder = f'{result_out_folder}/RawNNv4'
wavelength = Cubes().wavelength

# Picker skirtas sukurti apdorotam datasetui, bet jis jau yra tai tiesiog nuskaitys i atminti
picker = Picker(result_folder, n_jobs=1, overwrite_files = False, spec_take = 1, spec_number = 1000,
                result_file_name = 'raw-data-1000.npy', iterations = 1000*2, add_soil = False)

# sukuriam CV runner su nuskaitytais duomenim
cvrunner = CVRunner(data[3], data[1], data[2],
                    result_folder=os.path.join(result_folder, res_folder), n_jobs = 1)

# skirtas CV duomenu paruosimui. keep_elems nurodo kokius elementus tirsim. 
# conv reikia jei konvoliucinis tinklas naudojamas
cvrunner.add_step('dataprep', PrepData(data[1], data[2], job_type='fields', sampler = 'under',
                                       transform_type = 'ohe', Augalas = Augalas, keep_elems=['N'], conv=False))

# sukuriam klasifikatoriu pagal model ir earlyStopping (es), 
# history_folder saugos training'o istorija, jei nereikia none palikt
cvrunner.add_step('classify', Classifier(model, None, epochs = epochs, verbose=1, 
                                         history_folder = f'{result_out_folder}/RawNNv4_hist'))

# surenka klasifikavimo rezultata
cvrunner.add_step('collect', Collector(result_folder=res_folder))

# gauta rezultata i csv iraso
cvrunner.add_step('write', Writer(data[1], add_metrics_to_csv=True,
                                  result_folder=res_folder,
                                  job_type='fields', Augalas = Augalas))

union = Pipeline([('pikcer', picker), # nuskaito dataseta
                  ('cv', cvrunner) # leidzia CV pagal grupes arba laukus
                 ])

X = union.transform(data[0])
