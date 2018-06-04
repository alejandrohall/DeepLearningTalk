from deeplearning_toolbox import DeepLearningModel
from keras.applications.inception_v3 import InceptionV3

if __name__ == '__main__':
    base_model = InceptionV3()

    dl_model = DeepLearningModel(base_model,
                                 '/media/alejandrohall/41241e44-b9e9-4349-9bf2-49a6f4ed7c6d/skin_cancer_binary/sample/')

    dl_model.finetune(input_shape=(500, 500), full_train=True)

    dl_model.compile()
    dl_model.train(batch_size=10, early_stopping=True, epochs=1)
