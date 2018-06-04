import itertools
from keras.models import clone_model


def __generate_cartesian_product(parameters):
    keys = list(parameters.keys())
    values = [parameters[key] for key in keys]

    all_parameters = list()
    for element in itertools.product(*values):
        parameters_n = dict()
        for index in range(len(keys)):
            parameters_n[keys[index]] = element[index]

        all_parameters.append(parameters_n)

    return all_parameters


def grid_search(dl_model, parameters):

    all_parameters = list()

    if True in parameters['full_train'] and False in parameters['full_train'] and len(parameters['layers_to_finetune']) >= 1:

        # Generate cartesian product for non full training (matter num of layers)
        parameters['full_train'].remove(True)
        all_parameters.append(*__generate_cartesian_product(parameters))

        # Generate cartesian product for full training (doesn't matter num of layers)
        parameters['full_train'].append(True)
        parameters['full_train'].remove(False)
        parameters['layers_to_finetune'] = [parameters['layers_to_finetune'][0]]
        all_parameters.append(*__generate_cartesian_product(parameters))

    elif True in parameters['full_train'] and False not in parameters['full_train']:
        parameters['layers_to_finetune'] = [parameters['layers_to_finetune'][0]]
        all_parameters = __generate_cartesian_product(parameters)

    elif True not in parameters['full_train'] and False in parameters['full_train']:
        all_parameters = __generate_cartesian_product(parameters)

    clean_model = dl_model.model

    scores = list()
    for parameters_set in all_parameters:
        dl_model.model = clone_model(clean_model)

        dl_model.model.set_weights(clean_model.get_weights())

        dl_model.finetune(input_shape=parameters_set['input_shape'], full_train=parameters_set['full_train'],
                          layers_to_finetune=parameters_set['layers_to_finetune'])

        dl_model.compile(optimizer=parameters_set['optimizer'])

        print('\nTraining with {}'.format(parameters_set))
        score = dl_model.train(epochs=parameters_set['epochs'], batch_size=parameters_set['batch_size'])

        scores.append(score[0])

    best_parameters = all_parameters[scores.index(min(scores))]

    print('\n'+str(all_parameters))
    print(scores)

    print('The best parameters are {}'.format(best_parameters))
