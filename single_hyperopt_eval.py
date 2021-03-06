from hyperopt import hp, tpe, Trials, fmin, STATUS_OK
import numpy as np
from hyperopt_eval_functions import run_hyp_model, generate_poisoned_filepaths, calc_loss
import csv
import time
import pickle

def eval_model(params, drawplots=1, model=None): # modified version of eval_model used for hyperopt
    np.random.seed(123)
    model_name = str(params['model_name'])
    model_subname = str(params['model_subname'])
    dimres_method = str(params['dimres_method'])
    model_epochs = int(params['model_epochs'])
    model_dim_size = int(params['model_dim_size'])
    db_params = [params['min_samples'], params['min_samples']]
    #db_params = [params['db_eps'], 1]
    time_start = time.time()

    global ITERATION
    ITERATION += 1

    print("\nFinding original clusters...\n")
    if model is None:
        model_flag = 0
    else:
        pass
    clustered_data, model, original_filepaths, original_encoded_filepaths = run_hyp_model(model_name=model_name, model_subname=model_subname, model_epochs=model_epochs, dimres_method=dimres_method,
                                                                                          model_dim_size=model_dim_size, db_params=db_params, drawplots=drawplots, pretrained_model=model)
    if model_flag == 0:
        print("\nTime taken to build new model and find clusters: {} seconds".format(time.time()-time_start))
    else:
        print("\nTime taken to and find clusters from existing model: {} seconds".format(time.time()-time_start))

    print("\nPoisoning data...")
    new_fp_df = generate_poisoned_filepaths(filepaths=original_filepaths, clustered_data=clustered_data)

    # for filepath in new_fp_df["new filepath"]:
    #     original_filepaths.append(filepath)
    #
    # new_filepaths = original_filepaths

    time_start_2 = time.time()
    print("\nFinding new clusters...\n")
    new_clustered_data, model, combined_filepaths, new_encoded_filepaths = run_hyp_model(model_name=model_name, model_subname=model_subname, model_epochs=model_epochs, pretrained_model=model,
                                                                                         model_dim_size=model_dim_size, filepaths=original_filepaths, encoded_fps=original_encoded_filepaths,
                                                                                         new_filepaths=new_fp_df["new filepath"].tolist(), dimres_method=dimres_method,
                                                                                         db_params=db_params, drawplots=drawplots)
    print("\nTime taken to find new filepath clusters using existing model: {} seconds".format(time.time() - time_start_2))

    print("\nCalculating loss rate...")
    loss = calc_loss(new_fp_df, new_clustered_data, combined_filepaths)

    print(loss, "%")

    run_time = (time.time() - time_start)/60

    print("\nTotal Time Elapsed with this evaluation: {} minutes".format(run_time))

    # avoid the trivial solution
    if max(new_clustered_data[:, -1]) < 50:
        loss = 999
    else:
        pass

    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, ITERATION, run_time])


    return {'loss': loss, 'params': params, 'iteration': ITERATION,
            'train_time': run_time, 'status': STATUS_OK}  #, new_clustered_data, new_fp_df, model
valid_trials = Trials()
ITERATION = 0
out_file = "eval_trials.csv"
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'train_time'])
of_connection.close()
space = {
    'model_name': "Doc2Vec",
    'dimres_method': "umap",
    'model_subname': 'SG',
    'model_epochs': 180,
    'model_dim_size': 160,
    # 'db_eps': 0.4,
    'min_samples': 3
}
eval_model(params=space)