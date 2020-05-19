from hyperopt import hp, tpe, Trials, fmin, STATUS_OK
import numpy as np
from hyperopt_eval_functions import run_hyp_model, generate_poisoned_filepaths, calc_loss
import csv
import time
import pickle

def eval_model(params, drawplots=0, model=None): # modified version of eval_model used for hyperopt
    np.random.seed(123)
    model_name = str(params['model_name'])
    model_subname = str(params['model_subname'])
    dimres_method = str(params['dimres_method'])
    model_epochs = int(params['model_epochs'])
    model_dim_size = int(params['model_dim_size'])
    db_params = [params['min_samples'], params['min_cluster_size']]
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
    writer.writerow([loss, params, ITERATION, run_time, max(new_clustered_data[:,-1])])

    global tpe_trials
    pickle.dump(tpe_trials, open("tpe_trials.p", "wb"))

    return {'loss': loss, 'params': params, 'iteration': ITERATION,
            'train_time': run_time, 'status': STATUS_OK}  #, new_clustered_data, new_fp_df, model




space = {
    'model_name': hp.choice('model_name', ['Doc2Vec']),
    'model_subname': hp.choice('model_subname', ['SG']),
    'dimres_method': hp.choice('dimres_method', ["pca"]),
    'model_epochs': 180, #hp.quniform('model_epochs', 80, 200, 10),
    'model_dim_size': hp.quniform('model_dim_size', 80, 400, 10),
    #'db_eps': hp.quniform('db_eps', 0.2, 0.4, 0.05)
    'min_cluster_size': 3, #hp.quniform('min_cluster_size', 2, 7, 1),
    'min_samples': 3 #hp.quniform('min_samples', 2,4,1)
}


tpe_trials = Trials()
# use this line instead if you're continuing
# tpe_trials = pickle.load(open("tpe_trials.p", "rb"))

ITERATION = 0

out_file = "eval_trials.csv"
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'train_time'])
of_connection.close()


# Run evals with the tpe algorithm
tpe_best = fmin(fn=eval_model, space=space,
                algo=tpe.suggest, trials=tpe_trials,
                max_evals=100, rstate=np.random.RandomState(50))

print(tpe_best)