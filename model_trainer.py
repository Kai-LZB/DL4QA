#! -*- coding:utf-8 -*-
'''
@author: Kai Zheng
'''
from data_util import word_seg, word_to_syn, calc_idf, gen_word_vec, gen_input
#from keras.models import load_model
import numpy as np
import sys
import layers
import config

def train_model(w2v_file, syntrans_file, idf_file, model):
    
    """# Gernerate input for NN
    g = gen_input(w2v_file, syntrans_file, idf_file) # Generator, memory friendly
    q_sample, a_sample, idf_sample, Y_sample = g.next() # first sample is empty"""
    
    f = open("history.log", 'a')
    # Model fitting
    for i in range(config.TrainConfig.TRAIN_EPOCHS): # each epoch goes through one round
        g = gen_input(w2v_file, syntrans_file, idf_file) # Generator, memory friendly
        q_sample, a_sample, idf_sample, Y_sample = next(g) # first sample is empty
        cnt = 0
        while(True):
            cnt += 1
            #print(cnt)
            try:
                q_sample, a_sample, idf_sample, Y_sample = next(g)
            except StopIteration:
                break

            # if cnt in (1808, 5531):
            #     continue

            if np.array([a_sample]).shape[2] <= 1:
                print(cnt)
                f.write('Skipped sample set No. %d for stability.\n' % cnt)
                continue

            X = [np.array([q_sample]), np.array([a_sample]), np.array([idf_sample])] # batch is the exact input; must be a list, not a tuple
            Y = np.array([Y_sample])
            try:
                history = model.fit(X, Y, batch_size=1)

            except Exception as e:
                f.write('Error occurred in sample No. %d:\n' % cnt)
                f.write(str(e))
                f.write('\n')
        if i % 1 == 0:
            my_model.save_weights("my_model_weights.h5")
            
            
    my_model.save_weights("my_model_weights.h5")
    
    f.close()
    
def evaluate_model(w2v_file, syntrans_file, idf_file, score_file, model):
    g = gen_input(w2v_file, syntrans_file, idf_file)
    sc_text = open(score_file, 'w')
    cnt = 0
    while(True):
        cnt += 1
        try:
            q_sample, a_sample, idf_sample, Y_sample = next(g)
        except StopIteration:
            break
        X = [np.array([q_sample]), np.array([a_sample]), np.array([idf_sample])] # batch is the exact input; must be a list, not a tuple
        #Y = np.array([Y_list[j]])
        try:
            score_all = model.predict(X, batch_size=1)
        except:
            print(cnt, len(X[0]), X[0].shape)
            for _ in range(X[0].shape[1]):
                sc_text.write('-1') # Unable to predict sample? Write -1 to proceed
                sc_text.write('\n')
        else:
            for score_sample in score_all:
                for score in score_sample:
                    #print score
                    sc_text.write(str(score))
                    sc_text.write('\n')
    
    sc_text.close()
        

if __name__ == '__main__':
    
    if len(sys.argv) >= 3:
        train_case = sys.argv[1]
        test_case = sys.argv[2]
        use_trained = sys.argv[3] # This is a function under testing. 0 is stable
    else:
        train_case = "training"
        test_case = "randomed_labeled_testing" # "training" #"randomed_labeled_testing" # "develop"
        use_trained = True  # This is a function under testing. 0 is stable
    
    
    # Pre-processing
    
    # word_seg("debug.data", "debug.seg")
    # word_to_syn("debug.seg", "debug.syntrans")
    # calc_idf("debug.syntrans", "debug.idf")
    
    # word_seg("develop.data", "develop.seg")
    # word_to_syn("develop.seg", "develop.syntrans")
    # calc_idf("develop.syntrans", "develop.idf")
    
    # word_seg("training.data", "training.seg")
    #word_to_syn("training.seg", "training.syntrans")
    # calc_idf("training.syntrans", "training.idf")
    
    #word_seg("testing.data", "testing.seg")
    #word_to_syn("testing.seg", "testing.syntrans")
    # calc_idf("testing.syntrans", "testing.idf")
    
    #word_seg("randomed_labeled_testing.data", "randomed_labeled_testing.seg")
    #word_to_syn("randomed_labeled_testing.seg", "randomed_labeled_testing.syntrans")
    # calc_idf("randomed_labeled_testing.syntrans", "randomed_labeled_testing.idf")
    # gen_word_vec()


    
    my_model = layers.DL4AQS_model().model
    lf = config.TrainConfig.LOSS_FUNC
    optmz = config.TrainConfig.OPTIMIZER


    
    print("------Creating model for training------")

    my_model.compile(optimizer=optmz, loss=lf)

    if use_trained:
        my_model.load_weights("my_model_weights.h5")
        #my_model.optimizer


    
    print("------Training on %s data.------" % train_case)
    
    train_model("training.w2v", "%s.syntrans" % train_case, "%s.idf" % train_case, my_model)

    
    print("------Saving model------")
    
    my_model.save_weights("my_model_weights.h5")
    

    
    print("------Loading model for testing------")
    
    my_model = layers.DL4AQS_model().model
    my_model.load_weights("my_model_weights.h5")
    
    #my_model = load_model("my_model.h5")
    
    
    print("------testing on %s data.------" % test_case)
    
    evaluate_model("training.w2v", "%s.syntrans" % test_case, "%s.idf" % test_case, "%s.scores" % test_case, my_model)
    
