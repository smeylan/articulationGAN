import argparse
import os
import re
import itertools as it
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from scipy.io.wavfile import read, write
import scipy.stats
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import tempfile
import scipy
import uuid
import copy
import string
import glob 
from Levenshtein import distance as lev
import gc
import wandb
import yaml
import time
from torch.profiler import profile, record_function, ProfilerActivity


from infowavegan import WaveGANGenerator, ArticulationGANGenerator, WaveGANDiscriminator, ArticulationGANDiscriminator, WaveGANQNetwork, ArticulationGANQNetwork
from articulatory.utils import load_model


torch.autograd.set_detect_anomaly(True)

class AudioDataSet:
    def __init__(self, datadir, slice_len, NUM_CATEG, vocab, word_means=None, sigma=None):
        print("Loading data")
        dir = os.listdir(datadir)
        x = np.zeros((len(dir), 1, slice_len))
        y = np.zeros((len(dir), NUM_CATEG)) 
 
        i = 0
        files = []
        categ_labels = []
        for file in tqdm(dir):
            files.append(file)
            audio = read(os.path.join(datadir, file))[1]
            if audio.shape[0] < slice_len:
                audio = np.pad(audio, (0, slice_len - audio.shape[0]))
            audio = audio[:slice_len]

            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32767
            elif audio.dtype == np.float32:
                pass
            else:
                raise NotImplementedError('Scipy cannot process atypical WAV files.')
            audio /= np.max(np.abs(audio))
            x[i, 0, :] = audio

            # extract the label
            word = file.split('_')[0]
            j = vocab.index(word)
            y[i, j] = 1            
            categ_labels.append(j)
            i += 1

        categ_labels = np.array(categ_labels)

        if word_means is not None:
            # condition on the y values to choose real-valued positions in the semantic space associated with each of the words
            # implicitly, we could have generated these in the semantic space first, and run a classifier to get y values
            sem_vector_store = {} 
            for i in range(NUM_CATEG):
                sem_vector_store[i] = scipy.stats.multivariate_normal.rvs( mean=word_means[i], cov=sigma, size = len(files))

            # the category label i indexes the key in sem_vector_store, j indexes the row (ie many rows are not used)                 
            sem_vector =  [sem_vector_store[i][j,:] for i,j in zip(categ_labels, range(len(files)))]   
            self.sem_vector = torch.from_numpy(np.array(sem_vector, dtype=np.float32))            

        self.len = len(x)
        self.audio = torch.from_numpy(np.array(x, dtype=np.float32))
        self.labels = torch.from_numpy(np.array(y, dtype=np.float32))
        

    def __getitem__(self, index):
        if hasattr(self,'sem_vector'):
            return ((self.audio[index], self.labels[index], self.sem_vector[index]))
        else:
            return ((self.audio[index], self.labels[index]))

    def __len__(self):
        return self.len

def get_architecture_appropriate_c(architecture, num_categ, batch_size):
    if ARCHITECTURE == 'ciwgan':
        c = torch.nn.functional.one_hot(torch.randint(0, num_categ, (batch_size,)),
                                                num_classes=num_categ).to(device)
    elif ARCHITECTURE == 'fiwgan':
        c = torch.zeros([batch_size, num_categ], device=device).bernoulli_()
    else:
        assert False, "Architecture not recognized."
    return c

def gradient_penalty(G, D, reals, fakes, epsilon):
    x_hat = epsilon * reals + (1 - epsilon) * fakes
    scores = D(x_hat)
    grad = torch.autograd.grad(
        outputs=scores,
        inputs=x_hat,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True
    )[0]
    grad_norm = grad.view(grad.shape[0], -1).norm(p=2, dim=1)  # norm along each batch
    penalty = ((grad_norm - 1) ** 2).unsqueeze(1)
    return penalty


def Q2_cnn(selected_candidate_wavs, Q2, architecture):
    print('in Q2_cnn')
    if ARCHITECTURE == "ciwgan":
        Q2_probs = torch.softmax(Q2(selected_candidate_wavs), dim=1)
        # add a column for UNKs
        zeros = torch.zeros([Q2_probs.shape[0],1], device = device) + .00000001
        Q_network_probs_with_unk = torch.hstack((Q2_probs, zeros))
        return Q_network_probs_with_unk
    elif ARCHITECTURE == "eiwgan":
        # this directly returns the embeddings (of the dimensionsionality NUM_DIMS)
        return(Q2(selected_candidate_wavs))
    elif ARCHITECTURE == "fiwgan":
        Q2_binary = torch.sigmoid(Q2(selected_candidate_wavs))
        return Q2_binary
    else:
        raise ValueError('architecture for Q2_cnn must be one of (ciwgan, eiwgan, fiwgan)')


def write_out_wavs(architecture, G_z_2d, labels, vocab, logdir, epoch):
    # returns probabilities and a set of indices; takes a smaller number of arguments
    files_for_asr = []
    epoch_path = os.path.join(logdir,'audio_files',str(epoch))
    if not os.path.exists(epoch_path):
        os.makedirs(epoch_path)    

    labels_local = labels.cpu().detach().numpy()
    # for each file in the batch, write out a wavfile
    for j in range(G_z_2d.shape[0]):
        audio_buffer = G_z_2d[j,:].detach().cpu().numpy()
        if architecture == 'ciwgan':
            true_word = vocab[np.argwhere(labels_local[j,:])[0][0]]
        else:
            true_word = ''
        tf = os.path.join(epoch_path,true_word + '_' + str(uuid.uuid4())+".wav")
        write(tf, 16000, audio_buffer[0])
        files_for_asr.append(copy.copy(tf))
    return(files_for_asr)

def get_non_UNK_in_Q2_ciwgan(Q_network_probs, threshold, device):
    # compute entropies
    log_probs = torch.log(Q_network_probs + .000000001)
    prod = Q_network_probs * log_probs
    entropy = -torch.sum(prod, dim =1)        
    indices_of_recognized_words = torch.argwhere( entropy <= torch.tensor([threshold], device=device)).flatten()
    return indices_of_recognized_words

def get_non_UNK_in_Q2_fiwgan(Q_network_features, threshold, device):
    closest_vertices = torch.round(Q_network_features)
    distance = EuclideanLoss()
    distances = distance(Q_network_features, closest_vertices)
    #assert distances.shape == (Q_network_features.shape[0],)
    indices_of_recognized_words = torch.argwhere( distances <= torch.tensor([threshold], device=device)).flatten()
    return indices_of_recognized_words

def get_non_UNK_in_Q2_eiwgan(Q_network_features, word_means, threshold, device):
    distances = get_sem_vector_distance_to_means(Q_network_features, word_means)
    best = torch.min(distances, 0).values
    #assert best.shape == (Q_network_features.shape[0],)
    indices_of_recognized_words = torch.argwhere( best <= torch.tensor([threshold], device=device)).flatten()
    return indices_of_recognized_words

def one_hot_classify_sem_vector(Q2_sem_vecs, word_means):
    print('do a one_hot classification of the sem_vecs: find the closest word mean for each')
    distances = get_sem_vector_distance_to_means(Q2_sem_vecs, word_means)
    best = torch.argmin(distances,0)
    return best

def get_sem_vector_distance_to_means(Q2_sem_vecs, word_means):    
    dists = [] 
    for x in range(word_means.shape[0]):
        dists.append(torch.sqrt(torch.sum((Q2_sem_vecs - word_means[x]) ** 2, dim=1)))    
    distances = torch.vstack(dists)
    #assert distances.shape == (word_means.shape[0], Q2_sem_vecs.shape[0])
    return distances

class EuclideanLoss(nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()

    def forward(self, inputs, targets):
        return torch.sqrt(torch.sum((inputs - targets) ** 2, dim=1))

def get_replacement_features(architecture, num_examples, feature_size, vocab_size, device):
    return_tensor = None
    if architecture == 'ciwgan':
        random_labels = torch.randint(low=0, high=vocab_size, size = (num_examples,), device=device)
        onehot_per_word = F.one_hot(random_labels, num_classes = vocab_size + 1).to(device)
        return_tensor = onehot_per_word
    elif architecture == 'fiwgan':
        # high parameter is exclusive
        return_tensor = torch.randint(low=0, high=2, size = (num_examples, feature_size), device=device)   
    elif architecture == 'eiwgan':        
        return_tensor = torch_uniform((num_examples, feature_size), -1, 1, device)        
    else:
        raise NotImplementedError
    
    return return_tensor

def add_noise_to_label(original_features, replacement_features, q2_noise_probability, device):
    assert original_features.shape == replacement_features.shape
    assert len(original_features.shape) == 2
    mask_by_example_dim = torch.bernoulli(torch.ones(original_features.shape[0], device=device).fill_(q2_noise_probability))
    mask_features = mask_by_example_dim.unsqueeze(1).repeat(1, original_features.shape[-1]).int()

    candidate_referents = torch.where(mask_features == 1, replacement_features, original_features)
    return candidate_referents

def synthesize(model, x, config, step):
    '''
    Given batch of EMA data and EMA model, synthesizes speech output
    Args:
        x: (batch, art_len, num_feats)

    Return:
        signal: (batch, audio_len)
    '''
    t0 =  time.time()
    batch_size = x.shape[0]
    params_key = "generator_params"
    audio_chunk_len = config["batch_max_steps"]
    in_chunk_len = int(audio_chunk_len/config["hop_size"])
    past_out_len = config[params_key]["ar_input"]

    # NOTE extra_art not supported
    ins = [x[:, i:i+in_chunk_len, :] for i in range(0, x.shape[1], in_chunk_len)]
    prev_samples = torch.zeros((batch_size, config[params_key]["out_channels"], past_out_len), dtype=x.dtype, device=x.device)
    outs = []

    for cin in ins: # a2w cin (batch_size, in_chunk_len, num_feats)
        cin = cin.permute(0, 2, 1)  # a2w (batch_size, num_feats, in_chunk_len)
        cout = model(cin, ar=prev_samples)  # a2w (batch_size, 1, audio_chunk_length)
        outs.append(cout[:, 0, :])
        if past_out_len <= audio_chunk_len:
            prev_samples = cout[:, :, -past_out_len:]
        else:
            prev_samples[:, :, :-in_chunk_len] = prev_samples[:, :, in_chunk_len:].clone()
            prev_samples[:, :, -in_chunk_len:] = cout
    out = torch.unsqueeze(torch.cat(outs, dim=1), 1)  # w2a (batch_size, seq_len, num_feats)
    t1 =  time.time()
    time_checkpoint(t0, t1, 'synthesize time', step)

    return out

def time_checkpoint(t1, t2, label, step):
    elapsed = int(1000.*(t2-t1))
    print('Timed interval: '+label+' took '+str(elapsed)+' ms')
    wandb.log({"Time/"+label: elapsed}, step=step)

def torch_uniform(shape, low, high, device):
    rands = torch.rand(shape, device = device) #, dtype=torch.float64
    return(low + rands * (high - low))

def sample_multivatiate_normal(word_indices, word_means, sigma, NUM_CATEG, BATCH_SIZE):
    sem_vectors = []
    for categ_index in range(NUM_CATEG):
        sem_vectors.append(torch.distributions.MultivariateNormal(word_means[categ_index].double(), sigma.double()).sample((BATCH_SIZE,)))                

    sem_vec_store = torch.stack(sem_vectors).float()
    c = torch.vstack([sem_vec_store[i,j,:] for i,j in zip(word_indices, range(BATCH_SIZE))])
    return(c)

def sample_multivatiate_normal_for_categ(categ_index, word_means, sigma, BATCH_SIZE):
    return(torch.distributions.MultivariateNormal(word_means[categ_index].double(), sigma.double()).sample((BATCH_SIZE,)).float())                

    
    

if __name__ == "__main__":
    # Training Arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--architecture',
        type=str,
        required=True,
        help='What kind of latent space? Can be ciwgan, fiwgan, or eiwgan'
    )

    parser.add_argument(
        '--synthesizer',
        type=str,
        required=True,
        help='Can be WavGAN or ArticulationGAN'
    )

    parser.add_argument(
        '--log_dir',
        type=str,
        required=True,
        help='Log/Results Directory. Results will be stored by wandb_group / wandb_name / wandb_id (see below)'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='directory with labeled waveforms'
    )

    parser.add_argument(
        '--num_categ',
        type=int,
        default=0,
        help='Q-net categories'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=5000,
        help='Epochs'
    )
    parser.add_argument(
        '--slice_len',
        type=int,
        default=16384,
        help='Length of training data'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size'
    )

    parser.add_argument(
        '--save_int',
        type=int,
        default=25,
        help='Save interval in epochs'
    )

    parser.add_argument(
        '--track_q2',
        type=int,
        help='Track the results with the Q2 network; to backpropagate from Q2 to Q see "backprop_from_Q2"'
    )
    parser.add_argument(
        '--backprop_from_Q2',
        type=int,
        help='Update the Q network from the Q2 network'
    )

    parser.add_argument(
        '--production_start_epoch',
        type=int,
        help='Do n-1 epochs of pretraining the child Q network in the reference game. 0 means produce from the beginning',
        default=0
    )

    parser.add_argument(
        '--comprehension_interval',
        type=int,
        help='How often, in terms of epochs should the Q network be re-trained in the reference game. THe high default means that this is never run',
        default = 10000000
    )

    parser.add_argument(
        '--q2_unk_threshold',
        type=float,
        help="Float representing the entropy or distance threshold that is maximally tolerated to backprop the example. Otherwise, consider the example an UNK",
        default=100000
    )

    parser.add_argument(
        '--wandb_project',
        type=str,
        help='Name of the project for tracking in Weights and Biases',        
    )

    parser.add_argument(
        '--wandb_group',
        type=str,
        help='Name of the group / experiment to which this version (id) belongs',        
    )

    parser.add_argument(
        '--wandb_name',
        type=str,
        help='Name of this specific run',        
    )

    parser.add_argument(
        '--wavegan_disc_nupdates',
        type=int,
        help='On what interval, in steps, should the discriminator be updated? On other steps, the model updates the generator',
        default = 4
    )

    parser.add_argument(
        '--wavegan_q2_nupdates',
        type=int,
        help='On what interval, in steps, should the loss on the Q prediction of the Q2 labels be used to update the ! network ',
        default = 8
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        help="Float for the learning rate",
        default=1e-4
    )

    parser.add_argument(
        '--num_q2_training_epochs',
        type=int,
        help='Number of epochs to traine the adult model',
        default = 25
    )

    parser.add_argument(
        '--vocab',
        type=str,
        required=True,
        help='Space-separated vocabulary. Indices of words here will be used as the ground truth.'
    )

    parser.add_argument(
        '--q2_batch_size',
        type=int,
        help='Number of candidates to evaluate for each word to choose the best candidate',
        default = 6
    )

    parser.add_argument(
        '--q2_noise_probability',
        type=float,
        help="Probability that the action taken by Q2 is affected by noise and does not match Q's referent",
        default=0
    )

    # ema parameters
    parser.add_argument(
        '--emadir',
        type=str,
        required=True,
        help='EMA Weights Directory'
    )

    parser.add_argument(
        '--num_channels',
        type=int,
        default=13,
        help='Size of articulatory generator output'
    )

    parser.add_argument(
        '--kernel_len',
        type=int,
        default=7,
        help='Sets the generator kernel length, must be odd'
    )

    args = parser.parse_args()
    train_Q = True
    track_Q2 = bool(args.track_q2)
    vocab = args.vocab.split()

    ARCHITECTURE = args.architecture

    if args.synthesizer == 'ArticulationGAN':
        assert args.slice_len == 20480, "ArticulationGAN only supports a slice length of 20480"

    assert args.synthesizer in ("WavGAN","ArticulationGAN"), "synthesizer must be one of 'ArticulationGAN' or 'WavGAN'"

    assert args.architecture in ("eiwgan", 'fiwgan', 'ciwgan'), "architecture must be one of 'ciwgan', 'fiwgan' or 'eiwgan'"
    
    assert args.kernel_len % 2 == 1, f"generator kernel length must be odd, got: {args.kernel_len}"

    # Parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load the vocal tract simulator
    if args.num_channels == 12:
        synthesis_checkpoint_path = args.emadir + "/mngu0_fema2w_12ch/best_mel_ckpt.pkl"
        synthesis_config_path = args.emadir + "/mngu0_fema2w_12ch/config.yml"
    elif args.num_channels == 13:
        synthesis_checkpoint_path = args.emadir + "/mngu0_fema2w_13ch/best_mel_ckpt.pkl"
        synthesis_config_path = args.emadir + "/mngu0_fema2w_13ch/config.yml"

    with open(synthesis_config_path) as f:
        synthesis_config = yaml.load(f, Loader=yaml.Loader)


    datadir = args.data_dir

    # Epochs and Intervals
    NUM_EPOCHS = args.num_epochs
    WAVEGAN_DISC_NUPDATES = args.wavegan_disc_nupdates
    WAVEGAN_Q2_NUPDATES = args.wavegan_q2_nupdates
    Q2_EPOCH_START = 0 # in case we want to only run Q2 after a certain epoch. Less of a concern when a fast Q2 is used
    WAV_OUTPUT_N = 5
    SAVE_INT = args.save_int
    PRODUCTION_START_EPOCH = args.production_start_epoch
    COMPREHENSION_INTERVAL = args.comprehension_interval
    SELECTION_THRESHOLD = args.q2_unk_threshold

    #Sizes of things
    SLICE_LEN = args.slice_len
    NUM_CATEG = len(args.vocab.split(' '))
    BATCH_SIZE = args.batch_size

    # GAN Learning rates
    LAMBDA = 10
    LEARNING_RATE = args.learning_rate
    BETA1 = 0.5
    BETA2 = 0.9
    
    # Verbosity
    label_stages = True

    # Q2 parameters
    NUM_Q2_TRAINING_EPOCHS = args.num_q2_training_epochs
    Q2_BATCH_SIZE = args.q2_batch_size
   
    gpu_properties = torch.cuda.get_device_properties('cuda')
    kwargs = {
       'project' :  args.wandb_project,        
       'config' : args.__dict__,
       'group' : args.wandb_group,
       'name' : args.wandb_name,
       'config': {
            'slurm_job_id' : os.getenv('SLURM_JOB_ID'),
            'gpu_name' : gpu_properties.name,
            'gpu_memory' : gpu_properties.total_memory
        }
    }
    wandb.init(**kwargs)

    logdir = os.path.join(args.log_dir, args.wandb_group, args.wandb_name, wandb.run.id)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if ARCHITECTURE == "eiwgan":
        if NUM_CATEG != 4:
            raise ValueError('NUM_CATEG must be 4 for hard-coded word means in Eiwgan. Make this variable later.')
        word_means = np.array([[-.5,-.5],[-.5,.5],[.5,-.5],[.5,.5]])
        NUM_DIM = len(word_means[0])
        sigma = np.matrix([[.025,0],[0,.025]])
        
        dataset = AudioDataSet(datadir, SLICE_LEN, NUM_CATEG, vocab, word_means, sigma)

        word_means = torch.from_numpy(word_means).to(device)
        sigma = torch.from_numpy(sigma).to(device) 
    else:
        dataset = AudioDataSet(datadir, SLICE_LEN, NUM_CATEG, vocab)

    dataloader = DataLoader(
        dataset,
        BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        drop_last=True
    )

    def make_new():
        if args.synthesizer == "WavGAN":
            G = WaveGANGenerator(slice_len=SLICE_LEN, ).to(device).train()
            D = WaveGANDiscriminator(slice_len=SLICE_LEN).to(device).train()
        elif args.synthesizer == "ArticulationGAN":
            padding_len = (int)((args.kernel_len - 1)/2)
            G = ArticulationGANGenerator(nch=args.num_channels, kernel_len=args.kernel_len, padding_len=padding_len, use_batchnorm=False).to(device).train()        
            D = ArticulationGANDiscriminator(slice_len=SLICE_LEN).to(device).train()
        EMA = load_model(synthesis_checkpoint_path, synthesis_config)
        EMA.remove_weight_norm()
        EMA = EMA.eval().to(device)        

        # Optimizers
        optimizer_G = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
        optimizer_D = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

        Q, optimizer_Q_to_G, optimizer_Q_to_Q, criterion_Q, criterion_Q2 = (None, None, None, None, None)
        if args.synthesizer == 'WavGAN':
            create_Q_network_func = WaveGANQNetwork 
        elif args.synthesizer == 'ArticulationGAN':
            create_Q_network_func = ArticulationGANQNetwork 
        else:
            raise NotImplementedError
        if train_Q:
            if args.architecture in ('ciwgan','fiwgan'):
                Q = create_Q_network_func(slice_len=SLICE_LEN, num_categ=NUM_CATEG).to(device).train()
            elif args.architecture == 'eiwgan':
                Q = create_Q_network_func(slice_len=SLICE_LEN, num_categ=NUM_DIM).to(device).train()
                # number of dimensions in the sematnic space, not number of words
            else:
                raise ValueError('Architecure not recognized! Must be fiwgan or ciwgan')

            #optimizer_Q_to_G = optim.RMSprop(G.parameters(), lr=LEARNING_RATE)
            optimizer_Q_to_QG = optim.RMSprop(it.chain(G.parameters(), Q.parameters()), lr=LEARNING_RATE)            
            # just update the G parameters            

            if args.architecture == 'fiwgan':
                print("Training a fiwGAN with ", NUM_CATEG, " categories.")
                criterion_Q = torch.nn.BCEWithLogitsLoss() # binary cross entropy                
            elif args.architecture == 'eiwgan':
                print("Training a eiwGAN with ", NUM_CATEG, " categories.")
                criterion_Q = EuclideanLoss()

            elif args.architecture == 'ciwgan':
                print("Training a ciwGAN with ", NUM_CATEG, " categories.")
                # NOTE: one hot -> category nr. transformation
                # CE loss needs logit, category -> loss
                criterion_Q = lambda inpt, target: torch.nn.CrossEntropyLoss()(inpt, target.max(dim=1)[1])
            else:
                raise ValueError('Architecure not recognized! Must be fiwgan or ciwgan')                


        if track_Q2:
            if ARCHITECTURE in ("ciwgan","fiwgan"):
                Q2 = create_Q_network_func(slice_len=SLICE_LEN, num_categ=NUM_CATEG).to(device).train()
            elif ARCHITECTURE == "eiwgan":
                Q2 = create_Q_network_func(slice_len=SLICE_LEN, num_categ=NUM_DIM).to(device).train()

            optimizer_Q2_to_QG = optim.RMSprop(it.chain(G.parameters(), Q.parameters()), lr=LEARNING_RATE)
            optimizer_Q2_to_Q2 = optim.RMSprop(Q2.parameters(), lr=LEARNING_RATE)

            if PRODUCTION_START_EPOCH > 0:
                optimizer_Q_to_Q = optim.RMSprop(Q.parameters(), lr=LEARNING_RATE)

            
            if ARCHITECTURE == 'fiwgan':
                criterion_Q2 = torch.nn.BCEWithLogitsLoss() # binary cross entropy

            elif ARCHITECTURE == "eiwgan":
                criterion_Q2 = EuclideanLoss()            
            
            elif ARCHITECTURE == "ciwgan":
                criterion_Q2 = lambda inpt, target: torch.nn.CrossEntropyLoss()(inpt, target.max(dim=1)[1])

            else:
                raise NotImplementedError        

        return G, D, EMA, optimizer_G, optimizer_D, Q, Q2, optimizer_Q_to_QG, optimizer_Q2_to_QG, optimizer_Q2_to_Q2, optimizer_Q_to_Q, criterion_Q, criterion_Q2

    # Load models    
    G, D, EMA, optimizer_G, optimizer_D, Q, Q2, optimizer_Q_to_QG, optimizer_Q2_to_QG, optimizer_Q2_to_Q2, optimizer_Q_to_Q, criterion_Q, criterion_Q2 = make_new()
    
    start_epoch = 0
    start_step = 0

    print("Starting a new training. MNLL does not have support for resuming traning")

    step = start_step

    Q2_network_path = 'saved_networks/adult_pretrained_Q_network_'+args.synthesizer+str(NUM_CATEG)+'_'+ARCHITECTURE+'.torch'

    use_cached_Q2 = True
    if os.path.exists(Q2_network_path) and use_cached_Q2:
        print('Loading a Previous Adult Q2 CNN Network')
        Q2 = torch.load(Q2_network_path).to(device)

    else:
        print('Training an Adult Q2 CNN Network')
        step = start_step
        for epoch in range(start_epoch + 1, NUM_Q2_TRAINING_EPOCHS):
            print("Epoch {} of {}".format(epoch, NUM_Q2_TRAINING_EPOCHS))
            print("-----------------------------------------")

            pbar = tqdm(dataloader)            
            for i, trial in enumerate(pbar):            
                reals = trial[0].to(device)
                labels = trial[1].to(device)  
                if ARCHITECTURE == 'eiwgan':  
                    continuous_labels = trial[2].to(device)
                optimizer_Q2_to_Q2.zero_grad()
                Q2_logits = Q2(reals)    
                
                if ARCHITECTURE == 'ciwgan':
                    Q2_comprehension_loss = criterion_Q(Q2_logits, labels[:,0:NUM_CATEG]) # Note we exclude the UNK label --  child never intends to produce unk
                elif ARCHITECTURE == 'fiwgan':
                    Q2_comprehension_loss = criterion_Q(Q2_logits, labels)
                elif ARCHITECTURE == "eiwgan":                    
                    Q2_comprehension_loss = torch.mean(criterion_Q(Q2_logits, continuous_labels))

                Q2_comprehension_loss.backward()

                wandb.log({"Loss/Q2 to Q2": Q2_comprehension_loss.detach().item()}, step=step)
                optimizer_Q2_to_Q2.step()
                step += 1
        if not os.path.exists('saved_networks/'):
            os.makedirs('saved_networks/')
        torch.save(Q2, 'saved_networks/adult_pretrained_Q_network_'+args.synthesizer+str(NUM_CATEG)+'_'+ARCHITECTURE+'.torch')
    
    # freeze Q2
    Q2.eval()        
    for p in Q2.parameters():
        p.requires_grad = False

    for epoch in range(start_epoch + 1, NUM_EPOCHS):

        print("Epoch {} of {}".format(epoch, NUM_EPOCHS))
        print("-----------------------------------------")
        t1 = time.time()

        pbar = tqdm(dataloader)        

        for i, trial in enumerate(pbar):
            
            reals = trial[0].to(device)
            labels = trial[1].to(device)            
           
            if (epoch <= PRODUCTION_START_EPOCH) or (epoch % COMPREHENSION_INTERVAL == 0):


                # Just train the Q network from external data
                if ARCHITECTURE == 'eiwgan':
                    # pretraining not implemented yet for eiwgan. Should be simple though -- Q(reals) in the same way, with a Euclidean loss function                
                    adult_label_to_recover = trial[2].to(device)
                else:
                    raise NotImplementedError
                
                if label_stages:
                    print('Updating Child Q network to identify referents')

                optimizer_Q_to_Q.zero_grad()
                child_recovers_from_adult = Q(reals)    
                Q_comprehension_loss = torch.mean(criterion_Q2(child_recovers_from_adult, adult_label_to_recover))
                Q_comprehension_loss.backward()
                wandb.log({"Loss/Q to Q": Q_comprehension_loss.detach().item()}, step=step)
                optimizer_Q_to_Q.step()
                step += 1
                            

            else:
                # Discriminator Update
                t2 = time.time()
                optimizer_D.zero_grad()                

                epsilon = torch.rand(BATCH_SIZE, 1, 1, device=device).repeat(1, 1, SLICE_LEN)
                
                if ARCHITECTURE == "eiwgan":
                    # draw from the semantic space a c that will need to be encoded
                
                    words = torch.nn.functional.one_hot(torch.randint(0, NUM_CATEG, (BATCH_SIZE,)),
                             num_classes=NUM_CATEG).detach().numpy() # randomly generate a bunch of one-hots
                    word_indices = [x[1] for x in np.argwhere(words)]                                                                                       
                    c = sample_multivatiate_normal(word_indices, word_means, sigma, NUM_CATEG, BATCH_SIZE)
                    _z = torch_uniform([BATCH_SIZE, 100 - NUM_DIM], -1,1, device) 
                    z = torch.cat((c, _z), dim=1)


                elif ARCHITECTURE == 'ciwgan':
                    c = get_architecture_appropriate_c(ARCHITECTURE, NUM_CATEG, BATCH_SIZE)
                    _z = torch_uniform([BATCH_SIZE, 100 - (NUM_CATEG + 1)], -1, 1, device)                    
                    zeros = torch.zeros([BATCH_SIZE,1], device = device)
                    z = torch.cat((c, zeros, _z), dim=1)
                elif ARCHITECTURE == 'fiwgan': 
                    c = get_architecture_appropriate_c(ARCHITECTURE, NUM_CATEG, BATCH_SIZE)
                    _z = torch_uniform([BATCH_SIZE, 100 - NUM_CATEG], -1, 1, device)
                    z = torch.cat((c, _z), dim=1)
                else:
                    raise NotImplementedError
                
                if args.synthesizer == "WavGAN":    
                    fakes = G(z)
                elif args.synthesizer == "ArticulationGAN":    
                    fakes = synthesize(EMA, G(z).permute(0, 2, 1), synthesis_config, step)                    
             
                # shuffle the reals so that the matched item for discrim is not necessarily from the same referent. This is because the GAN is learning to distinguish *unconditioned* draws from G with real examples                
                shuffled_reals = reals[torch.randperm(reals.shape[0]),:,:]

                penalty = gradient_penalty(G, D, shuffled_reals, fakes, epsilon)
                D_loss = torch.mean(D(fakes) - D(shuffled_reals) + LAMBDA * penalty)
                
                wandb.log({"Loss/D": D_loss.detach().item()}, step=step)
                D_loss.backward()
                if label_stages:
                    print('Discriminator update!')
                optimizer_D.step()            
                optimizer_D.zero_grad()
                t3 = time.time()
                time_checkpoint(t2, t3, 'Discriminator update', step)


                if i % WAVEGAN_DISC_NUPDATES == 0:
                    t4 = time.time()
                    optimizer_G.zero_grad()      
                    EMA.zero_grad()                        
                    
                    if label_stages:
                        print('D -> G  update')
                    
                    if ARCHITECTURE == "eiwgan":
                        # draw from the semantic space a c that will need to be encoded
                
                        words = torch.nn.functional.one_hot(torch.randint(0, NUM_CATEG, (BATCH_SIZE,)),
                             num_classes=NUM_CATEG).detach().numpy() # randomly generate a bunch of one-hots
                        word_indices = [x[1] for x in np.argwhere(words)]                                                                    
                        c = sample_multivatiate_normal(word_indices, word_means, sigma, NUM_CATEG, BATCH_SIZE)
                        _z = torch_uniform([BATCH_SIZE, 100 - NUM_DIM], -1,1, device)                    
                        z = torch.cat((c, _z), dim=1)
                    elif ARCHITECTURE == 'fiwgan':
                        c = get_architecture_appropriate_c(ARCHITECTURE, NUM_CATEG, BATCH_SIZE)
                        _z = torch_uniform([BATCH_SIZE, 100 - NUM_CATEG], -1, 1, device)
                        z = torch.cat((c, _z), dim=1)
                    elif ARCHITECTURE == 'ciwgan':
                        c = get_architecture_appropriate_c(ARCHITECTURE, NUM_CATEG, BATCH_SIZE)
                        _z = torch_uniform([BATCH_SIZE, 100 - (NUM_CATEG + 1)], -1, 1, device)
                        zeros = torch.zeros([BATCH_SIZE,1], device = device)
                        z = torch.cat((c, zeros, _z), dim=1)

                    if args.synthesizer == "WavGAN":
                        G_z_for_G_update = G(z) # generate again using the same labels
                    elif args.synthesizer == "ArticulationGAN":
                        articul_out = G(z)
                        G_z_for_G_update = synthesize(EMA, articul_out.permute(0, 2, 1), synthesis_config, step)

                    # G Loss                    
                    G_loss = torch.mean(-D(G_z_for_G_update))
                    G_loss.backward(retain_graph=True)
                    # Update
                    optimizer_G.step()
                    optimizer_G.zero_grad()
                    if label_stages:
                        print('Generator update!')
                    wandb.log({"Loss/G": G_loss.detach().item()}, step=step)
                    t5 = time.time()
                    time_checkpoint(t4, t5, 'D -> G update', step)


                    if (epoch % WAV_OUTPUT_N == 0) & (i <= 1):
                        t6 = time.time()
                        print('Sampling .wav outputs (but not running them through Q2)...')
                        as_words = torch.from_numpy(words).to(device) if ARCHITECTURE == 'eiwgan' else c
                        write_out_wavs(ARCHITECTURE, G_z_for_G_update, as_words, vocab, logdir, epoch)
                        t7 = time.time()
                        time_checkpoint(t6, t7, 'WAV writeout', step)                        
                        # but don't do anything with it; just let it write out all of the audio files
                
                    # Q2 Loss: Update G and Q to better imitate the Q2 model
                    if (i != 0) and track_Q2 and (i % WAVEGAN_Q2_NUPDATES == 0) & (epoch >= Q2_EPOCH_START):
                        
                        if label_stages:
                            print('Starting Q2 evaluation...')                        

                        t8 = time.time()
                        optimizer_Q2_to_QG.zero_grad() # clear the gradients for the Q update

                        selected_candidate_wavs = []  
                        selected_Q_estimates = []

                        print('Choosing '+str(Q2_BATCH_SIZE)+' best candidates for each word...')

                        # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
                        #    with record_function("num_candidates_section"):

                        if ARCHITECTURE == 'ciwgan':
                            predicted_value_loss = torch.nn.CrossEntropyLoss()
                            selected_referents = []
                            for categ_index in range(NUM_CATEG):
                                
                                num_candidates_to_consider_per_word = 1 # increasing this breaks stuff. Results in considering a larger space

                                # generate a large numver of possible candidates
                                candidate_referents = torch.zeros([Q2_BATCH_SIZE*num_candidates_to_consider_per_word, NUM_CATEG+1], device=device)
                                candidate_referents[:,categ_index] = 1                                     
                                _z = torch_uniform([Q2_BATCH_SIZE*num_candidates_to_consider_per_word, 100 - (NUM_CATEG + 1)], -1, 1, device)                                

                                # generate new candidate wavs
                                if args.synthesizer == "WavGAN":
                                    candidate_wavs = G(torch.cat((candidate_referents, _z), dim=1))
                                elif args.synthesizer == "ArticulationGAN":
                                    candidate_wavs = synthesize(EMA, G(torch.cat((candidate_referents, _z), dim=1)).permute(0, 2, 1), synthesis_config, step)                                
                                candidate_Q_estimates = Q(candidate_wavs)

                                # select the Q2_BATCH_SIZE items that are most likely to produce the correct response
                                candidate_predicted_values = torch.Tensor([predicted_value_loss(candidate_Q_estimates[i], candidate_referents[i,0:NUM_CATEG]) for i in range(candidate_referents.shape[0])])                                
                                # order by their predicted score
                                candidate_ordering = torch.argsort(candidate_predicted_values, dim=- 1, descending=False, stable=False)

                                # select a subset of the candidates
                                selected_candidate_wavs.append(torch.narrow(candidate_wavs[candidate_ordering,:], dim=0, start=0, length=Q2_BATCH_SIZE)[:,0].clone())
                                selected_referents.append(torch.narrow(candidate_referents[candidate_ordering,:], dim=0, start=0, length=Q2_BATCH_SIZE).clone())
                                selected_Q_estimates.append(torch.narrow(candidate_Q_estimates[candidate_ordering,:], dim=0, start=0, length=Q2_BATCH_SIZE).clone())

                                del candidate_referents
                                del candidate_wavs
                                del candidate_Q_estimates
                                gc.collect()
                                torch.cuda.empty_cache()
                            

                            print('collapsing candidates')
                            selected_candidate_wavs = torch.vstack(selected_candidate_wavs)
                            selected_referents =  torch.vstack(selected_referents)
                            selected_Q_estimates = torch.vstack(selected_Q_estimates)  

                        if ARCHITECTURE == 'fiwgan':
                            predicted_value_loss = torch.nn.BCEWithLogitsLoss()
                            selected_referents = []
                            for categ_index in range(NUM_CATEG):
                                
                                num_candidates_to_consider_per_word = 1 # increasing this breaks stuff. Results in considering a larger space

                                # generate a large numver of possible candidates
                                candidate_referents = np.zeros([Q2_BATCH_SIZE*num_candidates_to_consider_per_word, NUM_CATEG], dtype=np.float32)
                                candidate_referents[:,categ_index] = 1     
                                candidate_referents = torch.from_numpy(candidate_referents).to(device)                                
                                _z = torch_uniform([Q2_BATCH_SIZE*num_candidates_to_consider_per_word, 100 - NUM_CATEG], -1, 1, device)


                                # generate new candidate wavs
                                if args.synthesizer == "WavGAN":
                                    candidate_wavs = G(torch.cat((candidate_referents, _z), dim=1))
                                elif args.synthesizer == "ArticulationGAN":
                                    candidate_wavs = synthesize(EMA, G(torch.cat((candidate_referents, _z), dim=1)).permute(0, 2, 1), synthesis_config, step)                                

                                candidate_Q_estimates = Q(candidate_wavs)

                                # select the Q2_BATCH_SIZE items that are most likely to produce the correct response
                                candidate_predicted_values = torch.Tensor([predicted_value_loss(candidate_Q_estimates[i], candidate_referents[i,0:NUM_CATEG]) for i in range(candidate_referents.shape[0])])                                
                                # order by their predicted score
                                candidate_ordering = torch.argsort(candidate_predicted_values, dim=- 1, descending=False, stable=False)

                                # select a subset of the candidates
                                selected_candidate_wavs.append(torch.narrow(candidate_wavs[candidate_ordering,:], dim=0, start=0, length=Q2_BATCH_SIZE)[:,0].clone())
                                selected_referents.append(torch.narrow(candidate_referents[candidate_ordering,:], dim=0, start=0, length=Q2_BATCH_SIZE).clone())
                                selected_Q_estimates.append(torch.narrow(candidate_Q_estimates[candidate_ordering,:], dim=0, start=0, length=Q2_BATCH_SIZE).clone())

                                del candidate_referents
                                del candidate_wavs
                                del candidate_Q_estimates
                                gc.collect()
                                torch.cuda.empty_cache()

                            print('collapsing candidates')
                            selected_candidate_wavs = torch.vstack(selected_candidate_wavs)
                            selected_referents =  torch.vstack(selected_referents)
                            selected_Q_estimates = torch.vstack(selected_Q_estimates)  
                        
                        elif ARCHITECTURE == 'eiwgan':                            
                            
                            selected_meanings = []
                            selected_referents = []
                            for categ_index in range(NUM_CATEG):
                                
                                # increasing this breaks stuff. Results in considering a larger space
                                num_candidates_to_consider_per_word = 1 

                                # propagae the categorical label associated with the Gaussian for checking what Q2 infers
                                candidate_referents = torch.zeros([Q2_BATCH_SIZE*num_candidates_to_consider_per_word, NUM_CATEG+1], device=device)
                                candidate_referents[:,categ_index] = 1                                

                                print('check if multivaraite sampleing is working')
                                                                
                                #candidate_meanings rather than candidate references                                
                                candidate_meanings = sample_multivatiate_normal_for_categ(categ_index, word_means, sigma, Q2_BATCH_SIZE*num_candidates_to_consider_per_word)
                                _z = torch_uniform([Q2_BATCH_SIZE*num_candidates_to_consider_per_word, 100 - NUM_DIM], -1,1, device) 

                                # generate new candidate wavs
                                if args.synthesizer == "WavGAN":
                                    candidate_wavs = G(torch.cat((candidate_meanings, _z), dim=1))
                                elif args.synthesizer == "ArticulationGAN":                                        
                                    candidate_wavs = synthesize(EMA, G(torch.hstack([candidate_meanings, _z])).permute(0, 2, 1), synthesis_config, step)

                                candidate_Q_estimates = Q(candidate_wavs)

                                # select the Q2_BATCH_SIZE items that are most likely to produce the correct response -- those that have the smallest distance under the model

                                # compute the distances
                                candidate_predicted_values = criterion_Q2(candidate_Q_estimates, candidate_meanings)

                                # order by their predicted score
                                candidate_ordering = torch.argsort(candidate_predicted_values, dim=- 1, descending=True, stable=False)

                                # select a subset of the candidates
                                selected_candidate_wavs.append(torch.narrow(candidate_wavs[candidate_ordering,:], dim=0, start=0, length=Q2_BATCH_SIZE)[:,0])
                                selected_meanings.append(torch.narrow(candidate_meanings[candidate_ordering,:], dim=0, start=0, length=Q2_BATCH_SIZE))
                                selected_referents.append(torch.narrow(candidate_referents[candidate_ordering,:], dim=0, start=0, length=Q2_BATCH_SIZE))
                                selected_Q_estimates.append(torch.narrow(candidate_Q_estimates[candidate_ordering,:], dim=0, start=0, length=Q2_BATCH_SIZE))


                                del candidate_meanings
                                del candidate_referents
                                del candidate_wavs
                                del candidate_Q_estimates
                                gc.collect()
                                torch.cuda.empty_cache()
                            

                            print('collapsing candidates')
                            selected_candidate_wavs = torch.vstack(selected_candidate_wavs)
                            selected_meanings =  torch.vstack(selected_meanings)
                            selected_referents = torch.vstack(selected_referents)
                            selected_Q_estimates = torch.vstack(selected_Q_estimates)  

                        
                        # import pdb; pdb.set_trace()

                        t9 = time.time()
                        time_checkpoint(t8, t9, 'Utterance selection', step)                        


                        print('Recognizing G output with Q2 model...')
                        t10 = time.time()                        

                        Q2_features = Q2_cnn(selected_candidate_wavs.unsqueeze(1), Q2, ARCHITECTURE)
                        assert len(Q2_features.shape) == 2

                        replacement_features = get_replacement_features(ARCHITECTURE, Q2_features.shape[0], Q2_features.shape[1], len(vocab), device)
                        mixed_Q2_features = add_noise_to_label(Q2_features, replacement_features, args.q2_noise_probability, device)


                        if ARCHITECTURE == 'ciwgan':
                            mixed_indices_of_recognized_words = get_non_UNK_in_Q2_ciwgan(mixed_Q2_features, SELECTION_THRESHOLD, device)
                            pure_indices_of_recognized_words = get_non_UNK_in_Q2_ciwgan(Q2_features, SELECTION_THRESHOLD, device)
                        
                        elif ARCHITECTURE == 'fiwgan':
                            mixed_indices_of_recognized_words = get_non_UNK_in_Q2_fiwgan(mixed_Q2_features, SELECTION_THRESHOLD, device)
                            pure_indices_of_recognized_words = get_non_UNK_in_Q2_fiwgan(Q2_features, SELECTION_THRESHOLD, device)
                        
                        elif ARCHITECTURE == "eiwgan":                            
                            mixed_indices_of_recognized_words = get_non_UNK_in_Q2_eiwgan(mixed_Q2_features, word_means, SELECTION_THRESHOLD, device)
                            pure_indices_of_recognized_words = get_non_UNK_in_Q2_eiwgan(Q2_features, word_means, SELECTION_THRESHOLD, device)
                        else:
                            raise NotImplementedError
                        
                        total_recognized_words = len(pure_indices_of_recognized_words)

                        if len(mixed_indices_of_recognized_words) > 0:
                            
                            print('Comparing Q predictions to Q2 output')        

                            # Q_of_selected_candidates is the expected value of each utterance

                            #Q_prediction = torch.softmax(selected_Q_estimates, dim=1)  
                            zero_tensor = torch.zeros(selected_Q_estimates.shape[0],1, device=device)  # for padding the UNKs, in logit space                              
    
                            # this is a one shot game for each reference, so implicitly the value before taking the action is 0. I might update this later, i.e., think about this in terms of sequences                   
                                                    
                            # compute the cross entropy between the Q network and the Q2 outputs, which are class labels recovered by the adults                                                    
                            if ARCHITECTURE == 'ciwgan':    
                        
                                Q_prediction = torch.softmax(selected_Q_estimates, dim=1)
                                augmented_Q_prediction = torch.log(torch.hstack((Q_prediction, zero_tensor)) + .0000001)
                                
                                mixed_Q2_loss = criterion_Q2(augmented_Q_prediction[mixed_indices_of_recognized_words], mixed_Q2_features[mixed_indices_of_recognized_words])   
                                with torch.no_grad():
                                    Q2_loss = criterion_Q2(augmented_Q_prediction[pure_indices_of_recognized_words], Q2_features[pure_indices_of_recognized_words])   

                                if not torch.equal(torch.argmax(selected_referents, dim=1), torch.argmax(Q_prediction, dim =1)):
                                    print("Child model produced an utterance that they don't think will invoke the correct action. Consider choosing action from a larger set of actions. Disregard if this is early in training and the Q network is not trained yet.")

                                # count the number of words that Q recovers the same thing as Q2
                                Q_recovers_Q2 = torch.eq(torch.argmax(augmented_Q_prediction[pure_indices_of_recognized_words], dim=1), torch.argmax(Q2_features[pure_indices_of_recognized_words], dim=1)).cpu().numpy().tolist()
                                Q2_recovers_child = torch.eq(torch.argmax(selected_referents[pure_indices_of_recognized_words], dim=1), torch.argmax(Q2_features[pure_indices_of_recognized_words], dim=1)).cpu().numpy().tolist()

                            elif ARCHITECTURE == 'fiwgan':

                                Q_prediction = torch.softmax(selected_Q_estimates, dim=1)
                                augmented_Q_prediction = torch.log(Q_prediction + .0000001)   
                                mixed_Q2_loss = criterion_Q2(augmented_Q_prediction[mixed_indices_of_recognized_words], mixed_Q2_features[mixed_indices_of_recognized_words])   
                                with torch.no_grad():
                                    Q2_loss = criterion_Q2(augmented_Q_prediction[pure_indices_of_recognized_words], Q2_features[pure_indices_of_recognized_words])   
                                def count_binary_vector_matches(raw_set1, raw_set2):
                                    matches_mask = []
                                    threshold = 0.5
                                    binarize = lambda vector : (vector >= threshold).int()
                                    set1, set2 = binarize(raw_set1), binarize(raw_set2)
                                    assert set1.shape == set2.shape
                                    for i in range(set1.shape[0]):
                                        match_int = 1 if torch.all(set1[i] == set2[i]) else 0
                                        matches_mask.append(match_int)
                                    return matches_mask

                                Q_recovers_Q2 = count_binary_vector_matches(augmented_Q_prediction[pure_indices_of_recognized_words], Q2_features[pure_indices_of_recognized_words])
                                Q2_recovers_child = count_binary_vector_matches(selected_referents[pure_indices_of_recognized_words], Q2_features[pure_indices_of_recognized_words])
                            
                            elif ARCHITECTURE == 'eiwgan': 
                                Q_prediction = selected_Q_estimates
                                embeddings_path = os.path.join(logdir,'meaning_embeddings')
                                if not os.path.exists(embeddings_path):
                                    os.makedirs(embeddings_path)

                                #'Write out the initial vectors, the Q predictions, and the Q2 interpretations
                                pd.DataFrame(selected_meanings.detach().cpu().numpy()).to_csv(os.path.join(embeddings_path,str(epoch)+'_selected_meaning.csv'))
                                pd.DataFrame(Q_prediction.detach().cpu().numpy()).to_csv(os.path.join(embeddings_path,str(epoch)+'_Q_prediction.csv'))
                                pd.DataFrame(Q2_features.detach().cpu().numpy()).to_csv(os.path.join(embeddings_path,str(epoch)+'_Q2_sem_vecs.csv'))

                                Q2_loss = torch.mean(criterion_Q2(Q_prediction[mixed_indices_of_recognized_words], mixed_Q2_features[mixed_indices_of_recognized_words]))                                
                                print('Check if we recover the one-hot that was used to draw the continuously valued vector')                                
                                Q2_recovers_child = torch.eq(torch.argmax(selected_referents[pure_indices_of_recognized_words], dim=1), one_hot_classify_sem_vector(Q2_features[pure_indices_of_recognized_words], word_means)).cpu().numpy().tolist()                                
                                                                                                                    
                            #this is where we would compute the loss
                            if args.backprop_from_Q2:
                                if ARCHITECTURE in ('ciwgan', 'fiwgan'):
                                    mixed_Q2_loss.backward(retain_graph=True)
                                elif ARCHITECTURE == 'eiwgan':
                                    Q2_loss.backward(retain_graph=True)
                                else:
                                    raise NotImplementedError
                            else:
                                print('Computing Q2 network loss but not backpropagating...')
                                
                            print('Gradients on the Q network:')
                            print('Q layer 0: '+str(np.round(torch.sum(torch.abs(Q.downconv_0.conv.weight.grad)).cpu().numpy(), 10)))
                            print('Q layer 1: '+str(np.round(torch.sum(torch.abs(Q.downconv_1.conv.weight.grad)).cpu().numpy(), 10)))
                            print('Q layer 2: '+str(np.round(torch.sum(torch.abs(Q.downconv_2.conv.weight.grad)).cpu().numpy(), 10)))
                            print('Q layer 3: '+str(np.round(torch.sum(torch.abs(Q.downconv_3.conv.weight.grad)).cpu().numpy(), 10)))
                            print('Q layer 4: '+str(np.round(torch.sum(torch.abs(Q.downconv_4.conv.weight.grad)).cpu().numpy(), 10)))

                            #print('Q2 -> Q update!')
                            #this is where we would do the step
                            if args.backprop_from_Q2:
                                print('Q2 -> Q update!')
                                optimizer_Q2_to_QG.step()
                            optimizer_Q2_to_QG.zero_grad()

                            total_Q2_recovers_child = np.sum(Q2_recovers_child)

                            if ARCHITECTURE in ('ciwgan', 'fiwgan'):
                                total_Q_recovers_Q2 = np.sum(Q_recovers_Q2)                                                            

                            wandb.log({"Loss/Q2 to Q": Q2_loss.detach().item()}, step=step)
                        
                        if len(pure_indices_of_recognized_words) == 0:
                            if ARCHITECTURE in ('eiwgan','ciwgan'):
                                total_Q2_recovers_child = 0
                            if ARCHITECTURE  == 'ciwgan':
                                total_Q_recovers_Q2 = 0

                        wandb.log({"Metric/Number of Referents Recovered by Q2": total_Q2_recovers_child}, step=step)

                        if ARCHITECTURE  in ('ciwgan', 'fiwgan'):
                            # How often does the Q network repliacte the Q2 network
                            wandb.log({"Metric/Number of Q2 references replicated by Q": total_Q_recovers_Q2}, step=step)
                            wandb.log({"Metric/Proportion Recognized Words Among Total": total_recognized_words / (Q2_BATCH_SIZE *NUM_CATEG)}, step=step) 


                        t11 = time.time()                        
                        time_checkpoint(t10, t11, 'adult evaluation', step)
                        

                   
                    if label_stages:
                        print('Q -> G, Q update')
                    t12 = time.time()

                    if ARCHITECTURE == 'ciwgan':
                        c = get_architecture_appropriate_c(ARCHITECTURE, NUM_CATEG, BATCH_SIZE)
                        _z = torch_uniform([BATCH_SIZE, 100 - (NUM_CATEG + 1)], -1, 1, device)
                        zeros = torch.zeros([BATCH_SIZE,1], device = device)
                        z = torch.cat((c, zeros, _z), dim=1)

                    if ARCHITECTURE == 'fiwgan':
                        c = get_architecture_appropriate_c(ARCHITECTURE, NUM_CATEG, BATCH_SIZE)
                        _z = torch_uniform([BATCH_SIZE, 100 - NUM_CATEG], -1, 1, device)
                        z = torch.cat((c, _z), dim=1)
                    
                    elif ARCHITECTURE == "eiwgan":
                        # draw from the semantic space a c that will need to be encoded
                
                        words = torch.nn.functional.one_hot(torch.randint(0, NUM_CATEG, (BATCH_SIZE,)),
                             num_classes=NUM_CATEG).detach().numpy() # randomly generate a bunch of one-hots
                        word_indices = [x[1] for x in np.argwhere(words)]                                                                    
                        c = sample_multivatiate_normal(word_indices, word_means, sigma, NUM_CATEG, BATCH_SIZE)
                        _z = torch_uniform([BATCH_SIZE, 100 - NUM_DIM], -1,1, device)
                        z = torch.cat((c, _z), dim=1)                            
                    
                    if args.synthesizer == "WavGAN":
                        G_z_for_Q_update = G(z) # generate again using the same labels
                    elif args.synthesizer == "ArticulationGAN":
                        G_z_for_Q_update = synthesize(EMA, G(z).permute(0, 2, 1), synthesis_config, step) 

                    optimizer_Q_to_QG.zero_grad()
                    if ARCHITECTURE == "eiwgan":
                        
                        Q_production_loss = torch.mean(criterion_Q(Q(G_z_for_Q_update), c))
                        # distance in the semantic space between what the child expects the adult to revover and what the child actually does

                    elif ARCHITECTURE in {"ciwgan", "fiwgan"}:
                                                
                        Q_production_loss = criterion_Q(Q(G_z_for_Q_update), c[:,0:NUM_CATEG])

                    Q_production_loss.backward()
                    wandb.log({"Loss/Q to G": Q_production_loss.detach().item()}, step=step)
                    optimizer_Q_to_QG.step()
                    optimizer_Q_to_QG.zero_grad()

                    t13 = time.time()
                    time_checkpoint(t12, t13, 'Q -> G update', step) 
                
                t16 = time.time()
                time_checkpoint(t2, t16, 'Step duration', step)
                step += 1
                
        # save out the articulation images
        if 'articul_out' in locals(): # may be undefined when fitting Q2
            t14 = time.time()
            artic_path = os.path.join(logdir,'artic_trajectories',str(epoch))
            if not os.path.exists(artic_path):
                os.makedirs(artic_path)

            for i in range(args.num_channels):
                articul = articul_out[0,i,:].cpu().detach().numpy()                    
                plt.plot(range(len(articul)), articul)
                plt.savefig(os.path.join(artic_path, "articulation_channel_"+str(i)+".png"))
                plt.close()

            t15 = time.time()
            time_checkpoint(t14, t15, 'Artic images', epoch)                 

        t17 = time.time()
        time_checkpoint(t1, t17, 'Epoch duration', epoch)

        if epoch % SAVE_INT == 0:
            if G is not None:
                torch.save(G.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_G.pt'))
            if D is not None:
                torch.save(D.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_D.pt'))
            if train_Q:
                torch.save(Q.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Q.pt'))                
                # these is no Q2 network to save, nor QQ            

            if optimizer_G is not None:
                torch.save(optimizer_G.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Gopt.pt'))
            if optimizer_D is not None:
                torch.save(optimizer_D.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Dopt.pt'))
            if train_Q:
                torch.save(optimizer_Q_to_QG.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Q_to_Gopt.pt'))            
            if train_Q and track_Q2 and optimizer_Q2_to_QG is not None:
                torch.save(optimizer_Q2_to_QG.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Q_to_Q2opt.pt'))

            if ('last_path_prefix' in locals()) or ('last_path_prefix' in globals()):
                os.system('rm '+last_path_prefix)

            last_path_prefix = os.path.join(logdir, 'epoch'+str(epoch)+'_step'+str(step)+'*')
