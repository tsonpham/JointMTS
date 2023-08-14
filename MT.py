import tensorflow as tf
import numpy as np
import pickle

def invert_for_mt_and_timeshift(objstats, learning_rate=2e-1, epochs=200, random_seed=None, 
                                M_sigma=1, T_sigma=1.5, fname=None):
    '''
    fname: filename to record history file
    '''
    ns = len(objstats)
    nc, ne, nt = objstats[0]['Gtensor'].shape
    ## gather data to creat Gtensor
    Gtensor = np.array([_['Gtensor'] for _ in objstats])
    Gmed = np.median(np.abs(Gtensor))
    Obs =     np.array([_['obsdata'] for _ in objstats])
    Omed = np.median(np.abs(Obs))
    ## preserve factor to scale back MT solution after inversion
    scale_factor = Omed / Gmed

    ########### tensorflow block, which will be run on GPU if available
    ## Fourier transforms of Gtensor and Obs waveforms 
    Gtensor   = tf.constant(Gtensor/Gmed, tf.float64)
    Gtensor_f = tf.signal.rfft(Gtensor, tf.constant([2*nt]))
    Obs       = tf.constant(Obs/Omed, tf.float64)
    Obs_f     = tf.signal.rfft(Obs, tf.constant([2*nt]))
    ## declare an optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    ## randomised initial solutions
    if random_seed is not None: np.random.seed(random_seed)
    M = tf.Variable(np.random.normal(0, M_sigma, (1, ne)))
    t = tf.Variable(np.random.normal(0, T_sigma, (ns, 1, 1)))

    ## frequency vector
    omega = tf.ones(Obs_f.shape, tf.float64) * tf.constant(np.fft.rfftfreq(2*nt)*2*np.pi, tf.float64)
    
    ## space holder to record loss and variable evolution
    history_loss, history_M, history_t = [], [], []
    for epoch in range(epochs):
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:
            ## observed data shifted instaneously by a time t
            Obs_f_real = tf.cos(omega*t)*tf.math.real(Obs_f) + tf.sin(omega*t)*tf.math.imag(Obs_f)
            Obs_f_imag = tf.cos(omega*t)*tf.math.imag(Obs_f) - tf.sin(omega*t)*tf.math.real(Obs_f)
            ## different of prediction and shifted observation
            Diff_real = tf.squeeze(M @ tf.math.real(Gtensor_f)) - Obs_f_real
            Diff_imag = tf.squeeze(M @ tf.math.imag(Gtensor_f)) - Obs_f_imag
            ## mean squared error
            loss_value = tf.reduce_mean(tf.square(Diff_real) + tf.square(Diff_imag))
            
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, [M, t])
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, [M, t]))
        # Save history for inspection
        history_loss.append(float(loss_value))
        history_M.append(np.squeeze(M.numpy())*scale_factor)
        history_t.append(np.squeeze(t.numpy()))

    ## return a dictionary recording history evoluiton of loss function and its variables
    idx = np.argmin(history_loss)
    Msol = history_M[idx] # MT solution is in North, East, Down coordinate system
    tshift = history_t[idx]
    history = {'loss': np.array(history_loss), 'M': np.array(history_M), 
               't': np.array(history_t), 'stations': objstats,
               'Mopt': Msol, 'topt': tshift}
    if fname is None:
        return history
    else:
        with open(fname, 'wb') as fp: pickle.dump(history, fp)
        
def invert_for_mt(objstats, learning_rate=2e-1, epochs=200, random_seed=None, 
                  M_sigma=1, fname=None):
    '''
    fname: filename to record history file
    '''
    ns = len(objstats)
    nc, ne, nt = objstats[0]['Gtensor'].shape
    ## gather data to creat Gtensor
    Gtensor = np.array([_['Gtensor'] for _ in objstats])
    Gmed = np.median(np.abs(Gtensor))
    Obs =     np.array([_['obsdata'] for _ in objstats])
    Omed = np.median(np.abs(Obs))
    ## preserve factor to scale back MT solution after inversion
    scale_factor = Omed / Gmed

    ########### tensorflow block, which will be run on GPU if available
    ## Fourier transforms of Gtensor and Obs waveforms 
    Gtensor   = tf.constant(Gtensor/Gmed, tf.float64)
    Gtensor_f = tf.signal.rfft(Gtensor, tf.constant([2*nt]))
    Obs       = tf.constant(Obs/Omed, tf.float64)
    Obs_f     = tf.signal.rfft(Obs, tf.constant([2*nt]))
    ## declare an optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    ## randomised initial solutions
    if random_seed is not None: np.random.seed(random_seed)
    M = tf.Variable(np.random.normal(0, M_sigma, (1, ne)))
    t = tf.constant(np.zeros((ns, 1, 1)))

    ## frequency vector
    omega = tf.ones(Obs_f.shape, tf.float64) * tf.constant(np.fft.rfftfreq(2*nt)*2*np.pi, tf.float64)
    
    ## space holder to record loss and variable evolution
    history_loss, history_M, history_t = [], [], []
    for epoch in range(epochs):
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:
            ## observed data shifted instaneously by a time t
            Obs_f_real = tf.cos(omega*t)*tf.math.real(Obs_f) + tf.sin(omega*t)*tf.math.imag(Obs_f)
            Obs_f_imag = tf.cos(omega*t)*tf.math.imag(Obs_f) - tf.sin(omega*t)*tf.math.real(Obs_f)
            ## different of prediction and shifted observation
            Diff_real = tf.squeeze(M @ tf.math.real(Gtensor_f)) - Obs_f_real
            Diff_imag = tf.squeeze(M @ tf.math.imag(Gtensor_f)) - Obs_f_imag
            ## mean squared error
            loss_value = tf.reduce_mean(tf.square(Diff_real) + tf.square(Diff_imag))
            
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, [M])
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, [M]))
        # Save history for inspection
        history_loss.append(float(loss_value))
        history_M.append(np.squeeze(M.numpy())*scale_factor)
        history_t.append(np.squeeze(t.numpy()))

    ## return a dictionary recording history evoluiton of loss function and its variables
    idx = np.argmin(history_loss)
    Msol = history_M[idx] # MT solution is in North, East, Down coordinate system
    tshift = history_t[idx]
    history = {'loss': np.array(history_loss), 'M': np.array(history_M), 
               't': np.array(history_t), 'stations': objstats,
               'Mopt': Msol, 'topt': tshift}
    if fname is None:
        return history
    else:
        with open(fname, 'wb') as fp: pickle.dump(history, fp)
