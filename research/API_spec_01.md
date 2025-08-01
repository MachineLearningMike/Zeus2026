### Requirements Specification of dataset

1. Assumption and notations

- data source <br>
  "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3" 
  <br>

- Normalized virtual dataset, mean, and std <br>
    Virtual dataset is a dataset kept in yoru mind and not on a physical memory or drive.
    The virtual dataset is a continuously and choronologically increasingly arranged sequence of frames, with the shape of (T = nTotalHours, C, H, W), where nTotalHours is the number of **hourly grids** that are available in our dataset.  Virtual dataset is normalized as follows:
        $$\bar{x_{c}} = \frac{1}{T*H*W} \sum_{t=0}^{T-1}\sum_{h=0}^{H-1}\sum_{w=0}^{W-1}x[t,c,h,w]$$        $$\sigma_{c} = \sqrt{\frac{1}{T*H*W} \sum_{t=0}^{T-1}\sum_{h=0}^{H-1}\sum_{w=0}^{W-1}(x[t,c,h,w] - \bar{x_{c}})^2}$$
        
        Virtual dataset x is normalized as follows:
            x = (x - x.mean(axis=(0,2,3), keepdims=True)) / x.std(axis=(0,2,3), keepdims=True)

        The following value should be kept for later use:
            mean = x.mean(axis=(0,2,3), keepdims=False)
            std = x.std(axis=(0,2,3), keepdims=False)

- data presentation from data_loader
        
        `
        D = [...]   # total days
        V = [...]   # total variables
        def data(d, v):         # return shape: (24, H, W)
            return the numpy array of the data for v and d.
        def var(d, v, mean):    # return shape (24, H, W)
            return np.mean( np.power(data(d,v) - mean, 2), axis=(0,1,2) )

        # This line will take a long time, reading thousands files.
        mean = [ [ np.mean(data(d,v), axis=(0,1,2)) for v in V ] for d in D ]
        mean = np.array(mean)               # shape: (D, V)
        mean = np.mean(mean_var, axis=0)        # shape: (V,)

        # This line will take a long time, reading thousands files.
        var = [ [ var(d, v, mean[v]) for v in V] for d in D ]
        var = np.array(var)                       # shape: (D, V)
        var = np.mean(var, axis=0)                # shape: (V,)
        std = np.power(var, 0.5)                  # shape: (V,)
        
        # For a given batch batch_x, of shape (B, T1, C, H, W), data_loader presents:
            mean = mean[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
            std = std[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
            batch_x = (batch_x - mean) / std        # this is presented by the data_loader.
        `
    
- Frame <br>
  A frame is a slice of normalized virtual dataset and has the following shape: <br>
    (C, H, W) = (6, 720, 1440)
    , where
    - C: the number of variables <br>
        We have currently 6 variables to predict. We restrict our ground truth to the same variables that we predict **for bravity**. Currently **C == 6**.<br>
        - 2m_temperature,
        - 10m_u_component_of_wind 
        - 10m_v_component_of_wind 
        - total_precipitation
        - 100m_u_component_of_wind
        - 100m_v_component_of_wind
        <br>
    - H: the number of latitudal grids <br>
        - The data source has H = 721, which comes from [-90, 90] / 0.25
        - We **remove** the south pole, and currently **H == 720**.
        <br>
    - W: the number of longitudal grids <br>
        - The data source has W = 1440, which comes from [-180, 180) / 0.25
        - We accept **W == 1440**.
        <br>  

- Frame Group, FG <br>
    A frame group is a continuously and choronologically increasingly arranged sequence of **frames**, with the shape of (FG, C, H, W), where FG is the pre-defined number of hourly grids. The starting frame of a frame group must be aligned with **0 o'clock** of a day. Currently **FG == 24**.

- Training data pair (X, Y) <br>
    For a training data pair (X, Y), X has continuously arranged T1 **frame groups**, packed in the shape of (T1 * FG, C, H, W), while Y has continuously arranged **varying** number of frame groups, packed in the shape of (T2 * FG, C, H, W). Y should immadiately follow X chronologically. Currently **T1 == 1**. It will increase in the future as we have more resources.

- Training data batch (batch_x, batch_y) of batch size B <br>
    - batch_x.shape = (B, T1 * FG, C, H, W) 
    - batch_y.shape = (B, T2 * FG, C, H, W)
    - (batch_x[n], batch_y[n]) constitutes a **training data pair**, for any applicable n.

- Trainig set, validation set, and test set <br>
    - The normalized virtual dataset has a total of around nTotalHours/FG frame groups.
    - We have a similar count of training data pairs.
    - A uniformly distributed 60% of the data pairs forms the trainig set
    - A uniformly distributed other 20% of the data pairs forms the validation set
    - A uniformly distributed remaing 20% of the data pairs forms the test set.

2. data_loader API <br>

- data_loader.prefetch(steps, batch, set) <br>

    data_loader asynchronously makes available a **training data batch** of size 'batch', where T2 = 'steps'. The data should sampled from training set, validation set, or test set, according to the value of 'set', which is one of 'trainig', 'validation', and 'test'. The available batches should put in to a FIFO queue maintained internally.
    
- data_loader.get() <br>

    data_loader waits until a batch of data pair is available in the internal FIFO queue, pops a pair from the queue, and returns it to the caller when called in the form of:

- data_loader.empty() <br>
    
    data_loader empties the FIFO queue.


3. tqdm prograss bar <br>

    data_loader should be able to work with tqdm as below: <br>


        pbar = zeus_tqdm(train_loader)

        new_T2 = env.get_next_steps()
        zeus_tqdm.prefetch(new_T2)        # Fetch one batch asynchronously`

        for batch_x, batch_y in pbar:         # Wait for the asynchronous .prefetch
            T2 = batch_y.shape[1] // self.config('FG')
            new_T2 = env.get_next_steps(); 
            zeus_tqdm.prefetch(new_T2)    # Fetch one batch asynchronously

            #... ... ...

            for step in range(T2):
                #... ... ...
                pbar.set_description('train loss: {:.4f}'.format(loss.item()))



