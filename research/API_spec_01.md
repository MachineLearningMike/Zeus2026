### Requirements Specification of dataset

1. Assumption and notations

- data source
  "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3" 
  <br>
    
- Snapshot
  A snapshot of data has the following dimensions:
    (C, H, W) = (6, 720, 1440)
    , where
    - C: the number of variables. 
        We have currently 6 variables to predict. We restrict the varibles in our ground truth to the same variables that we predict. Currently **C == 6**.<br>
        - 2m_temperature,
        - 10m_u_component_of_wind 
        - 10m_v_component_of_wind 
        - total_precipitation
        - 100m_u_component_of_wind
        - 100m_v_component_of_wind
        <br>
    - H: the number of latitudal grids
        - The data source has H = 721, which comes from [-90, 90] / 0.25
        - We **remove** the south pole, and currently **H == 720**.
        <br>
    - W: the number of longitudal grids
        - The data source has W = 1440, which comes from [-180, 180) / 0.25
        - We accept **W == 1440**.
        <br>

- Virtual dataset
    The virtual dataset is a continuous, choronologically increasingly ordered, sequence of snapshots, with the dimension of (nTimeGrids, C, H, W), where nTimeGrids is the number of time grids that are available in our trainig dataset.

- Snapshot Group, SG
    A snapshot group is a continuous, choronologically increasingly ordered, sequence of snapshots, with the dimension of (SG, C, H, W), where SG is the pre-defined number of time grids. The start time of a snapshot group must be aligned with 0 o'clock of a day. Currently **SG == 24**.

- Training data pair (X, Y), T1
    For a training data pair (X, Y), X has continuous T1 snapshot groups, packed in the form of (T1 * SG, C, H, W), while Y has continuous **varying** number of snapshot groups, packed in the form of (T2 * SG, C, H, W). Y should immadiately follow X chronologically.
    Currently **T1 == 1**.


2. data_loader API

- data_loader.prefetch(steps, batch)

    data_loader asynchronously makes 'batch' count of training data pairs (X, Y) available, where Y has 'steps' snapshot groups. The pair is pushed into the FIFO queue that is maintained internally. The batch should have batch_x of dimensions (batch, T1*SG, C, H, W), and batch_y of dimentions (batch, T2*SG, C, H, W), where T2 == steps.

    for any 0 <= n <= batch, (batch_x[n], batch_y[n]) should be a **trainig data pair**.

    <br>

- data_loader.get()

    data_loader waits until a batch of data pair is available in the internal FIFO queue, pops a pair from the queue, and returns it to the caller when called in the form of:

    `for batch_x, batch_y in train_pbar:`



3. tqdm prograss bar

    data_loader should be able to work with tqdm as below:

    `train_pbar = tqdm(env.train_loader)`
    `for batch_x, batch_y in train_pbar:`

