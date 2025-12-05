class Train_6_hourly():
    def __init__(self):
        pass

    def train_one_batch(snapshots, nCascades, model_dom, fn_take_substep, device):
        #------- Mike1 algorithm, a 6-hourly training algorithm that mitigates the bottleneck on data pipeline ------
        # It takes 528 training substeps, with varing cascade steps, with 2 + 32 raw data X.
        # The distribution of cascase steps is: 
        # [32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        # = [1-cascade's training steps, 2- ..., ... 32-cascade's trainig steps]
        # 528 * 0.2 sec == 106 sec. The data pipeline should load 25GB (read below) of data withinn 106 sec, 250MB/s.
        # 250MB/s in 4 drives: 60MB/s
        # The GPU thread(s) will no longer wait for data threads to finish loading a batch of training data, saving GPU cost.
        # We might not need an 4x expensive NVEm drive.
        # 32 * 6 = 8 days * 24 hours, it trains to forecast 8 future days
        # The maximum number of data X's that are simultaneously on cuda is 66,
        # At most 66 * 4MB * 95 variables = 25GB of GPU memory is taken by data
        # The data pipeline may simultaneously have at most 2 * (2 + 32) = 68 raw X's, taking 25GB of cpu-side memory.
        # Quite feasible on a 100GB-GPU virtual machine.
        # 528 training substeps with 34 raw X's, ... might allow multi-GPU training, although GPU-loading of data may conflict a lot.
        
        assert len(snapshots) == nCascades + 2
        assert len(snapshots) >= 2, "length of Xs should be >= 2"
        X0, X1 = snapshots[0], snapshots[1]; X0.to(device); X1.to(device); snapshots = snapshots[2:]
        oldQ = [(X0, X1)]

        assert len(snapshots) > 0, "length of Xs should be >= 3"

        nSubsteps = 0
        nSubsteps_by_cascade = [0] * nCascades
        step_loss = []; step_metrics = []
        while len(snapshots) > 0:  # loops at least once
            # State: 
            # len(Xs) > 0, len(oldQ) > 0, [x0.device == device and x1.device == device for (x0, x1) in oldQ]
            # 2 * len(oldQ) count of X is stored in oldQ in device, together with model parameters.
            # oldQ[0] stores raw X's
                
            Y = snapshots[0]; Y.to(device); snapshots = snapshots[1:]
            newQ = [(oldQ[0][1], Y)]  # assert len(newQ) > 0        # newQ[0] stores raw X's

            #-------------------- State check, no-remove ------------------
            # nXOnDevice = 0
            # assert len([x for x in snapshots if x.device == device]) == 0, "Some of Xs are on device"
            # assert len(oldQ) > 0, "oldQ is empty"
            # check = [(x0.device == device or x1.device is None) and (x1.device == device or x1.device is None) for (x0, x1) in oldQ]
            # assert all(check), "Some pairs in oldQ are not on device"
            # nXOnDevice += 2 * len(oldQ)
            # assert len(newQ) == 1
            # check = [(x0.device == device or x1.device is None) and (x1.device == device or x1.device is None) for (x0, x1) in newQ]
            # assert all(check), "The only pair in newQ are not on device"
            # nXOnDevice += 2
            # print(f"n of X that are on device: {nXOnDevice}, accum substeps taken: {nSubsteps}")

            substep_loss = []; substep_metrics = []
            while len(oldQ) > 0:     # loops at least once
                X0, X1 = oldQ[0]; oldQ = oldQ[1:]    # assert X.device == device
                cascade_step = len(newQ)
                step_result, loss, metrics = fn_take_substep(model_dom, X0, X1, Y, cascade_step) # forward, backward, and optimizer.step
                nSubsteps += 1
                nSubsteps_by_cascade[cascade_step-1] += 1
                # assert (step_result.device == device or step_result.device is None)

                newQ.append((X1, step_result))                      # +1 X
                X0.delete()  # del all and clean cuda immediately   # -1 X
                
                substep_loss.append(loss)
                substep_metrics.append(metrics)
                
            step_loss.append(substep_loss)
            step_metrics.append(substep_metrics)

            # state: len(oldQ) == 0, len(oldQ) > 0
            oldQ = newQ
        print("nSubsteps", nSubsteps)
        print("nSubsteps_by_cascade", nSubsteps_by_cascade)
        assert nSubsteps == int((nCascades+1)*(nCascades)/2)


if __name__ == "__main__":

    from copy import copy
    def take_substep(model_dom, X0, X1, Y, cascade_step):
        _Y = copy(Y); _Y.fresh = False; _Y.device = X0.device
        return _Y, 1.0, [2.0, 3.0]

    from zeus_dataset_Snapshot import Snapshot
    from datetime import datetime, timedelta, timezone

    nCascades = 32
    snapshots = [Snapshot(dt.year, dt.month, dt.day, dt.hour, meteo=None, device='cpu')
                 for dt in [datetime.now(timezone.utc)+timedelta(days=-10, hours=n) for n in range(nCascades+2)]]
    model_dom = {}
    device = "cuda"
        
    Train_6_hourly.train_one_batch(snapshots, nCascades, model_dom, take_substep, device)