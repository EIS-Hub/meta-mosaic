import os

# Code to accelerate JAX on TPU if available
if 'COLAB_TPU_ADDR' in os.environ:
    import jax.tools.colab_tpu
    jax.tools.colab_tpu.setup_tpu()
    print('Connected to TPU')

import time
import jax
import jax.numpy as jnp
import jax.random as random
from jax import vmap, jit, value_and_grad
from jax.lax import scan
from jax.nn import log_softmax
from jax.example_libraries import optimizers
from jax.tree_util import tree_map, tree_leaves
import numpy as np 
import wandb
import pickle

from network import lif_forward
from utils import get_h5py_files, calc_mosaic_stats, sparser, pruner, add_noise
from dataloader import dataset_numpy, NumpyLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# Training function
def train(key, batch_size, n_way, k_shot, n_inp, n_rec, n_out, n_inner_loop, thr_rec, thr_out, tau_rec, 
          lr_in, lr_out, tau_out, lr_drop, w_gain, n_epochs, patience, min_delta, lr_reduction_factor, target_fr, lambda_fr, sparser, beta, prune_start_step, prune_thr, n_core, W_mask, noise_start_step, noise_std):
    
    key, key_model = random.split(key, 2)
    
    @jit
    def l2_loss(x, alpha):
        return alpha * (x ** 2).mean()

    def net_step(net_params, x_t):
        ''' Single time step network inference (x_t => yhat_t)
        '''
        net_params, [z_rec, v_out] = lif_forward(net_params, x_t)
        return net_params, [z_rec, v_out]
    
    def predict(weights, X):
        _, net_const, net_dyn = param_initializer(key, n_inp, n_rec, n_out, thr_rec,
                                                  thr_out, tau_rec, tau_out, w_gain)
        
        _, [z_rec, v_out] = scan(net_step, [weights, net_const, net_dyn], X, length=100)
        Yhat = log_softmax(jnp.max(v_out, axis=0))
        return Yhat, [z_rec, v_out]

    v_predict = vmap(predict, in_axes=(None, 0))

    @jit
    def inner_loss(weights, X, Y):
        key_l, key_o = random.split(key, 2)
        weights = jax.lax.cond(epoch >= noise_start_step, 
                      lambda weights, key_l : add_noise(weights, key_l, noise_std),
                      lambda weights, key_l : weights,
                      weights, key)
        
        Yhat, [z_rec, v_out] = v_predict(weights, X)
        Y = one_hot(Y, n_out)
        num_correct = jnp.sum(jnp.equal(jnp.argmax(Yhat, 1), jnp.argmax(Y, 1)))
        loss_ce = -jnp.mean(jnp.sum(Yhat * Y, axis=1, dtype=jnp.float32))
        fr0 = 10*jnp.mean(jnp.sum(z_rec, 1))  
        L_fr = jnp.mean(target_fr-fr0) ** 2
        loss = loss_ce + L_fr * lambda_fr

        return loss, [num_correct, fr0, loss_ce, L_fr]

    @jit
    def outer_loss(weights, X, Y):
        key_l, key_o = random.split(key, 2)
        weights = jax.lax.cond(epoch >= noise_start_step, 
                      lambda weights, key_l : add_noise(weights, key_l, noise_std),
                      lambda weights, key_l : weights,
                      weights, key)
        
        Yhat, [z_rec, v_out] = v_predict(weights, X)
        Y = one_hot(Y, n_out)  
        num_correct = jnp.sum(jnp.equal(jnp.argmax(Yhat, 1), jnp.argmax(Y, 1)))
        loss_ce = -jnp.mean(jnp.sum(Yhat * Y, axis=1, dtype=jnp.float32))
        loss_sp =  sparser(weights[1], W_mask)
        fr0 = 10*jnp.mean(jnp.sum(z_rec, 1))
        L_fr = jnp.mean(target_fr-fr0) ** 2 
        loss = loss_ce + loss_sp + L_fr * lambda_fr

        loss += sum(
            l2_loss(w, alpha=0.01) 
            for w in tree_leaves(weights)
        )

        Yhat = jnp.argmax(Yhat, axis=1)
        return loss, [num_correct, fr0, loss_ce, L_fr, loss_sp, Yhat] #[num_correct, loss]
    
    def param_initializer(key, n_inp, n_rec, n_out, thr_rec, thr_out, tau_rec, tau_out, w_gain):
        ''' Initialize parameters
        '''
        key_inp, key_rec, key_out, key = random.split(key, 4)
        alpha = jnp.exp(-1e-3/tau_rec) 
        kappa = jnp.exp(-1e-3/tau_out)

        inp_weight = random.normal(key_inp, [n_inp, n_rec]) * w_gain
        rec_weight = random.normal(key_rec, [n_rec, n_rec]) * w_gain
        out_weight = random.normal(key_out, [n_rec, n_out]) * w_gain

        bias = jnp.zeros(n_rec)

        neuron_dyn = [jnp.zeros(n_rec), jnp.zeros(n_rec), jnp.zeros(n_out), jnp.zeros(n_out)]
        net_params = [[inp_weight, rec_weight, bias, out_weight], [thr_rec, thr_out, alpha, kappa], neuron_dyn]
        return net_params
    
    @jit
    def update_in(theta, sX, sY):
        # Calculate gradients 
        value, grads_in = value_and_grad(inner_loss, has_aux=True)(theta, sX, sY)

        # Inner update: SGD
        def inner_sgd_fn(g, p):
            p = p - lr_in * g
            return p
        
        updated_weights = tree_map(inner_sgd_fn, grads_in[3], theta[3])

        metrics ={'inner_num_correct': value[1][0],
                  'inner_loss': value[0],
                  'theta-inp': theta[0],
                  'theta-rec': theta[1],
                  'bias': theta[2],
                  'theta-out': theta[3],
                  'grad-in-inp': grads_in[0],
                  'grad-in-rec': grads_in[1],
                  'grad-in-bias': grads_in[2],
                  'grad-in-out': grads_in[3],
                  'inner_firing_rate_rec': value[1][1],
                  'inner_cross_entropy_loss': value[1][2],
                  'inner_firing_rate_loss': value[1][3],
                  'dW-out': updated_weights - theta[3]}
            
        return [theta[0], theta[1], theta[2], updated_weights], metrics

    @jit
    def maml_loss(theta, sX, sY, qX, qY):
        u_weights = theta

        for i in range(n_inner_loop):
            u_weights, metrics = update_in(u_weights, sX, sY)

        return outer_loss(u_weights, qX, qY), metrics

    def batched_maml_loss(theta, sX, sY, qX, qY):
        [task_losses, [outer_num_correct, fr0, loss_ce, L_fr, loss_sp, Yhat]], metrics = vmap(maml_loss, in_axes=(None, 0, 0, 0, 0))(theta, sX, sY, qX, qY)
        metrics['outer_num_correct'] = outer_num_correct
        metrics['outer_firing_rate_rec'] = jnp.mean(fr0)
        metrics['outer_cross_entropy_loss'] = jnp.mean(loss_ce)
        metrics['outer_firing_rate_loss'] = jnp.mean(L_fr)
        metrics['outer_sparsity_loss'] = jnp.mean(loss_sp)

        return jnp.mean(task_losses), metrics
    
    @jit
    def update_out(i, opt_state, weight, sX, sY, qX, qY):
        (L, metrics), grads_out = value_and_grad(batched_maml_loss, has_aux=True)(weight, sX, sY, qX, qY)

        sum_keys = ["inner_num_correct", "outer_num_correct"]
        mean_keys = ["inner_firing_rate_rec","outer_firing_rate_rec","outer_cross_entropy_loss", "outer_sparsity_loss","outer_firing_rate_loss","inner_cross_entropy_loss","inner_firing_rate_loss"]
        metrics = {k:jnp.sum(v, axis=0) if k in sum_keys else jnp.mean(v) if k in mean_keys else jnp.mean(v, axis=0) for (k,v) in metrics.items()}

        metrics['outer_loss'] = L

        opt_state = opt_update(i, grads_out, opt_state)

        return get_params(opt_state), opt_state, metrics

    def one_hot(x, n_class):
        return jnp.array(x[:, None] == jnp.arange(n_class), dtype=jnp.float32)
    
    #LSUV initialization
    '''
    Implementation of Layer-sequential unit-variance (LSUV) initialization
    https://doi.org/10.48550/arXiv.1511.06422
    '''
    def data_driven_init(state, x):
        tgt_mean =-0.52
        tgt_var =1.
        mean_tol =.1
        var_tol =.1
        done = False
        
        net_state = net_step(state, x)[0]

        while not done:
            done = True
            
            net_state = net_step(net_state, x)[0] #Data pass without weight updates
            
            inp_weight, rec_weight, bias, out_weight = net_state[0]
            weights = [inp_weight, rec_weight, out_weight]
            # bias_arr = [bias]

            neuron_dyn = net_state[2]
            V = [neuron_dyn[0], neuron_dyn[2]]

            v=0    
            for i in range(0,len(weights)):
                if i == len(weights)+1:
                    v+=1
                done=True

                st = V[v] #Membrane potential of the current layer
                kernel = jnp.array(weights[i])

                st = jnp.array(st)
                
                ## Adjusting weights to unit variance
                if jnp.abs(jnp.var(st) - tgt_var) > var_tol:
                    kernel = jnp.multiply(jnp.sqrt(tgt_var)/jnp.sqrt(jnp.maximum(jnp.var(st),1e-2)), kernel)
                    weights[i] = kernel
                    done *= False
                else:
                    done *=True

                if v == 1:
                    continue

                ## Adjusting bias to target mean
                bias = jnp.array(bias)

                if jnp.abs(jnp.mean(st) - tgt_mean) > mean_tol:
                    bias = jnp.subtract(bias, .15*(jnp.mean(st) - tgt_mean))
                    done *= False
                else:
                    done *= True

            inp_weight, rec_weight, out_weight = weights
            net_state = [[inp_weight, rec_weight, bias, out_weight], state[1], state[2]]

        return net_state

     # Preprocess data
    train_shd_file, test_shd_file = get_h5py_files()
    
    # Training dataset
    # Labels to exclude during training
    excluded_labels = [11, 12, 4, 3, 0]

    #Creating Meta-train dataloader
    meta_train_ds = dataset_numpy(train_shd_file['spikes'], 
                                train_shd_file['labels'], 
                                train_shd_file['extra'], 
                                n_way=n_way, k_shot=k_shot, 
                                excluded_labels=np.asarray(excluded_labels))

    train_dl = NumpyLoader(meta_train_ds, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=True)

    #Creating Meta-test dataloader
    meta_test1_ds = dataset_numpy(test_shd_file['spikes'], 
                                test_shd_file['labels'], 
                                test_shd_file['extra'], 
                                n_way=n_way, k_shot=k_shot, 
                                excluded_labels=np.asarray([]))

    test1_dl = NumpyLoader(meta_test1_ds, batch_size=1, num_workers=0, drop_last=False, shuffle=True)

    # Params for reduce learning rate on plateau
    # piecewise_lr = optimizers.piecewise_constant([lr_drop], [lr_out, lr_out/10])
    # nr_reduce = 1

    opt_init, opt_update, get_params = optimizers.adam(step_size=lr_out)
    net_state = param_initializer(key, n_inp, n_rec, n_out, thr_rec, \
                                     thr_out, tau_rec, tau_out, w_gain)

    sX, sY, qX, qY = next(iter(train_dl))
    for X in sX:    
        net_state = data_driven_init(net_state, X)
    
    weight = net_state[0]
    rec_W = weight[1]
    opt_state = opt_init(weight)

    # Start Meta-training
    train_loss = []; t = time.time(); step = 0; best_acc = 80;
    for epoch in range(n_epochs):
        t0 = time.time()
        inner_acc = 0
        outer_acc = 0
        n = 0
        loss = []

        for b_idx, (sX, sY, qX, qY) in enumerate(iter(train_dl)):
            weight[1] = rec_W
            
            weight, opt_state, metrics = update_out(step, opt_state, weight, sX, sY, qX, qY)
            rec_W = pruner(step, prune_start_step, prune_thr, weight[1])
        
            step += 1
            loss.append(metrics["outer_loss"])
            inner_acc += metrics['inner_num_correct']
            outer_acc += metrics['outer_num_correct']
            n += (batch_size*n_way*k_shot)

        weight[1] = rec_W
        curr_loss = np.average(loss)
        train_loss.append(curr_loss)

        metrics['train_n'] = n
        metrics['inner_num_correct'] = inner_acc
        metrics['outer_num_correct'] = outer_acc
        metrics['inner_accuracy'] = (inner_acc*100)/n
        metrics['outer_accuracy'] = (outer_acc*100)/n

        train_acc = ((inner_acc+outer_acc)*100)/(n*2)

        plt.matshow(np.absolute(weight[1])>0)
        wandb.log({'Weight Matrix (W1 > 0)': plt, 'Training accuracy': train_acc})
        plt.close()
        occupancy = calc_mosaic_stats(weight[1], W_mask, beta)
        run.log({'Mosaic occupancy': wandb.Table(columns=[str(i)+' HOP' for i in range(len(occupancy))], data=[occupancy])})  

        print(f'Epoch: {epoch} - Inner Loss: {metrics["inner_loss"]:.3f} - Outer Loss: {metrics["outer_loss"]:.3f} - inner_NC: {inner_acc} - outer_NC: {outer_acc} - Tot: {n} - Inner Accuracy: {(inner_acc*100)/n:.3f} - Outer Accuracy: {(outer_acc*100)/n:.3f} - Time : {(time.time()-t0):.3f} s')
        print(f'Inner loop: Firing rate rec: {metrics["inner_firing_rate_rec"]:.2f} - Outer loop: Firing rate rec: {metrics["outer_firing_rate_rec"]:.2f}')
        
        # Code implementing reduce learning rate on plateau
        # if len(train_loss) > patience and all(abs(curr_loss - prev_loss) < min_delta for prev_loss in train_loss[-patience:]):
        #     print("Reducing learning rate...")
        #     # for prev_loss in train_loss[-patience:]:
        #     #     print(curr_loss - prev_loss, end=" ")
        #     new_lr = lr_out*np.power(lr_reduction_factor, nr_reduce)
        #     nr_reduce += 1
        #     opt_init, opt_update, get_params = optimizers.adam(step_size=new_lr)
        #     opt_state = get_params(opt_state)
        #     opt_state = opt_init(opt_state)
        #     print("Current loss:", curr_loss)
        #     print("Learning rate reduced to ", new_lr)

        if epoch % 10 == 0:
            train_acc = ((inner_acc+outer_acc)*100)/(n*2)
            print(f'Epoch: {epoch} - Loss: {metrics["outer_loss"]:.3f} - Time : {(time.time()-t0):.3f} s' +
                f' - Training acc: {train_acc:.2f}')
            
            if ((outer_acc*100)/n) > best_acc:
                print("Saving model...")
                best_acc = (outer_acc*100)/n
                # d = datetime.now().strftime("%d%B")
                pickle.dump(opt_state, open(os.path.join('./models/', "checkpoint.pkl"), "wb"))
                print("Model saved.")

            # Meta-testing 1
            wb_metrics_1 = {}
            t0 = time.time()
            n = 0
            outer_num_correct = 0
            inner_num_correct = 0

            true_labels = []
            predicted_labels = []

            o_loss, i_loss, o_loss_ce, i_loss_ce, o_loss_fr, i_loss_fr, o_loss_sp = [], [], [], [], [], [], []
            o_fr_rec, i_fr_rec = [], []

            theta = weight

            for b_idx, (sX, sY, qX, qY) in enumerate(iter(test1_dl)):
                n+=len(sY)*len(sY[0])
                
                # print("INNER LOOP")
                loss, [num_correct, fr0, loss_ce, L_Fr] = vmap(inner_loss, in_axes=(None, 0, 0))(theta, sX, sY)
                i_fr_rec.append(jnp.mean(fr0))
                i_loss_ce.append(jnp.mean(loss_ce))
                i_loss_fr.append(jnp.mean(L_Fr))
                # i_fr_out.append(jnp.mean(fr1))
                i_loss.append(jnp.mean(loss))
                inner_num_correct += jnp.sum(num_correct)
                
                # n-inner loop updates during Meta-testing
                u_weights = theta
                
                for i in range(n_inner_loop):
                    u_weights, wb_metrics = vmap(update_in, in_axes=(None, 0, 0))(u_weights, sX, sY)
                    u_weights = tree_map(lambda W: jnp.mean(W, axis=0), u_weights)

                loss, [num_correct, fr0, loss_ce, L_Fr, loss_sp, Yhat] = vmap(outer_loss, in_axes=(None, 0, 0))(u_weights, qX, qY)

                true_labels.append(jnp.ravel(jnp.asarray(qY)))
                predicted_labels.append(jnp.ravel(Yhat))

                o_fr_rec.append(jnp.mean(fr0))
                o_loss_ce.append(jnp.mean(loss_ce))
                o_loss_fr.append(jnp.mean(L_Fr))
                o_loss_sp.append(jnp.mean(loss_sp))
                o_loss.append(jnp.mean(loss))
                outer_num_correct += jnp.sum(num_correct)

            true_labels = jnp.ravel(jnp.asarray(true_labels))
            predicted_labels = jnp.ravel(jnp.asarray(predicted_labels))

            jax.debug.print("true_labels: {true_labels}", true_labels=np.array(true_labels).shape)
            jax.debug.print("predicted_labels: {predicted_labels}", predicted_labels=np.array(predicted_labels).shape)

            true_labels = np.asarray(true_labels)
            predicted_labels = np.asarray(predicted_labels)

            excluded_indices = np.isin(true_labels, excluded_labels)

            excl_tot = np.sum(excluded_indices)
            excl_nc = np.sum(excluded_indices & (true_labels == predicted_labels))

            tot = len(true_labels) - excl_tot
            nc = np.sum(~excluded_indices & (true_labels == predicted_labels))

            conf_matrix = confusion_matrix(true_labels, predicted_labels)

            plt.matshow(conf_matrix)
            wandb.log({'Confusion Matrix': plt})

            wb_metrics_1['Excluded_accuracy_test_1'] = (excl_nc*100)/excl_tot
            wb_metrics_1['Accuracy_test_1'] = (nc*100)/tot
            wb_metrics_1['Outer_accuracy_test_1'] = (outer_num_correct*100)/n
            wb_metrics_1['Inner_accuracy_test_1'] = (inner_num_correct*100)/n
            wb_metrics_1['Outer_loss_test_1'] = np.average(o_loss)
            wb_metrics_1['Inner_loss_test_1'] = np.average(i_loss)
            wb_metrics_1['Outer_fr_rec_test_1'] = np.average(o_fr_rec)
            wb_metrics_1['Inner_fr_rec_test_1'] = np.average(i_fr_rec)
            wb_metrics_1['Outer_loss_ce_test_1'] = np.average(o_loss_ce)
            wb_metrics_1['Inner_loss_ce_test_1'] = np.average(i_loss_ce)
            wb_metrics_1['Outer_loss_fr_test_1'] = np.average(o_loss_fr)
            wb_metrics_1['Inner_loss_fr_test_1'] = np.average(i_loss_fr)
            wb_metrics_1['Outer_loss_sp_test_1'] = np.average(o_loss_sp)

            print(f'Meta-testing #1 (Seen classes) - Inner Loss: {wb_metrics_1["Inner_loss_test_1"]:.3f} - Outer Loss: {wb_metrics_1["Outer_loss_test_1"]:.3f} - Inner accuracy: {(inner_num_correct*100)/n:.3f} - Outer accuracy: {(outer_num_correct*100)/n:.3f} - Accuracy Seen: {((nc*100)/tot):.3f} - Accuracy Unseen: {((excl_nc*100)/excl_tot):.3f} - Time : {(time.time()-t0):.3f} s')
            print(f'Inner loop: Firing rate rec: {wb_metrics_1["Inner_fr_rec_test_1"]:.2f} - Outer loop: Firing rate rec: {wb_metrics_1["Outer_fr_rec_test_1"]:.2f}')


            try:
                wandb.log(wb_metrics_1)
            except Exception as e:
                print("ERROR:", e)
                continue
        
    t_end = time.time()
    print(f'Training completed in {(t_end-t):.2f} seconds ({(n_epochs/(t_end-t)):.2f} epoch/s)')
    print('Meta-training completed.')

    return train_loss


if __name__ == '__main__':
    from jax import random
    import argparse
    import wandb
    from utils import sparser, generate_mosaic_mask

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=3, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for batched meta-training')
    parser.add_argument('--n_way', type=int, default=5, help='Number of tasks')
    parser.add_argument('--k_shot', type=int, default=10, help='Number of samples per task')
    parser.add_argument('--n_epochs', type=int, default=5000, help='Number of iterations')
    parser.add_argument('--n_inp', type=int, default=256, help='Number of input neurons')
    parser.add_argument('--n_rec', type=int, default=256, help='Number of recurrent neurons')
    parser.add_argument('--n_out', type=int, default=20, help='Number of output neurons')
    parser.add_argument('--n_inner_loop', type=int, default=1, help='Number of inner loop gradient updates')
    parser.add_argument('--thr_rec', type=int, default=2, help='Threshold of recurrent neurons')
    parser.add_argument('--thr_out', type=int, default=1, help='Threshold of output neurons')
    parser.add_argument('--tau_rec', type=float, default=30e-3, help='Membrane time constant')
    parser.add_argument('--tau_out', type=float, default=30e-3, help='Output time constant')
    parser.add_argument('--w_gain', type=float, default=5e-1, help='base weight gain for init')
    parser.add_argument('--lr_in', type=float, default=1, help='Learning rate for the inner loop')
    parser.add_argument('--lr_out', type=float, default=1e-4, help='Learning rate for the output loop')
    parser.add_argument('--target_fr', type=float, default=8, help='Target firing rate')
    parser.add_argument('--lr_drop', type=int, default=12000, help='The step number for dropping the learning rate')
    parser.add_argument('--lambda_fr', type=float, default=0.01, help='Regularization parameter for the firing rate')
    parser.add_argument('--beta', type=float, default=8, help='Exponential scaling factor for the cost of increasing number of hops')
    parser.add_argument('--patience', type=int, default=5, help='Number of iterations before reducing learning rate on plateau')
    parser.add_argument('--min_delta', type=float, default=0.01, help='Minimum difference allowed in a plateau')
    parser.add_argument('--lr_reduction_factor', type=float, default=0.1, help='Factor of reduction for learning rate')
    parser.add_argument('--prune_str', type=int, default=10, help='Iteration timestep for pruning to start')
    parser.add_argument('--prune_thr', type=float, default=0.0005, help='Weight threshold for pruning')
    parser.add_argument('--n_core', type=int, default=64, help='Number of cores in Mosaic')
    parser.add_argument('--noise_str', type=int, default=10, help='Iteration timestep for noise injection to start')
    parser.add_argument('--noise_std', type=float, default=0.05, help='Std of noise added on inference')
    args = parser.parse_args()

    # Initialize wandb: Please use your own API key and project name
    run = wandb.init(project='meta-mosaic-MAML(Small-world| Rectified)', config=args)
    wandb.config.update(args)

    train_loss = train(key=random.PRNGKey(args.seed), 
                    batch_size=args.batch_size,
                    n_way=args.n_way,
                    k_shot=args.k_shot, 
                    n_inp=args.n_inp,
                    n_rec=args.n_rec,
                    n_out=args.n_out,
                    n_inner_loop=args.n_inner_loop,
                    thr_rec=args.thr_rec,
                    thr_out=args.thr_out,
                    tau_rec=args.tau_rec,
                    lr_in=args.lr_in,
                    lr_out=args.lr_out,
                    tau_out=args.tau_out,
                    lr_drop=args.lr_drop,
                    w_gain=args.w_gain,
                    n_epochs=args.n_epochs,
                    patience=args.patience,
                    min_delta=args.min_delta,
                    lr_reduction_factor=args.lr_reduction_factor,
                    target_fr=args.target_fr,
                    lambda_fr=args.lambda_fr,
                    sparser=sparser,
                    beta=args.beta,
                    prune_start_step=args.prune_str, 
                    prune_thr=args.prune_thr,
                    n_core=args.n_core,
                    W_mask=generate_mosaic_mask(args.n_rec, args.n_core, args.beta),
                    noise_start_step=args.noise_str,
                    noise_std=args.noise_std)