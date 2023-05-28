import torch
import numpy as np
from collections import defaultdict


#Find point with the lowest lower bound
def calculate_y_i(sortedYK,K):
    y_i = (sortedYK[0][0] + sortedYK[1][0])*0.5 - 0.5*(sortedYK[1][1] - sortedYK[0][1]) /  K
    return y_i

#Compute two new lower bounds
def calculate_z_i(sortedYK,K,y_i_w):
    z_left = (y_i_w[1] + sortedYK[0][1] - K * (y_i_w[0] - sortedYK[0][0])) * 0.5
    z_right = (y_i_w[1] + sortedYK[1][1] - K * (sortedYK[1][0] - y_i_w[0])) * 0.5
    # print(z_left == z_right)
    return z_left, z_right

class Recorder:
    def __init__(self,
                 nb_var,
                 coefs,
                 bound,
                 bound_error,
                 lip_const,
                 max_iter_per_dim,
                 max_feval):
        self.nb_var = nb_var
        self.coefs = coefs
        assert len(self.coefs) == self.nb_var
        self.lip_constant = lip_const * np.ones(self.nb_var)
        self.bound_error = bound_error * np.ones(self.nb_var)
        self.lower_bound = np.array([item[0] for item in bound])
        self.upper_bound = np.array([item[1] for item in bound])

        self.max_iter_per_dim = np.ones(self.nb_var) * max_iter_per_dim
        self.max_feval = max_feval
        self.cur_feval = 0
        self.break_recorder = defaultdict(int)
        self.net_val = []
    
    def get_opt_info(self, d):
        depth = d -1
        return(self.lower_bound[depth].item(), self.upper_bound[depth].item(), 
        self.lip_constant[depth].item(), self.max_iter_per_dim[depth].item(), 
        self.bound_error[depth].item(),)


def nested_lip_opt(model,
                   record,
                   cur_depth,
                   x,
                   sampled_data,
                   patch,
                   ground_true_label,
                   device='cuda'):
    # When only one dimension was not assigned a value
    if cur_depth == 0:
        # Assign it
        cur_loc = record.coefs[cur_depth]
        patch[cur_loc[0],cur_loc[1]] = x
        # Make a prediction 
        output = model((sampled_data+patch).unsqueeze(0).to(device))
        record.cur_feval += 1
        z = output.data[0][ground_true_label].item()
        record.net_val.append((z, patch.clone().detach()))
        return z
    else:
        cur_loc = record.coefs[cur_depth-1]
        patch[cur_loc[0],cur_loc[1]] = x
        nb_eval = 0
        lb, ub, lip_k, max_iter, bound_err = record.get_opt_info(cur_depth)
        # print('=== lb,ub',lb,ub)
        cur_depth -= 1
        w1 = nested_lip_opt(model, record, cur_depth, lb,
                            sampled_data, patch, ground_true_label)
        w2 = nested_lip_opt(model, record, cur_depth, ub,
                            sampled_data, patch, ground_true_label)
        nb_eval += 2
        # print('=== w1,w2', w1, w2)
        y_sorted = [(lb,w1),(ub,w2)]
        y_sorted.sort()

        # do
        y_2 = calculate_y_i(y_sorted, lip_k)
        w_y_2 = nested_lip_opt(model, record, cur_depth, y_2,
                                sampled_data, patch, ground_true_label)
        # print('=== y2,wy2', y_2, w_y_2)
        nb_eval += 1
        z_1, z_2 = calculate_z_i(y_sorted, lip_k, (y_2,w_y_2))
        z_all = [z_1, z_2]
        y_sorted.append((y_2,w_y_2))
        y_sorted.sort()
        l_i = min(z_all)
        u_i = min(y_sorted, key=lambda item: item[1])[1]
        # while
        # while (nb_eval <= max_iter and (u_i - l_i) > bound_err and model.eval_iters < overall_eval):
        # while True:
        iii = 0
        while iii < 1000:
            if (u_i - l_i) <= bound_err:
                # print('The distance between lower bound and upper bound is smaller than the bound error!')
                record.break_recorder['less than bound error'] += 1
                break
            elif nb_eval > max_iter:
                # print('Reach maximum iteration on a dimension!')
                record.break_recorder['Reach maximum iteration on a dimension'] += 1
                break
            elif record.cur_feval > record.max_feval:
                # print('Reach the maximum number of evaluation!')
                record.break_recorder['Reach the maximum number of evaluation'] += 1
                break
            # print('=== z_all ',[xx for xx in z_all])
            z_loc = np.argmin(z_all)
            # print('=== y sorted', y_sorted)
            y_i = calculate_y_i([y_sorted[z_loc],y_sorted[z_loc+1]],lip_k)
            # print('=== y_i', y_i)
            w_y_i = nested_lip_opt(model, record, cur_depth, y_i,
                                    sampled_data, patch, ground_true_label)
            nb_eval += 1
            z_left, z_right = calculate_z_i(y_sorted[z_loc:z_loc+2], lip_k, (y_i,w_y_i))
            # print('=== z_left, zright', z_left, z_right)
            z_all = z_all[:z_loc]+[z_left,z_right]+z_all[z_loc+1:]
            y_sorted.append((y_i, w_y_i))
            y_sorted.sort()
            l_i = min(z_all)
            u_i = min(y_sorted, key=lambda item: item[1])[1]
            # print('=== li,ui', l_i, u_i)

            iii += 1

        patch[cur_loc[0],cur_loc[1]] = y_sorted[0][0]
        return u_i


# def run_deepgo(model,
#               sampled_data,
#               overall_eval,
#               verify_type,
#               device='cpu'):

#     # Make prediction on the original input data
#     original_output = model(sampled_data)
#     original_prediction = torch.argmax(original_output[0])
#     original_up_bound = torch.max(original_output)

#     # Set up starting dimension and initial value for that dimension
#     nest_depth = model.setup()
#     x_loc = model.coefs[nest_depth]
#     x = sampled_data[0, x_loc[0], x_loc[1]]
#     # Run nested Lipschitz optimization
#     nested_lip_opt(model, nest_depth, x, sampled_data, 
#                    original_prediction, overall_eval)
#     if verify_type == 'min':
#         _, final_example = min(model.net_val, key=lambda item: item[0])
#     elif verify_type == 'max':
#         _, final_example = max(model.net_val, key=lambda item: item[0])
#     return final_example