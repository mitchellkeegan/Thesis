import torch

def sigmoid_grad(diffs,k):
    return k*torch.exp(-k*(diffs))/(torch.exp(-k*(diffs))+1)**2

class MyRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        print(ctx.generate_vmap_rule)
        ctx.save_for_backward(input)
        ctx.k = k
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        k = ctx.k
        grad_input = grad_output.clone()
        return k*torch.exp(-k*(input-0.5))/(torch.exp(-k*(input-0.5))+1)**2 * grad_input, None

# Not for actual use. This is a reference implementation for MyOnehot1d below since the actual function
# is implemented in a creative fashion to avoid indexing with max_idx (which vmap cannot do)
class MyOnehot1d_REFERENCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Assume input is of size (num_classes)
        ctx.save_for_backward(input)
        n_classes = input.shape[0]
        top2 = torch.topk(input, 2, dim=0)
        ctx.max_val, ctx.max_idx = top2[0][0], top2[1][0]
        ctx.penul_val = top2[0][1]
        return torch.nn.functional.one_hot(ctx.max_idx,n_classes).float()

    @staticmethod
    def backward(ctx, grad_output):
        n_classes = grad_output.shape[0]

        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        max_val = ctx.max_val
        max_idx = ctx.max_idx
        penul_val = ctx.penul_val
        grad = torch.zeros_like(grad_output)

        # The j^th element of diffs should be z_j - z_max, exceptt for when j=z_max in which case it is z_pen - z_max
        diffs = input - max_val
        diffs[max_idx] = penul_val - max_val

        # j^th element of sigs should be sig(z_j,z_max), except for when j=z_max in which case it is sig(z_pen,z_max)
        sigs = sigmoid_grad(diffs,20)

        grad = torch.zeros_like(grad_output)

        #
        mod_diff = -1*torch.ones_like(diffs)
        mod_diff[max_idx] = 1

        for idx in range(n_classes):
            if idx == max_idx:
                grad[idx] = torch.dot(mod_diff * sigs,grad_output)
            else:
                grad[idx] = sigs[idx]*(grad_output[idx] - grad_output[max_idx])
        return grad, None

# Old reference implementation based on inverse of difference instead of sigmoid
class MyOnehot1d_CRUDE_REFERENCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Assume input is of size (num_classes)
        ctx.save_for_backward(input)
        n_classes = input.shape[0]
        top2 = torch.topk(input, 2, dim=0)
        ctx.max_val, ctx.max_idx = top2[0][0], top2[1][0]
        ctx.penul_val = top2[0][1]
        return torch.nn.functional.one_hot(ctx.max_idx,n_classes).float()

    @staticmethod
    def backward(ctx, grad_output):
        n_classes = grad_output.shape[0]

        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        max_val = ctx.max_val
        max_idx = ctx.max_idx
        penul_val = ctx.penul_val
        grad = torch.ones_like(grad_output)

        # Max value
        diffs = max_val - input
        diffs[max_idx] = max_val - penul_val
        diffs = torch.clamp(1/diffs,min=-10,max=10)

        grad = torch.zeros_like(grad_output)
        mod_diff = -1*torch.ones_like(diffs)
        mod_diff[max_idx] = 1

        for idx in range(n_classes):
            if idx == max_idx:
                grad[idx] = torch.dot(mod_diff * diffs,grad_output)
            else:
                grad[idx] = diffs[idx]*(grad_output[idx] - grad_output[max_idx])
        return grad, None


# TODO Profile the backward pass. The vmapped version is MUCH faster on the forward pass but performance on the backward pass is unclear
class MyOnehot1d(torch.autograd.Function):
    generate_vmap_rule = True
    @staticmethod
    def forward(input):
        # Assume input is of size (num_classes)
        n_classes = input.shape[0]
        top2 = torch.topk(input, 2, dim=0)
        max_val, max_idx = top2[0][0], top2[1][0]
        penul_val = top2[0][1]
        onehot_out = torch.nn.functional.one_hot(max_idx,n_classes)

        diffs = (input - max_val) + onehot_out * (penul_val - max_val)
        sigs = sigmoid_grad(diffs,20)

        return onehot_out.float(), max_idx, sigs

    @staticmethod
    def setup_context(ctx,input,outputs):
        onehot, max_idx, diffs = outputs
        ctx.mark_non_differentiable(max_idx,diffs)
        ctx.save_for_backward(onehot, max_idx, diffs)

    @staticmethod
    def backward(ctx, grad_output, _0, _1):
        # The vmap function does not allow indexing on tensors, the onehot encoding is used as a proxy for normal indexing
        n_classes = grad_output.shape[0]
        onehot, max_idx, diffs = ctx.saved_tensors

        grad = torch.zeros_like(grad_output)
        mod_diff = -1*torch.ones_like(diffs) + 2*onehot
        # vmap does not allow to conditional statements, strange expression below is implicitly allowing us to check if idx == max_idx
        for idx in range(n_classes):
            grad[idx] = (onehot[idx] * torch.dot(mod_diff*diffs,grad_output) +
                         (torch.ones_like(onehot)-onehot)[idx] * diffs[idx]*(grad_output[idx] - torch.sum(grad_output*onehot)))
        return grad

# Old implementation based on inverse of difference instead of sigmoid
class MyOnehot1d_CRUDE(torch.autograd.Function):
    generate_vmap_rule = True
    @staticmethod
    def forward(input):
        # Assume input is of size (num_classes)
        n_classes = input.shape[0]
        top2 = torch.topk(input, 2, dim=0)
        max_val, max_idx = top2[0][0], top2[1][0]
        penul_val = top2[0][1]
        onehot_out = torch.nn.functional.one_hot(max_idx,n_classes)
        diffs = (max_val - input) + onehot_out * (max_val - penul_val)
        diffs = torch.clamp(1 / diffs, min=-10, max=10)

        return onehot_out.float(), max_idx, diffs

    @staticmethod
    def setup_context(ctx,input,outputs):
        onehot, max_idx, diffs = outputs
        ctx.mark_non_differentiable(max_idx,diffs)
        ctx.save_for_backward(onehot, max_idx, diffs)

    @staticmethod
    def backward(ctx, grad_output, _0, _1):
        n_classes = grad_output.shape[0]

        onehot, max_idx, diffs = ctx.saved_tensors


        grad = torch.zeros_like(grad_output)
        mod_diff = -1*torch.ones_like(diffs) + 2*onehot

        for idx in range(n_classes):
            grad[idx] = (onehot[idx] * torch.dot(mod_diff*diffs,grad_output) +
                         (torch.ones_like(onehot)-onehot)[idx] * diffs[idx]*(grad_output[idx] - torch.sum(grad_output*onehot)))

        return grad

MyOnehot = torch.vmap(torch.vmap(torch.vmap(MyOnehot1d.apply)))