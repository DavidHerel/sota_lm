import numpy as np
import torch
import os

class Perplexity_loss(torch.nn.Module):
    def __init__(self, num_models: int):
        """
        We instantiate weights of the ensemble model.
        """
        super().__init__()
        self.weights = torch.nn.Parameter(torch.ones((num_models, 1)))

    def forward(self, probabilities):
        """
        We calculate directly the cross entropy loss used in Perplexity.
        """
        weights_softmax = torch.softmax(self.weights, dim=0)
        lin_comb = weights_softmax * probabilities
        lin_comb_sum = torch.sum(lin_comb, dim=0)
        loss = -1 * torch.mean(torch.log(lin_comb_sum))
        return loss


def optimise_ensemble_weights(probabilities: np.ndarray, num_steps: int = 5000, lr: float=0.05):
    """
    Find optimal linear weights for the ensemble model given the word probabilities for each model.
    """
    probabilities = torch.from_numpy(probabilities)
    model = Perplexity_loss(probabilities.shape[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scaled_weights = None
    for t in range(num_steps):
        # Forward pass
        loss = model(probabilities)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 100 == 0:
            ppl = torch.exp(loss)
            scaled_weights = torch.softmax(model.weights, dim=0).detach().numpy()[:, 0]
            #print('Loss:', loss.item(), 'Perplexity:', ppl.item(), 'Weights:', scaled_weights)

    return scaled_weights


def softmax(x):
    x = np.exp(x)
    f_x = x / np.sum(x)
    return f_x


def combine_prob_text(txt_file):
    arr = []
    with open(txt_file, encoding="utf-8") as f:
        for line in f:
            temp = float(line)
            arr.append(temp)
    arr = np.array(arr)
    return arr


# given sequence of predicted and actual tokens - calculate CE loss
def calculate_sequence_loss(x):
    L = -1 * np.mean(np.log(x))
    ppl = np.exp(L)
    return L, ppl

#
# if __name__ == '__main__':
#     np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
#     # start a new wandb run to track this script
#     # wandb.init(
#     #     # set the wandb project where this run will be logged
#     #     project="ensemble_hyperparams",
#     #     entity="hereldav"
#     # )
#
#     rel_paths = ['ptb-ppl', 'wt2-ppl', 'wt103-ppl']
#     for path in rel_paths:
#         print('\n'+'-'*50)
#         print('\n'+path)
#
#         path = os.path.relpath(path)
#         val_files = sorted(os.listdir(os.path.join(path, 'valid')))
#         test_files = sorted(os.listdir(os.path.join(path, 'test')))
#
#
#         val_files_parsed = [f.replace('valid','') for f in val_files]
#         test_files_parsed = [f.replace('test','') for f in test_files]
#         assert val_files_parsed == test_files_parsed, 'Different names for validation and test files'
#
#         val_probabilities = np.vstack([combine_prob_text(os.path.join(path,'valid', file_name)) for file_name in val_files])
#         test_probabilities = np.vstack([combine_prob_text(os.path.join(path, 'test', file_name)) for file_name in test_files])
#
#         print("\nIndividual valid ppl of models")
#         for name,i in zip(val_files,val_probabilities):
#             #skip unigram cache
#             if "unigram" in name:
#                 continue
#             print(name +": "+str(round(calculate_sequence_loss(i)[1],2)))
#
#         print("\nIndividual test ppl of models")
#         for name, i in zip(test_files, test_probabilities):
#             # skip unigram cache
#             if "unigram" in name:
#                 continue
#             print(name + ": " + str(round(calculate_sequence_loss(i)[1], 2)))
#
#         weights = optimise_ensemble_weights(val_probabilities)
#         # print('\nModel path', path)
#         # print('Final weights', weights)
#
#         val_file_prob = (weights[:, np.newaxis] * val_probabilities).sum(axis=0)
#         test_file_prob = (weights[:, np.newaxis] * test_probabilities).sum(axis=0)
#
#         val_loss, val_ppl = calculate_sequence_loss(val_file_prob)
#         test_loss, test_ppl = calculate_sequence_loss(test_file_prob)
#         print('\nValidation Perplexity: ', val_ppl)
#         print('Test Perplexity: ', test_ppl)
#
#         print("\nName of files with weights")
#         for name, w in zip(test_files_parsed, weights):
#             print(name+': '+str(round(w, 2)))

