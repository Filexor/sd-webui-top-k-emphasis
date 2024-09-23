from math import ceil
import torch
import einops

from modules.devices import device

from scripts.parsing import EmphasisPair

class Emphasis:
    name: str = "Base"
    description: str = ""
    tokens: list[list[int]]
    multipliers: list[list[EmphasisPair]]
    z: torch.Tensor # [batch, token, channel]

    def after_transformers(self):
        pass

class TopKEmphasis(Emphasis):
    name = "Top K Emphasis"
    description = "Expanded emphasis method"

    def after_transformers_old(self):
        for i in range(self.z.shape[0]):
            for j in range(self.z.shape[1]):
                z_dec = self.z[i, j, :].sort(descending=True).values
                for k in self.multipliers[i][j].multipliers:
                    if k.key == "c":
                        if k.threshold < 0.0:
                            raise ValueError("Negative threshold is unacceptable.")
                        elif k.threshold == 0.0:
                            self.z[i, j, :] *= k.weight
                        elif k.threshold < 1.0:
                            threshold = z_dec[int(k.threshold * self.z.shape[2]) - 1]
                            self.z[i, j, :] *= torch.where(self.z[i, j, :] >= threshold, k.weight, 1.0)
                        else:
                            threshold = z_dec[min(int(k.threshold), self.z.shape[2]) - 1]
                            self.z[i, j, :] *= torch.where(self.z[i, j, :] >= threshold, k.weight, 1.0)

    def after_transformers(self):
        for i, pair in enumerate(self.multipliers):
            multiplier, threshold = to_structure_of_tensor(pair, "c")
            threshold = torch.where(threshold == 0.0, self.z.shape[2], threshold)
            threshold = torch.where(threshold < 1.0, threshold * self.z.shape[2], threshold)
            threshold = threshold - 1
            threshold = threshold.to(dtype=torch.int32).to(device)
            multiplier = multiplier.to(device)
            for j in range(threshold.shape[0]):
                z_dec = self.z[i, ...].sort(dim=1, descending=True).values
                selected_z_dec = z_dec.index_select(dim=1, index=threshold[j, :]).diag()
                expanded_z_dec = selected_z_dec.unsqueeze(1).expand(-1, self.z.shape[2])
                expanded_multiplier = multiplier[j, :].unsqueeze(1).expand(-1, self.z.shape[2])
                self.z[i, ...] *= torch.where(self.z[i, ...] >= expanded_z_dec, expanded_multiplier, 1.0)

def to_structure_of_tensor(input: list[EmphasisPair], key: str) -> tuple[torch.Tensor, torch.Tensor]:
    count = 0
    weights = []
    thresholds = []
    for i in input:
        weight = []
        threshold = []
        count_tokenwise = 0
        for j in i.multipliers:
            if j.key == key:
                count_tokenwise += 1
                weight.append(j.weight)
                threshold.append(j.threshold)
        count = max(count, count_tokenwise)
        weights.append(weight)
        thresholds.append(threshold)
    for i in weights:
        i += [1.0] * (count - len(i))
    for i in thresholds:
        i += [0.0] * (count - len(i))
    weight = torch.asarray(weights)
    weight = einops.rearrange(weight, "a b -> b a")
    threshold = torch.asarray(thresholds)
    threshold = einops.rearrange(threshold, "a b -> b a")
    return weight, threshold