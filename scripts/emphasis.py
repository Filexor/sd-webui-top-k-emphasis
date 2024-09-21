from math import ceil
import torch

from scripts.parsing import EmphasisPair


class Emphasis:
    name: str = "Base"
    description: str = ""
    tokens: list[list[int]]
    multipliers: list[list[EmphasisPair]]
    z: torch.Tensor

    def after_transformers(self):
        pass

class TopKEmphasis(Emphasis):
    name = "Top K Emphasis"
    description = "Expanded emphasis method"

    def after_transformers(self):
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
