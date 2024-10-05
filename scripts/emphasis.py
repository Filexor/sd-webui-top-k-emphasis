import torch
import einops

from modules.devices import device

from scripts.parsing import EmphasisPair


class TransferObject():
    def __init__(self, device: torch.device) -> None:
        self.targets_positive: list[int] = []
        self.targets_negative: list[int] = []
        self.targets_absolute_positive: list[int] = []
        self.targets_absolute_negative: list[int] = []
        self.multiplier: torch.Tensor = torch.asarray([1.0], device=device)
    
    def has_target(self) -> bool:
        return len(self.targets_positive) >= 1 or len(self.targets_negative) >= 1 or len(self.targets_absolute_positive) >= 1 or len(self.targets_absolute_negative) >= 1

def apply_emphasis(z: torch.Tensor, i:int, begin: int, end: int, target: torch.Tensor, 
                   preoffset: torch.Tensor, weight: torch.Tensor, postoffset: torch.Tensor, 
                   zero: torch.Tensor, one: torch.Tensor,
                   transfer_object: TransferObject):
    transfer_values = torch.zeros_like(target, dtype=torch.float32)
    transfer_absolute_values = torch.zeros_like(target, dtype=torch.float32)
    if transfer_object.has_target():
        transfer_values += torch.where(target, z[i, begin:end, :] * transfer_object.multiplier, zero)
        transfer_absolute_values += torch.where(target, z[i, begin:end, :] * transfer_object.multiplier, zero).abs()
        transfer_sum = transfer_values.sum(dim=-1)
        transfer_absolute_sum = transfer_absolute_values.sum(dim=-1)
    for transfer_target in transfer_object.targets_positive:
        z[i, begin:end, transfer_target] += transfer_sum
    for transfer_target in transfer_object.targets_negative:
        z[i, begin:end, transfer_target] -= transfer_sum
    for transfer_target in transfer_object.targets_absolute_positive:
        z[i, begin:end, transfer_target] += transfer_absolute_sum
    for transfer_target in transfer_object.targets_absolute_negative:
        z[i, begin:end, transfer_target] -= transfer_absolute_sum
    z[i, begin:end, :] += torch.where(target, preoffset, zero)
    z[i, begin:end, :] *= torch.where(target, weight, one)
    z[i, begin:end, :] += torch.where(target, postoffset, zero)

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

    def __init__(self, emphasis_view_update=False, debug=False, embedding_key='clip_l') -> None:
        super().__init__()
        self.emphasis_view_update = emphasis_view_update
        self.debug = debug
        self.embedding_key = embedding_key

    def after_transformers_old(self):
        for i in range(self.z.shape[0]):
            for j in range(self.z.shape[1]):
                z_des = self.z[i, j, :].sort(descending=True).values
                for k in self.multipliers[i][j].multipliers:
                    if k.key == "c":
                        if k.threshold < 0.0:
                            raise ValueError("Negative threshold is unacceptable.")
                        elif k.threshold == 0.0:
                            self.z[i, j, :] *= k.weight
                        elif k.threshold < 1.0:
                            threshold = z_des[int(k.threshold * self.z.shape[2]) - 1]
                            self.z[i, j, :] *= torch.where(self.z[i, j, :] >= threshold, k.weight, 1.0)
                        else:
                            threshold = z_des[min(int(k.threshold), self.z.shape[2]) - 1]
                            self.z[i, j, :] *= torch.where(self.z[i, j, :] >= threshold, k.weight, 1.0)

    def after_transformers_old2(self):
        for i, pair in enumerate(self.multipliers):
            multiplier, threshold = to_structure_of_tensor(pair, "c")
            threshold = torch.where(threshold == 0.0, self.z.shape[2], threshold)
            threshold = torch.where(threshold < 1.0, threshold * self.z.shape[2], threshold)
            threshold = threshold - 1
            threshold = threshold.to(dtype=torch.int32).to(device)
            multiplier = multiplier.to(device)
            for j in range(threshold.shape[0]):
                z_des = self.z[i, ...].sort(dim=1, descending=True).values
                selected_z_des = z_des.index_select(dim=1, index=threshold[j, :]).diag()
                expanded_z_des = selected_z_des.unsqueeze(1).expand(-1, self.z.shape[2])
                expanded_multiplier = multiplier[j, :].unsqueeze(1).expand(-1, self.z.shape[2])
                self.z[i, ...] *= torch.where(self.z[i, ...] >= expanded_z_des, expanded_multiplier, 1.0)

    def after_transformers(self):
        if self.emphasis_view_update:
            for i, pairs in enumerate(self.multipliers):
                for pair in pairs:
                    begin = 0 if pair.begin is None else int(pair.begin * self.z.shape[1]) if pair.begin < 1.0 else max(min(int(pair.begin), self.z.shape[1]), 0)
                    end = self.z.shape[1] if pair.end is None else int(pair.end * self.z.shape[1]) if pair.end < 1.0 else max(min(int(pair.end), self.z.shape[1]), 0)
                    for multiplier in pair.multipliers:
                        if multiplier.key == "c" or (multiplier.key == "l" and self.embedding_key == "clip_l") or (multiplier.key == "g" and self.embedding_key == "clip_g"):
                            mode = None
                            value = None
                            preoffset = 0.0
                            postoffset = 0.0
                            transfer_object = TransferObject(self.z.device)
                            for option in multiplier.options:
                                match option[0]:
                                    case "b" | "o" | "m" | "r":
                                        if mode is None:
                                            mode = option[0]
                                            if option[1] is not None:
                                                value = option[1]
                                            else:
                                                value = multiplier.threshold
                                    case "c":
                                        if mode is None:
                                            mode = option[0]
                                            if option[1] is not None:
                                                value = option[1]
                                            else:
                                                value = multiplier.threshold + 1
                                    case "n":
                                        pass    # This is not cross attention.
                                        # if mode is None and option[1] is not None:
                                        #     mode = option[0]
                                        #     value = option[1]
                                    case "pa":
                                        if option[1] is not None:
                                            preoffset += option[1]
                                    case "ps":
                                        if option[1] is not None:
                                            preoffset -= option[1]
                                    case "a":
                                        if option[1] is not None:
                                            postoffset += option[1]
                                    case "s":
                                        if option[1] is not None:
                                            postoffset -= option[1]
                                    case "ta":
                                        if option[1] is not None:
                                            transfer_value = min(int(option[1] * self.z.shape[2]), self.z.shape[2] - 1) if option[1] < 1.0 else max(min(int(option[1]), self.z.shape[2] - 1), 0)
                                            transfer_object.targets_positive.append(transfer_value)
                                    case "ts":
                                        if option[1] is not None:
                                            transfer_value = min(int(option[1] * self.z.shape[2]), self.z.shape[2] - 1) if option[1] < 1.0 else max(min(int(option[1]), self.z.shape[2] - 1), 0)
                                            transfer_object.targets_negative.append(transfer_value)
                                    case "taa":
                                        if option[1] is not None:
                                            transfer_value = min(int(option[1] * self.z.shape[2]), self.z.shape[2] - 1) if option[1] < 1.0 else max(min(int(option[1]), self.z.shape[2] - 1), 0)
                                            transfer_object.targets_absolute_positive.append(transfer_value)
                                    case "tas":
                                        if option[1] is not None:
                                            transfer_value = min(int(option[1] * self.z.shape[2]), self.z.shape[2] - 1) if option[1] < 1.0 else max(min(int(option[1]), self.z.shape[2] - 1), 0)
                                            transfer_object.targets_absolute_negative.append(transfer_value)
                                    case "tw":
                                        if option[1] is not None:
                                            transfer_object.multiplier = torch.asarray(option[1], device=self.z.device)
                            weight = torch.asarray([multiplier.weight], dtype=torch.float32, device=self.z.device)
                            preoffset = torch.asarray([preoffset], dtype=torch.float32, device=self.z.device)
                            postoffset = torch.asarray([postoffset], dtype=torch.float32, device=self.z.device)
                            zero = torch.asarray([0.0], dtype=torch.float32, device=self.z.device)
                            one = torch.asarray([1.0], dtype=torch.float32, device=self.z.device)
                            match mode:
                                case None:
                                    thres_top = multiplier.threshold
                                    thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    z_des = self.z[i, begin:end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des[:, thres_top]
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                    target = self.z[i, begin:end, :] >= z_des_exp
                                    if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(self.z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                case "b":
                                    if multiplier.threshold == 0.0 and value == 0.0:
                                        if self.debug: print("Emphasis will be skipped.")
                                        pass
                                    elif multiplier.threshold != 0.0 and value == 0.0:
                                        thres_top = multiplier.threshold
                                        #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                        thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                        thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                        thres_top = thres_top - 1
                                        thres_top = int(thres_top)
                                        z_des = self.z[i, begin:end, :].sort(dim=-1, descending=True).values
                                        z_des_sel = z_des[:, thres_top]
                                        z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        target = self.z[i, begin:end, :] >= z_des_exp
                                        if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                        apply_emphasis(self.z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                    elif multiplier.threshold == 0.0 and value != 0.0:
                                        thres_top = value
                                        #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                        thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                        thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                        thres_top = thres_top - 1
                                        thres_top = int(thres_top)
                                        z_des = self.z[i, begin:end, :].sort(dim=-1, descending=False).values
                                        z_des_sel = z_des[:, thres_top]
                                        z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        target = self.z[i, begin:end, :] <= z_des_exp
                                        if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                        apply_emphasis(self.z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                    elif multiplier.threshold != 0.0 and value != 0.0:
                                        thres_top = multiplier.threshold
                                        #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                        thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                        thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                        thres_top = thres_top - 1
                                        thres_top = int(thres_top)
                                        thres_bottom = value
                                        #thres_bottom = self.z.shape[2] if thres_bottom == 0 else thres_bottom
                                        thres_bottom = thres_bottom * self.z.shape[2] if thres_bottom < 1 else thres_bottom
                                        thres_bottom = self.z.shape[2] if thres_bottom > self.z.shape[2] else thres_bottom
                                        thres_bottom = thres_bottom - 1
                                        thres_bottom = int(thres_bottom)
                                        z_des = self.z[i, begin:end, :].sort(dim=-1, descending=True).values
                                        z_des_sel = z_des[:, thres_top]
                                        z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        z_asc = z_des.flip([-1])
                                        z_asc_sel = z_asc[:, thres_bottom]
                                        z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        target = (self.z[i, begin:end, :] >= z_des_exp) | (self.z[i, begin:end, :] <= z_asc_exp)
                                        if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                        apply_emphasis(self.z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                case "o":
                                    if multiplier.threshold == 0.0 and value == 0.0:
                                        target = torch.ones_like(self.z[i, begin:end, :], dtype=torch.bool, device=self.z.device)
                                        if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                        apply_emphasis(self.z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                    elif multiplier.threshold != 0.0 and value == 0.0:
                                        thres_top = multiplier.threshold
                                        #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                        thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                        thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                        thres_top = thres_top - 1
                                        thres_top = int(thres_top)
                                        z_des = self.z[i, begin:end, :].sort(dim=-1, descending=True).values
                                        z_des_sel = z_des[:, thres_top]
                                        z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        target = self.z[i, begin:end, :] < z_des_exp
                                        if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                        apply_emphasis(self.z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                    elif multiplier.threshold == 0.0 and value != 0.0:
                                        thres_top = value
                                        #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                        thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                        thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                        thres_top = thres_top - 1
                                        thres_top = int(thres_top)
                                        z_des = self.z[i, begin:end, :].sort(dim=-1, descending=False).values
                                        z_des_sel = z_des[:, thres_top]
                                        z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        target = self.z[i, begin:end, :] > z_des_exp
                                        if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                        apply_emphasis(self.z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                    elif multiplier.threshold != 0.0 and value != 0.0:
                                        thres_top = multiplier.threshold
                                        #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                        thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                        thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                        thres_top = thres_top - 1
                                        thres_top = int(thres_top)
                                        thres_bottom = value
                                        #thres_bottom = self.z.shape[2] if thres_bottom == 0 else thres_bottom
                                        thres_bottom = thres_bottom * self.z.shape[2] if thres_bottom < 1 else thres_bottom
                                        thres_bottom = self.z.shape[2] if thres_bottom > self.z.shape[2] else thres_bottom
                                        thres_bottom = thres_bottom - 1
                                        thres_bottom = int(thres_bottom)
                                        z_des = self.z[i, begin:end, :].sort(dim=-1, descending=True).values
                                        z_des_sel = z_des[:, thres_top]
                                        z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        z_asc = z_des.flip([-1])
                                        z_asc_sel = z_asc[:, thres_bottom]
                                        z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        target = (self.z[i, begin:end, :] < z_des_exp) & (self.z[i, begin:end, :] > z_asc_exp)
                                        if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                        apply_emphasis(self.z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                case "m":
                                    thres_top = multiplier.threshold
                                    thres_top = (self.z.shape[2] // 2) - thres_top if thres_top >= 1 else thres_top
                                    thres_top = (self.z.shape[2] // 2) - thres_top * (self.z.shape[2] // 2) if thres_top < 1 else thres_top
                                    thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    thres_bottom = value
                                    thres_bottom = (self.z.shape[2] // 2) - thres_bottom if thres_bottom >= 1 else thres_bottom
                                    thres_bottom = (self.z.shape[2] // 2) - thres_bottom * (self.z.shape[2] // 2) if thres_bottom < 1 else thres_bottom
                                    thres_bottom = self.z.shape[2] if thres_bottom > self.z.shape[2] else thres_bottom
                                    thres_bottom = thres_bottom - 1
                                    thres_bottom = int(thres_bottom)
                                    z_des = self.z[i, begin:end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des[:, thres_top]
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                    z_asc = z_des.flip([-1])
                                    z_asc_sel = z_asc[:, thres_bottom]
                                    z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                    target = (self.z[i, begin:end, :] < z_des_exp) & (self.z[i, begin:end, :] > z_asc_exp)
                                    if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(self.z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                case "r":
                                    thres_top = multiplier.threshold
                                    #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = self.z.shape[2] - 1 if thres_top > self.z.shape[2] - 1 else thres_top
                                    #thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    thres_bottom = value
                                    #thres_bottom = self.z.shape[2] if thres_bottom == 0 else thres_bottom
                                    thres_bottom = thres_bottom * self.z.shape[2] if thres_bottom < 1 else thres_bottom
                                    thres_bottom = self.z.shape[2] - 1 if thres_bottom > self.z.shape[2] - 1 else thres_bottom
                                    #thres_bottom = thres_bottom - 1
                                    thres_bottom = int(thres_bottom)
                                    z_des = self.z[i, begin:end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des[:, thres_top]
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                    z_asc = z_des
                                    z_asc_sel = z_asc[:, thres_bottom]
                                    z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                    target = (self.z[i, begin:end, :] <= z_des_exp) & (self.z[i, begin:end, :] >= z_asc_exp)
                                    if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(self.z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                case "c":
                                    thres_top = multiplier.threshold
                                    #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = self.z.shape[2] - 1 if thres_top > self.z.shape[2] else thres_top
                                    #thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    thres_bottom = value
                                    #thres_bottom = self.z.shape[2] if thres_bottom == 0 else thres_bottom
                                    thres_bottom = thres_bottom * self.z.shape[2] if thres_bottom < 1 else thres_bottom
                                    thres_bottom = self.z.shape[2] if thres_bottom > self.z.shape[2] else thres_bottom
                                    # thres_bottom = thres_bottom - 1
                                    thres_bottom = int(thres_bottom)
                                    target = torch.zeros_like(self.z[i, begin:end, :], dtype=torch.bool, device=self.z.device)
                                    target[:, thres_top:thres_bottom] = True
                                    if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(self.z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
        else:
            for i, pairs in enumerate(self.multipliers):
                for pair in pairs:
                    begin = 0 if pair.begin is None else int(pair.begin * self.z.shape[1]) if pair.begin < 1.0 else max(min(int(pair.begin), self.z.shape[1]), 0)
                    end = self.z.shape[1] if pair.end is None else int(pair.end * self.z.shape[1]) if pair.end < 1.0 else max(min(int(pair.end), self.z.shape[1]), 0)
                    z_des = self.z[i, begin:end, :].sort(dim=-1, descending=True).values
                    z_asc = z_des.flip([-1])
                    for multiplier in pair.multipliers:
                        if multiplier.key == "c" or (multiplier.key == "l" and self.embedding_key == "clip_l") or (multiplier.key == "g" and self.embedding_key == "clip_g"):
                            mode = None
                            value = None
                            preoffset = 0.0
                            postoffset = 0.0
                            transfer_object = TransferObject(self.z.device)
                            for option in multiplier.options:
                                match option[0]:
                                    case "b" | "o" | "m" | "r":
                                        if mode is None:
                                            mode = option[0]
                                            if option[1] is not None:
                                                value = option[1]
                                            else:
                                                value = multiplier.threshold
                                    case "c":
                                        if mode is None:
                                            mode = option[0]
                                            if option[1] is not None:
                                                value = option[1]
                                            else:
                                                value = multiplier.threshold + 1
                                    case "n":
                                        pass    # This is not cross attention.
                                        # if mode is None and option[1] is not None:
                                        #     mode = option[0]
                                        #     value = option[1]
                                    case "pa":
                                        if option[1] is not None:
                                            preoffset += option[1]
                                    case "ps":
                                        if option[1] is not None:
                                            preoffset -= option[1]
                                    case "a":
                                        if option[1] is not None:
                                            postoffset += option[1]
                                    case "s":
                                        if option[1] is not None:
                                            postoffset -= option[1]
                                    case "ta":
                                        if option[1] is not None:
                                            transfer_value = min(int(option[1] * self.z.shape[2]), self.z.shape[2] - 1) if option[1] < 1.0 else max(min(int(option[1]), self.z.shape[2] - 1), 0)
                                            transfer_object.targets_positive.append(transfer_value)
                                    case "ts":
                                        if option[1] is not None:
                                            transfer_value = min(int(option[1] * self.z.shape[2]), self.z.shape[2] - 1) if option[1] < 1.0 else max(min(int(option[1]), self.z.shape[2] - 1), 0)
                                            transfer_object.targets_negative.append(transfer_value)
                                    case "taa":
                                        if option[1] is not None:
                                            transfer_value = min(int(option[1] * self.z.shape[2]), self.z.shape[2] - 1) if option[1] < 1.0 else max(min(int(option[1]), self.z.shape[2] - 1), 0)
                                            transfer_object.targets_absolute_positive.append(transfer_value)
                                    case "tas":
                                        if option[1] is not None:
                                            transfer_value = min(int(option[1] * self.z.shape[2]), self.z.shape[2] - 1) if option[1] < 1.0 else max(min(int(option[1]), self.z.shape[2] - 1), 0)
                                            transfer_object.targets_absolute_negative.append(transfer_value)
                                    case "tw":
                                        if option[1] is not None:
                                            transfer_object.multiplier = torch.asarray(option[1], device=self.z.device)
                            weight = torch.asarray([multiplier.weight], dtype=torch.float32, device=self.z.device)
                            preoffset = torch.asarray([preoffset], dtype=torch.float32, device=self.z.device)
                            postoffset = torch.asarray([postoffset], dtype=torch.float32, device=self.z.device)
                            zero = torch.asarray([0.0], dtype=torch.float32, device=self.z.device)
                            one = torch.asarray([1.0], dtype=torch.float32, device=self.z.device)
                            match mode:
                                case None:
                                    thres_top = multiplier.threshold
                                    thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    #z_des = self.z[i, begin:end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des[:, thres_top]
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                    target = self.z[i, begin:end, :] >= z_des_exp
                                    if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(self.z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                case "b":
                                    if multiplier.threshold == 0.0 and value == 0.0:
                                        if self.debug: print("Emphasis will be skipped.")
                                        pass
                                    elif multiplier.threshold != 0.0 and value == 0.0:
                                        thres_top = multiplier.threshold
                                        #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                        thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                        thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                        thres_top = thres_top - 1
                                        thres_top = int(thres_top)
                                        #z_des = self.z[i, begin:end, :].sort(dim=-1, descending=True).values
                                        z_des_sel = z_des[:, thres_top]
                                        z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        target = self.z[i, begin:end, :] >= z_des_exp
                                        if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                        apply_emphasis(self.z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                    elif multiplier.threshold == 0.0 and value != 0.0:
                                        thres_top = value
                                        #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                        thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                        thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                        thres_top = thres_top - 1
                                        thres_top = int(thres_top)
                                        #z_des = self.z[i, begin:end, :].sort(dim=-1, descending=False).values
                                        z_des_sel = z_des[:, thres_top]
                                        z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        target = self.z[i, begin:end, :] <= z_des_exp
                                        if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                        apply_emphasis(self.z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                    elif multiplier.threshold != 0.0 and value != 0.0:
                                        thres_top = multiplier.threshold
                                        #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                        thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                        thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                        thres_top = thres_top - 1
                                        thres_top = int(thres_top)
                                        thres_bottom = value
                                        #thres_bottom = self.z.shape[2] if thres_bottom == 0 else thres_bottom
                                        thres_bottom = thres_bottom * self.z.shape[2] if thres_bottom < 1 else thres_bottom
                                        thres_bottom = self.z.shape[2] if thres_bottom > self.z.shape[2] else thres_bottom
                                        thres_bottom = thres_bottom - 1
                                        thres_bottom = int(thres_bottom)
                                        #z_des = self.z[i, begin:end, :].sort(dim=-1, descending=True).values
                                        z_des_sel = z_des[:, thres_top]
                                        z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        #z_asc = z_des.flip([-1])
                                        z_asc_sel = z_asc[:, thres_bottom]
                                        z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        target = (self.z[i, begin:end, :] >= z_des_exp) | (self.z[i, begin:end, :] <= z_asc_exp)
                                        if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                        apply_emphasis(self.z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                case "o":
                                    if multiplier.threshold == 0.0 and value == 0.0:
                                        target = torch.ones_like(self.z[i, begin:end, :], dtype=torch.bool, device=self.z.device)
                                        if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                        apply_emphasis(self.z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                    elif multiplier.threshold != 0.0 and value == 0.0:
                                        thres_top = multiplier.threshold
                                        #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                        thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                        thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                        thres_top = thres_top - 1
                                        thres_top = int(thres_top)
                                        #z_des = self.z[i, begin:end, :].sort(dim=-1, descending=True).values
                                        z_des_sel = z_des[:, thres_top]
                                        z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        target = self.z[i, begin:end, :] < z_des_exp
                                        if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                        apply_emphasis(self.z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                    elif multiplier.threshold == 0.0 and value != 0.0:
                                        thres_top = value
                                        #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                        thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                        thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                        thres_top = thres_top - 1
                                        thres_top = int(thres_top)
                                        #z_des = self.z[i, begin:end, :].sort(dim=-1, descending=False).values
                                        z_des_sel = z_des[:, thres_top]
                                        z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        target = self.z[i, begin:end, :] > z_des_exp
                                        if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                        apply_emphasis(self.z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                    elif multiplier.threshold != 0.0 and value != 0.0:
                                        thres_top = multiplier.threshold
                                        #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                        thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                        thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                        thres_top = thres_top - 1
                                        thres_top = int(thres_top)
                                        thres_bottom = value
                                        #thres_bottom = self.z.shape[2] if thres_bottom == 0 else thres_bottom
                                        thres_bottom = thres_bottom * self.z.shape[2] if thres_bottom < 1 else thres_bottom
                                        thres_bottom = self.z.shape[2] if thres_bottom > self.z.shape[2] else thres_bottom
                                        thres_bottom = thres_bottom - 1
                                        thres_bottom = int(thres_bottom)
                                        #z_des = self.z[i, begin:end, :].sort(dim=-1, descending=True).values
                                        z_des_sel = z_des[:, thres_top]
                                        z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        #z_asc = z_des.flip([-1])
                                        z_asc_sel = z_asc[:, thres_bottom]
                                        z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        target = (self.z[i, begin:end, :] < z_des_exp) & (self.z[i, begin:end, :] > z_asc_exp)
                                        if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                        apply_emphasis(self.z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                case "m":
                                    thres_top = multiplier.threshold
                                    thres_top = (self.z.shape[2] // 2) - thres_top if thres_top >= 1 else thres_top
                                    thres_top = (self.z.shape[2] // 2) - thres_top * (self.z.shape[2] // 2) if thres_top < 1 else thres_top
                                    thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    thres_bottom = value
                                    thres_bottom = (self.z.shape[2] // 2) - thres_bottom if thres_bottom >= 1 else thres_bottom
                                    thres_bottom = (self.z.shape[2] // 2) - thres_bottom * (self.z.shape[2] // 2) if thres_bottom < 1 else thres_bottom
                                    thres_bottom = self.z.shape[2] if thres_bottom > self.z.shape[2] else thres_bottom
                                    thres_bottom = thres_bottom - 1
                                    thres_bottom = int(thres_bottom)
                                    #z_des = self.z[i, begin:end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des[:, thres_top]
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                    #z_asc = z_des.flip([-1])
                                    z_asc_sel = z_asc[:, thres_bottom]
                                    z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                    target = (self.z[i, begin:end, :] < z_des_exp) & (self.z[i, begin:end, :] > z_asc_exp)
                                    if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(self.z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                case "r":
                                    thres_top = multiplier.threshold
                                    #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = self.z.shape[2] - 1 if thres_top > self.z.shape[2] - 1 else thres_top
                                    #thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    thres_bottom = value
                                    #thres_bottom = self.z.shape[2] if thres_bottom == 0 else thres_bottom
                                    thres_bottom = thres_bottom * self.z.shape[2] if thres_bottom < 1 else thres_bottom
                                    thres_bottom = self.z.shape[2] - 1 if thres_bottom > self.z.shape[2] - 1 else thres_bottom
                                    #thres_bottom = thres_bottom - 1
                                    thres_bottom = int(thres_bottom)
                                    #z_des = self.z[i, begin:end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des[:, thres_top]
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                    #z_asc = z_des
                                    z_asc_sel = z_des.index_select(dim=-1, index=thres_bottom).diag()
                                    z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                    target = (self.z[i, begin:end, :] <= z_des_exp) & (self.z[i, begin:end, :] >= z_asc_exp)
                                    if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(self.z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                case "c":
                                    thres_top = multiplier.threshold
                                    #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = self.z.shape[2] - 1 if thres_top > self.z.shape[2] else thres_top
                                    #thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    thres_bottom = value
                                    #thres_bottom = self.z.shape[2] if thres_bottom == 0 else thres_bottom
                                    thres_bottom = thres_bottom * self.z.shape[2] if thres_bottom < 1 else thres_bottom
                                    thres_bottom = self.z.shape[2] if thres_bottom > self.z.shape[2] else thres_bottom
                                    # thres_bottom = thres_bottom - 1
                                    thres_bottom = int(thres_bottom)
                                    target = torch.zeros_like(self.z[i, begin:end, :], dtype=torch.bool, device=self.z.device)
                                    target[:, thres_top:thres_bottom] = True
                                    if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(self.z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)

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

def emphasis_b(z, multipliers, emphasis_view_update, embedding_key, debug):
    if emphasis_view_update:
        for i, pairs in enumerate(multipliers):
            for pair in pairs:
                begin = 0 if pair.begin is None else int(pair.begin * z.shape[1]) if pair.begin < 1.0 else max(min(int(pair.begin), z.shape[1]), 0)
                end = z.shape[1] if pair.end is None else int(pair.end * z.shape[1]) if pair.end < 1.0 else max(min(int(pair.end), z.shape[1]), 0)
                for multiplier in pair.multipliers:
                    if multiplier.key == "b" or (multiplier.key == "bl" and embedding_key == "clip_l") or (multiplier.key == "bg" and embedding_key == "clip_g"):
                        mode = None
                        value = None
                        preoffset = 0.0
                        postoffset = 0.0
                        transfer_object = TransferObject(z.device)
                        for option in multiplier.options:
                            match option[0]:
                                case "b" | "o" | "m" | "r":
                                    if mode is None:
                                        mode = option[0]
                                        if option[1] is not None:
                                            value = option[1]
                                        else:
                                            value = multiplier.threshold
                                case "c":
                                    if mode is None:
                                        mode = option[0]
                                        if option[1] is not None:
                                            value = option[1]
                                        else:
                                            value = multiplier.threshold + 1
                                case "n":
                                    pass    # This is not cross attention.
                                    # if mode is None and option[1] is not None:
                                    #     mode = option[0]
                                    #     value = option[1]
                                case "pa":
                                    if option[1] is not None:
                                        preoffset += option[1]
                                case "ps":
                                    if option[1] is not None:
                                        preoffset -= option[1]
                                case "a":
                                    if option[1] is not None:
                                        postoffset += option[1]
                                case "s":
                                    if option[1] is not None:
                                        postoffset -= option[1]
                                case "ta":
                                    if option[1] is not None:
                                        transfer_value = min(int(option[1] * z.shape[2]), z.shape[2] - 1) if option[1] < 1.0 else max(min(int(option[1]), z.shape[2] - 1), 0)
                                        transfer_object.targets_positive.append(transfer_value)
                                case "ts":
                                    if option[1] is not None:
                                        transfer_value = min(int(option[1] * z.shape[2]), z.shape[2] - 1) if option[1] < 1.0 else max(min(int(option[1]), z.shape[2] - 1), 0)
                                        transfer_object.targets_negative.append(transfer_value)
                                case "taa":
                                    if option[1] is not None:
                                        transfer_value = min(int(option[1] * z.shape[2]), z.shape[2] - 1) if option[1] < 1.0 else max(min(int(option[1]), z.shape[2] - 1), 0)
                                        transfer_object.targets_absolute_positive.append(transfer_value)
                                case "tas":
                                    if option[1] is not None:
                                        transfer_value = min(int(option[1] * z.shape[2]), z.shape[2] - 1) if option[1] < 1.0 else max(min(int(option[1]), z.shape[2] - 1), 0)
                                        transfer_object.targets_absolute_negative.append(transfer_value)
                                case "tw":
                                    if option[1] is not None:
                                        transfer_object.multiplier = torch.asarray(option[1], device=z.device)
                        weight = torch.asarray([multiplier.weight], dtype=torch.float32, device=z.device)
                        preoffset = torch.asarray([preoffset], dtype=torch.float32, device=z.device)
                        postoffset = torch.asarray([postoffset], dtype=torch.float32, device=z.device)
                        zero = torch.asarray([0.0], dtype=torch.float32, device=z.device)
                        one = torch.asarray([1.0], dtype=torch.float32, device=z.device)
                        match mode:
                            case None:
                                thres_top = multiplier.threshold
                                thres_top = z.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                thres_top = thres_top - 1
                                thres_top = int(thres_top)
                                z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des[:, thres_top]
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                target = z[i, begin:end, :] >= z_des_exp
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                            case "b":
                                if multiplier.threshold == 0.0 and value == 0.0:
                                    if debug: print("Emphasis will be skipped.")
                                    pass
                                elif multiplier.threshold != 0.0 and value == 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des[:, thres_top]
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, begin:end, :] >= z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                elif multiplier.threshold == 0.0 and value != 0.0:
                                    thres_top = value
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    z_des = z[i, begin:end, :].sort(dim=-1, descending=False).values
                                    z_des_sel = z_des[:, thres_top]
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, begin:end, :] <= z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                elif multiplier.threshold != 0.0 and value != 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    thres_bottom = value
                                    #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                    thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                    thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                    thres_bottom = thres_bottom - 1
                                    thres_bottom = int(thres_bottom)
                                    z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des[:, thres_top]
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    z_asc = z_des.flip([-1])
                                    z_asc_sel = z_asc[:, thres_bottom]
                                    z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = (z[i, begin:end, :] >= z_des_exp) | (z[i, begin:end, :] <= z_asc_exp)
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                            case "o":
                                if multiplier.threshold == 0.0 and value == 0.0:
                                    target = torch.ones_like(z[i, begin:end, :], dtype=torch.bool, device=z.device)
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                elif multiplier.threshold != 0.0 and value == 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des[:, thres_top]
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, begin:end, :] < z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                elif multiplier.threshold == 0.0 and value != 0.0:
                                    thres_top = value
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    z_des = z[i, begin:end, :].sort(dim=-1, descending=False).values
                                    z_des_sel = z_des[:, thres_top]
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, begin:end, :] > z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                elif multiplier.threshold != 0.0 and value != 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    thres_bottom = value
                                    #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                    thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                    thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                    thres_bottom = thres_bottom - 1
                                    thres_bottom = int(thres_bottom)
                                    z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des[:, thres_top]
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    z_asc = z_des.flip([-1])
                                    z_asc_sel = z_asc[:, thres_bottom]
                                    z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = (z[i, begin:end, :] < z_des_exp) & (z[i, begin:end, :] > z_asc_exp)
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                            case "m":
                                thres_top = multiplier.threshold
                                thres_top = (z.shape[2] // 2) - thres_top if thres_top >= 1 else thres_top
                                thres_top = (z.shape[2] // 2) - thres_top * (z.shape[2] // 2) if thres_top < 1 else thres_top
                                thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                thres_top = thres_top - 1
                                thres_top = int(thres_top)
                                thres_bottom = value
                                thres_bottom = (z.shape[2] // 2) - thres_bottom if thres_bottom >= 1 else thres_bottom
                                thres_bottom = (z.shape[2] // 2) - thres_bottom * (z.shape[2] // 2) if thres_bottom < 1 else thres_bottom
                                thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                thres_bottom = thres_bottom - 1
                                thres_bottom = int(thres_bottom)
                                z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des[:, thres_top]
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                z_asc = z_des.flip([-1])
                                z_asc_sel = z_asc[:, thres_bottom]
                                z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                target = (z[i, begin:end, :] < z_des_exp) & (z[i, begin:end, :] > z_asc_exp)
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                            case "r":
                                thres_top = multiplier.threshold
                                #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                thres_top = z.shape[2] - 1 if thres_top > z.shape[2] - 1 else thres_top
                                #thres_top = thres_top - 1
                                thres_top = int(thres_top)
                                thres_bottom = value
                                #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                thres_bottom = z.shape[2] - 1 if thres_bottom > z.shape[2] - 1 else thres_bottom
                                #thres_bottom = thres_bottom - 1
                                thres_bottom = int(thres_bottom)
                                z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des[:, thres_top]
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                z_asc = z_des
                                z_asc_sel = z_asc[:, thres_bottom]
                                z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                target = (z[i, begin:end, :] <= z_des_exp) & (z[i, begin:end, :] >= z_asc_exp)
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                            case "c":
                                thres_top = multiplier.threshold
                                #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                thres_top = z.shape[2] - 1 if thres_top > z.shape[2] else thres_top
                                #thres_top = thres_top - 1
                                thres_top = int(thres_top)
                                thres_bottom = value
                                #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                # thres_bottom = thres_bottom - 1
                                thres_bottom = int(thres_bottom)
                                target = torch.zeros_like(z[i, begin:end, :], dtype=torch.bool, device=z.device)
                                target[:, thres_top:thres_bottom] = True
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
    else:
        for i, pairs in enumerate(multipliers):
            for pair in pairs:
                begin = 0 if pair.begin is None else int(pair.begin * z.shape[1]) if pair.begin < 1.0 else max(min(int(pair.begin), z.shape[1]), 0)
                end = z.shape[1] if pair.end is None else int(pair.end * z.shape[1]) if pair.end < 1.0 else max(min(int(pair.end), z.shape[1]), 0)
                z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                z_asc = z_des.flip([-1])
                for multiplier in pair.multipliers:
                    if multiplier.key == "b" or (multiplier.key == "bl" and embedding_key == "clip_l") or (multiplier.key == "bg" and embedding_key == "clip_g"):
                        mode = None
                        value = None
                        preoffset = 0.0
                        postoffset = 0.0
                        transfer_object = TransferObject(z.device)
                        for option in multiplier.options:
                            match option[0]:
                                case "b" | "o" | "m" | "r":
                                    if mode is None:
                                        mode = option[0]
                                        if option[1] is not None:
                                            value = option[1]
                                        else:
                                            value = multiplier.threshold
                                case "c":
                                    if mode is None:
                                        mode = option[0]
                                        if option[1] is not None:
                                            value = option[1]
                                        else:
                                            value = multiplier.threshold + 1
                                case "n":
                                    pass    # This is not cross attention.
                                    # if mode is None and option[1] is not None:
                                    #     mode = option[0]
                                    #     value = option[1]
                                case "pa":
                                    if option[1] is not None:
                                        preoffset += option[1]
                                case "ps":
                                    if option[1] is not None:
                                        preoffset -= option[1]
                                case "a":
                                    if option[1] is not None:
                                        postoffset += option[1]
                                case "s":
                                    if option[1] is not None:
                                        postoffset -= option[1]
                                case "ta":
                                    if option[1] is not None:
                                        transfer_value = min(int(option[1] * z.shape[2]), z.shape[2] - 1) if option[1] < 1.0 else max(min(int(option[1]), z.shape[2] - 1), 0)
                                        transfer_object.targets_positive.append(transfer_value)
                                case "ts":
                                    if option[1] is not None:
                                        transfer_value = min(int(option[1] * z.shape[2]), z.shape[2] - 1) if option[1] < 1.0 else max(min(int(option[1]), z.shape[2] - 1), 0)
                                        transfer_object.targets_negative.append(transfer_value)
                                case "taa":
                                    if option[1] is not None:
                                        transfer_value = min(int(option[1] * z.shape[2]), z.shape[2] - 1) if option[1] < 1.0 else max(min(int(option[1]), z.shape[2] - 1), 0)
                                        transfer_object.targets_absolute_positive.append(transfer_value)
                                case "tas":
                                    if option[1] is not None:
                                        transfer_value = min(int(option[1] * z.shape[2]), z.shape[2] - 1) if option[1] < 1.0 else max(min(int(option[1]), z.shape[2] - 1), 0)
                                        transfer_object.targets_absolute_negative.append(transfer_value)
                                case "tw":
                                    if option[1] is not None:
                                        transfer_object.multiplier = torch.asarray(option[1], device=z.device)
                        weight = torch.asarray([multiplier.weight], dtype=torch.float32, device=z.device)
                        preoffset = torch.asarray([preoffset], dtype=torch.float32, device=z.device)
                        postoffset = torch.asarray([postoffset], dtype=torch.float32, device=z.device)
                        zero = torch.asarray([0.0], dtype=torch.float32, device=z.device)
                        one = torch.asarray([1.0], dtype=torch.float32, device=z.device)
                        match mode:
                            case None:
                                thres_top = multiplier.threshold
                                thres_top = z.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                thres_top = thres_top - 1
                                thres_top = int(thres_top)
                                #z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des[:, thres_top]
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                target = z[i, begin:end, :] >= z_des_exp
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                            case "b":
                                if multiplier.threshold == 0.0 and value == 0.0:
                                    if debug: print("Emphasis will be skipped.")
                                    pass
                                elif multiplier.threshold != 0.0 and value == 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    #z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des[:, thres_top]
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, begin:end, :] >= z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                elif multiplier.threshold == 0.0 and value != 0.0:
                                    thres_top = value
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    #z_des = z[i, begin:end, :].sort(dim=-1, descending=False).values
                                    z_des_sel = z_des[:, thres_top]
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, begin:end, :] <= z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                elif multiplier.threshold != 0.0 and value != 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    thres_bottom = value
                                    #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                    thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                    thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                    thres_bottom = thres_bottom - 1
                                    thres_bottom = int(thres_bottom)
                                    #z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des[:, thres_top]
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    #z_asc = z_des.flip([-1])
                                    z_asc_sel = z_asc[:, thres_bottom]
                                    z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = (z[i, begin:end, :] >= z_des_exp) | (z[i, begin:end, :] <= z_asc_exp)
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                            case "o":
                                if multiplier.threshold == 0.0 and value == 0.0:
                                    target = torch.ones_like(z[i, begin:end, :], dtype=torch.bool, device=z.device)
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                elif multiplier.threshold != 0.0 and value == 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    #z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des[:, thres_top]
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, begin:end, :] < z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                elif multiplier.threshold == 0.0 and value != 0.0:
                                    thres_top = value
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    #z_des = z[i, begin:end, :].sort(dim=-1, descending=False).values
                                    z_des_sel = z_des[:, thres_top]
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, begin:end, :] > z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                elif multiplier.threshold != 0.0 and value != 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    thres_bottom = value
                                    #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                    thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                    thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                    thres_bottom = thres_bottom - 1
                                    thres_bottom = int(thres_bottom)
                                    #z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des[:, thres_top]
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    #z_asc = z_des.flip([-1])
                                    z_asc_sel = z_asc[:, thres_bottom]
                                    z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = (z[i, begin:end, :] < z_des_exp) & (z[i, begin:end, :] > z_asc_exp)
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                            case "m":
                                thres_top = multiplier.threshold
                                thres_top = (z.shape[2] // 2) - thres_top if thres_top >= 1 else thres_top
                                thres_top = (z.shape[2] // 2) - thres_top * (z.shape[2] // 2) if thres_top < 1 else thres_top
                                thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                thres_top = thres_top - 1
                                thres_top = int(thres_top)
                                thres_bottom = value
                                thres_bottom = (z.shape[2] // 2) - thres_bottom if thres_bottom >= 1 else thres_bottom
                                thres_bottom = (z.shape[2] // 2) - thres_bottom * (z.shape[2] // 2) if thres_bottom < 1 else thres_bottom
                                thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                thres_bottom = thres_bottom - 1
                                thres_bottom = int(thres_bottom)
                                #z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des[:, thres_top]
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                #z_asc = z_des.flip([-1])
                                z_asc_sel = z_asc[:, thres_bottom]
                                z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                target = (z[i, begin:end, :] < z_des_exp) & (z[i, begin:end, :] > z_asc_exp)
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                            case "r":
                                thres_top = multiplier.threshold
                                #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                thres_top = z.shape[2] - 1 if thres_top > z.shape[2] - 1 else thres_top
                                #thres_top = thres_top - 1
                                thres_top = int(thres_top)
                                thres_bottom = value
                                #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                thres_bottom = z.shape[2] - 1 if thres_bottom > z.shape[2] - 1 else thres_bottom
                                #thres_bottom = thres_bottom - 1
                                thres_bottom = int(thres_bottom)
                                #z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des[:, thres_top]
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                #z_asc = z_des
                                z_asc_sel = z_des.index_select(dim=-1, index=thres_bottom).diag()
                                z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                target = (z[i, begin:end, :] <= z_des_exp) & (z[i, begin:end, :] >= z_asc_exp)
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                            case "c":
                                thres_top = multiplier.threshold
                                #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                thres_top = z.shape[2] - 1 if thres_top > z.shape[2] else thres_top
                                #thres_top = thres_top - 1
                                thres_top = int(thres_top)
                                thres_bottom = value
                                #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                # thres_bottom = thres_bottom - 1
                                thres_bottom = int(thres_bottom)
                                target = torch.zeros_like(z[i, begin:end, :], dtype=torch.bool, device=z.device)
                                target[:, thres_top:thres_bottom] = True
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
    return z

def emphasis_crossattention(z: torch.Tensor, multipliers_pos: list[list[EmphasisPair]], multipliers_neg: list[list[EmphasisPair]], key: str, crossattentioncounter, emphasis_view_update, debug):
    multipliers = multipliers_neg + multipliers_pos
    if emphasis_view_update:
        for i, pairs in enumerate(multipliers):
            for pair in pairs:
                begin = 0 if pair.begin is None else int(pair.begin * z.shape[1]) if pair.begin < 1.0 else max(min(int(pair.begin), z.shape[1]), 0)
                end = z.shape[1] if pair.end is None else int(pair.end * z.shape[1]) if pair.end < 1.0 else max(min(int(pair.end), z.shape[1]), 0)
                for multiplier in pair.multipliers:
                    if multiplier.key == key:
                        mode = None
                        value = None
                        preoffset = 0.0
                        postoffset = 0.0
                        transfer_object = TransferObject(z.device)
                        crossattentioncountertargets = []
                        for option in multiplier.options:
                            match option[0]:
                                case "b" | "o" | "m" | "r":
                                    if mode is None:
                                        mode = option[0]
                                        if option[1] is not None:
                                            value = option[1]
                                        else:
                                            value = multiplier.threshold
                                case "c":
                                    if mode is None:
                                        mode = option[0]
                                        if option[1] is not None:
                                            value = option[1]
                                        else:
                                            value = multiplier.threshold + 1
                                case "n":
                                    if option[1] is not None:
                                        crossattentioncountertargets.append(int(option[1]))
                                case "pa":
                                    if option[1] is not None:
                                        preoffset += option[1]
                                case "ps":
                                    if option[1] is not None:
                                        preoffset -= option[1]
                                case "a":
                                    if option[1] is not None:
                                        postoffset += option[1]
                                case "s":
                                    if option[1] is not None:
                                        postoffset -= option[1]
                                case "ta":
                                    if option[1] is not None:
                                        transfer_value = min(int(option[1] * z.shape[2]), z.shape[2] - 1) if option[1] < 1.0 else max(min(int(option[1]), z.shape[2] - 1), 0)
                                        transfer_object.targets_positive.append(transfer_value)
                                case "ts":
                                    if option[1] is not None:
                                        transfer_value = min(int(option[1] * z.shape[2]), z.shape[2] - 1) if option[1] < 1.0 else max(min(int(option[1]), z.shape[2] - 1), 0)
                                        transfer_object.targets_negative.append(transfer_value)
                                case "taa":
                                    if option[1] is not None:
                                        transfer_value = min(int(option[1] * z.shape[2]), z.shape[2] - 1) if option[1] < 1.0 else max(min(int(option[1]), z.shape[2] - 1), 0)
                                        transfer_object.targets_absolute_positive.append(transfer_value)
                                case "tas":
                                    if option[1] is not None:
                                        transfer_value = min(int(option[1] * z.shape[2]), z.shape[2] - 1) if option[1] < 1.0 else max(min(int(option[1]), z.shape[2] - 1), 0)
                                        transfer_object.targets_absolute_negative.append(transfer_value)
                                case "tw":
                                    if option[1] is not None:
                                        transfer_object.multiplier = torch.asarray(option[1], device=z.device)
                        if len(crossattentioncountertargets) != 0 and crossattentioncounter not in crossattentioncountertargets:
                            continue
                        weight = torch.asarray([multiplier.weight], dtype=torch.float32, device=z.device)
                        preoffset = torch.asarray([preoffset], dtype=torch.float32, device=z.device)
                        postoffset = torch.asarray([postoffset], dtype=torch.float32, device=z.device)
                        zero = torch.asarray([0.0], dtype=torch.float32, device=z.device)
                        one = torch.asarray([1.0], dtype=torch.float32, device=z.device)
                        match mode:
                            case None:
                                thres_top = multiplier.threshold
                                thres_top = z.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                thres_top = thres_top - 1
                                thres_top = int(thres_top)
                                z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des[:, thres_top]
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                target = z[i, begin:end, :] >= z_des_exp
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                            case "b":
                                if multiplier.threshold == 0.0 and value == 0.0:
                                    if debug: print("Emphasis will be skipped.")
                                    pass
                                elif multiplier.threshold != 0.0 and value == 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des[:, thres_top]
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, begin:end, :] >= z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                elif multiplier.threshold == 0.0 and value != 0.0:
                                    thres_top = value
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    z_des = z[i, begin:end, :].sort(dim=-1, descending=False).values
                                    z_des_sel = z_des[:, thres_top]
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, begin:end, :] <= z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                elif multiplier.threshold != 0.0 and value != 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    thres_bottom = value
                                    #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                    thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                    thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                    thres_bottom = thres_bottom - 1
                                    thres_bottom = int(thres_bottom)
                                    z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des[:, thres_top]
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    z_asc = z_des.flip([-1])
                                    z_asc_sel = z_asc[:, thres_bottom]
                                    z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = (z[i, begin:end, :] >= z_des_exp) | (z[i, begin:end, :] <= z_asc_exp)
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                            case "o":
                                if multiplier.threshold == 0.0 and value == 0.0:
                                    target = torch.ones_like(z[i, begin:end, :], dtype=torch.bool, device=z.device)
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                elif multiplier.threshold != 0.0 and value == 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des[:, thres_top]
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, begin:end, :] < z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                elif multiplier.threshold == 0.0 and value != 0.0:
                                    thres_top = value
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    z_des = z[i, begin:end, :].sort(dim=-1, descending=False).values
                                    z_des_sel = z_des[:, thres_top]
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, begin:end, :] > z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                elif multiplier.threshold != 0.0 and value != 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    thres_bottom = value
                                    #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                    thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                    thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                    thres_bottom = thres_bottom - 1
                                    thres_bottom = int(thres_bottom)
                                    z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des[:, thres_top]
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    z_asc = z_des.flip([-1])
                                    z_asc_sel = z_asc[:, thres_bottom]
                                    z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = (z[i, begin:end, :] < z_des_exp) & (z[i, begin:end, :] > z_asc_exp)
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                            case "m":
                                thres_top = multiplier.threshold
                                thres_top = (z.shape[2] // 2) - thres_top if thres_top >= 1 else thres_top
                                thres_top = (z.shape[2] // 2) - thres_top * (z.shape[2] // 2) if thres_top < 1 else thres_top
                                thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                thres_top = thres_top - 1
                                thres_top = int(thres_top)
                                thres_bottom = value
                                thres_bottom = (z.shape[2] // 2) - thres_bottom if thres_bottom >= 1 else thres_bottom
                                thres_bottom = (z.shape[2] // 2) - thres_bottom * (z.shape[2] // 2) if thres_bottom < 1 else thres_bottom
                                thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                thres_bottom = thres_bottom - 1
                                thres_bottom = int(thres_bottom)
                                z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des[:, thres_top]
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                z_asc = z_des.flip([-1])
                                z_asc_sel = z_asc[:, thres_bottom]
                                z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                target = (z[i, begin:end, :] < z_des_exp) & (z[i, begin:end, :] > z_asc_exp)
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                            case "r":
                                thres_top = multiplier.threshold
                                #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                thres_top = z.shape[2] - 1 if thres_top > z.shape[2] - 1 else thres_top
                                #thres_top = thres_top - 1
                                thres_top = int(thres_top)
                                thres_bottom = value
                                #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                thres_bottom = z.shape[2] - 1 if thres_bottom > z.shape[2] - 1 else thres_bottom
                                #thres_bottom = thres_bottom - 1
                                thres_bottom = int(thres_bottom)
                                z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des[:, thres_top]
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                z_asc = z_des
                                z_asc_sel = z_asc[:, thres_bottom]
                                z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                target = (z[i, begin:end, :] <= z_des_exp) & (z[i, begin:end, :] >= z_asc_exp)
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                            case "c":
                                thres_top = multiplier.threshold
                                #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                thres_top = z.shape[2] - 1 if thres_top > z.shape[2] else thres_top
                                #thres_top = thres_top - 1
                                thres_top = int(thres_top)
                                thres_bottom = value
                                #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                # thres_bottom = thres_bottom - 1
                                thres_bottom = int(thres_bottom)
                                target = torch.zeros_like(z[i, begin:end, :], dtype=torch.bool, device=z.device)
                                target[:, thres_top:thres_bottom] = True
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
    else:
        for i, pairs in enumerate(multipliers):
            for pair in pairs:
                begin = 0 if pair.begin is None else int(pair.begin * z.shape[1]) if pair.begin < 1.0 else max(min(int(pair.begin), z.shape[1]), 0)
                end = z.shape[1] if pair.end is None else int(pair.end * z.shape[1]) if pair.end < 1.0 else max(min(int(pair.end), z.shape[1]), 0)
                z_des = None
                z_asc = None
                for multiplier in pair.multipliers:
                    if multiplier.key == key:
                        mode = None
                        value = None
                        preoffset = 0.0
                        postoffset = 0.0
                        transfer_object = TransferObject(z.device)
                        crossattentioncountertargets = []
                        for option in multiplier.options:
                            match option[0]:
                                case "b" | "o" | "m" | "r":
                                    if mode is None:
                                        mode = option[0]
                                        if option[1] is not None:
                                            value = option[1]
                                        else:
                                            value = multiplier.threshold
                                case "c":
                                    if mode is None:
                                        mode = option[0]
                                        if option[1] is not None:
                                            value = option[1]
                                        else:
                                            value = multiplier.threshold + 1
                                case "n":
                                    if option[1] is not None:
                                        crossattentioncountertargets.append(int(option[1]))
                                case "pa":
                                    if option[1] is not None:
                                        preoffset += option[1]
                                case "ps":
                                    if option[1] is not None:
                                        preoffset -= option[1]
                                case "a":
                                    if option[1] is not None:
                                        postoffset += option[1]
                                case "s":
                                    if option[1] is not None:
                                        postoffset -= option[1]
                                case "ta":
                                    if option[1] is not None:
                                        transfer_value = min(int(option[1] * z.shape[2]), z.shape[2] - 1) if option[1] < 1.0 else max(min(int(option[1]), z.shape[2] - 1), 0)
                                        transfer_object.targets_positive.append(transfer_value)
                                case "ts":
                                    if option[1] is not None:
                                        transfer_value = min(int(option[1] * z.shape[2]), z.shape[2] - 1) if option[1] < 1.0 else max(min(int(option[1]), z.shape[2] - 1), 0)
                                        transfer_object.targets_negative.append(transfer_value)
                                case "taa":
                                    if option[1] is not None:
                                        transfer_value = min(int(option[1] * z.shape[2]), z.shape[2] - 1) if option[1] < 1.0 else max(min(int(option[1]), z.shape[2] - 1), 0)
                                        transfer_object.targets_absolute_positive.append(transfer_value)
                                case "tas":
                                    if option[1] is not None:
                                        transfer_value = min(int(option[1] * z.shape[2]), z.shape[2] - 1) if option[1] < 1.0 else max(min(int(option[1]), z.shape[2] - 1), 0)
                                        transfer_object.targets_absolute_negative.append(transfer_value)
                                case "tw":
                                    if option[1] is not None:
                                        transfer_object.multiplier = torch.asarray(option[1], device=z.device)
                        if len(crossattentioncountertargets) != 0 and crossattentioncounter not in crossattentioncountertargets:
                            continue
                        if z_des is None or z_asc is None:
                            z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                            z_asc = z_des.flip([-1])
                        weight = torch.asarray([multiplier.weight], dtype=torch.float32, device=z.device)
                        preoffset = torch.asarray([preoffset], dtype=torch.float32, device=z.device)
                        postoffset = torch.asarray([postoffset], dtype=torch.float32, device=z.device)
                        zero = torch.asarray([0.0], dtype=torch.float32, device=z.device)
                        one = torch.asarray([1.0], dtype=torch.float32, device=z.device)
                        match mode:
                            case None:
                                thres_top = multiplier.threshold
                                thres_top = z.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                thres_top = thres_top - 1
                                thres_top = int(thres_top)
                                #z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des[:, thres_top]
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                target = z[i, begin:end, :] >= z_des_exp
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                            case "b":
                                if multiplier.threshold == 0.0 and value == 0.0:
                                    if debug: print("Emphasis will be skipped.")
                                    pass
                                elif multiplier.threshold != 0.0 and value == 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    #z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des[:, thres_top]
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, begin:end, :] >= z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                elif multiplier.threshold == 0.0 and value != 0.0:
                                    thres_top = value
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    #z_des = z[i, begin:end, :].sort(dim=-1, descending=False).values
                                    z_des_sel = z_des[:, thres_top]
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, begin:end, :] <= z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                elif multiplier.threshold != 0.0 and value != 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    thres_bottom = value
                                    #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                    thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                    thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                    thres_bottom = thres_bottom - 1
                                    thres_bottom = int(thres_bottom)
                                    #z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des[:, thres_top]
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    #z_asc = z_des.flip([-1])
                                    z_asc_sel = z_asc[:, thres_bottom]
                                    z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = (z[i, begin:end, :] >= z_des_exp) | (z[i, begin:end, :] <= z_asc_exp)
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                            case "o":
                                if multiplier.threshold == 0.0 and value == 0.0:
                                    target = torch.ones_like(z[i, begin:end, :], dtype=torch.bool, device=z.device)
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                elif multiplier.threshold != 0.0 and value == 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    #z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des[:, thres_top]
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, begin:end, :] < z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                elif multiplier.threshold == 0.0 and value != 0.0:
                                    thres_top = value
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    #z_des = z[i, begin:end, :].sort(dim=-1, descending=False).values
                                    z_des_sel = z_des[:, thres_top]
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, begin:end, :] > z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                                elif multiplier.threshold != 0.0 and value != 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = int(thres_top)
                                    thres_bottom = value
                                    #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                    thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                    thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                    thres_bottom = thres_bottom - 1
                                    thres_bottom = int(thres_bottom)
                                    #z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des[:, thres_top]
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    #z_asc = z_des.flip([-1])
                                    z_asc_sel = z_asc[:, thres_bottom]
                                    z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = (z[i, begin:end, :] < z_des_exp) & (z[i, begin:end, :] > z_asc_exp)
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                            case "m":
                                thres_top = multiplier.threshold
                                thres_top = (z.shape[2] // 2) - thres_top if thres_top >= 1 else thres_top
                                thres_top = (z.shape[2] // 2) - thres_top * (z.shape[2] // 2) if thres_top < 1 else thres_top
                                thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                thres_top = thres_top - 1
                                thres_top = int(thres_top)
                                thres_bottom = value
                                thres_bottom = (z.shape[2] // 2) - thres_bottom if thres_bottom >= 1 else thres_bottom
                                thres_bottom = (z.shape[2] // 2) - thres_bottom * (z.shape[2] // 2) if thres_bottom < 1 else thres_bottom
                                thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                thres_bottom = thres_bottom - 1
                                thres_bottom = int(thres_bottom)
                                #z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des[:, thres_top]
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                #z_asc = z_des.flip([-1])
                                z_asc_sel = z_asc[:, thres_bottom]
                                z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                target = (z[i, begin:end, :] < z_des_exp) & (z[i, begin:end, :] > z_asc_exp)
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                            case "r":
                                thres_top = multiplier.threshold
                                #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                thres_top = z.shape[2] - 1 if thres_top > z.shape[2] - 1 else thres_top
                                #thres_top = thres_top - 1
                                thres_top = int(thres_top)
                                thres_bottom = value
                                #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                thres_bottom = z.shape[2] - 1 if thres_bottom > z.shape[2] - 1 else thres_bottom
                                #thres_bottom = thres_bottom - 1
                                thres_bottom = int(thres_bottom)
                                #z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des[:, thres_top]
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                #z_asc = z_des
                                z_asc_sel = z_des.index_select(dim=-1, index=thres_bottom).diag()
                                z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                target = (z[i, begin:end, :] <= z_des_exp) & (z[i, begin:end, :] >= z_asc_exp)
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                            case "c":
                                thres_top = multiplier.threshold
                                #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                thres_top = z.shape[2] - 1 if thres_top > z.shape[2] else thres_top
                                #thres_top = thres_top - 1
                                thres_top = int(thres_top)
                                thres_bottom = value
                                #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                # thres_bottom = thres_bottom - 1
                                thres_bottom = int(thres_bottom)
                                target = torch.zeros_like(z[i, begin:end, :], dtype=torch.bool, device=z.device)
                                target[:, thres_top:thres_bottom] = True
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                z_des = None
                z_asc = None
    return z

def emphasis_crossattention2(z: torch.Tensor, heads, multipliers_pos: list[list[EmphasisPair]], multipliers_neg: list[list[EmphasisPair]], keys: list[str], crossattentioncounter, emphasis_view_update, debug):
    z = einops.rearrange(z, "(i h) l t -> i h t l", h=heads)
    latent_size = z.shape[3]
    multipliers = multipliers_neg + multipliers_pos
    if emphasis_view_update:
        for i, pairs in enumerate(multipliers):
            for pair in pairs:
                for multiplier in pair.multipliers:
                    if multiplier.key not in keys:
                        continue
                    mode = None
                    value = None
                    preoffset = 0.0
                    postoffset = 0.0
                    transfer_object = TransferObject(z.device)
                    crossattentioncountertargets = []
                    for option in multiplier.options:
                        match option[0]:
                            case "b" | "o" | "m" | "r":
                                if mode is None:
                                    mode = option[0]
                                    if option[1] is not None:
                                        value = option[1]
                                    else:
                                        value = multiplier.threshold
                            case "c":
                                if mode is None:
                                    mode = option[0]
                                    if option[1] is not None:
                                        value = option[1]
                                    else:
                                        value = multiplier.threshold + 1
                            case "n":
                                if option[1] is not None:
                                    crossattentioncountertargets.append(int(option[1]))
                            case "pa":
                                if option[1] is not None:
                                    preoffset += option[1]
                            case "ps":
                                if option[1] is not None:
                                    preoffset -= option[1]
                            case "a":
                                if option[1] is not None:
                                    postoffset += option[1]
                            case "s":
                                if option[1] is not None:
                                    postoffset -= option[1]
                            case "ta":
                                if option[1] is not None:
                                    if multiplier.key == keys[0]:
                                        if option[1] < 1.0:
                                            transfer_value = min(int(option[1] * z.shape[1] * z.shape[3]), z.shape[1] * z.shape[3] - 1)
                                        else:
                                            max(min(int(option[1]), z.shape[1] * z.shape[3] - 1), 0)
                                    elif multiplier.key == keys[1]:
                                        if option[1] < 1.0:
                                            transfer_value = min(int(option[1] * z.shape[3]), z.shape[3] - 1)
                                        else:
                                            max(min(int(option[1]), z.shape[3] - 1), 0)
                                    elif multiplier.key == keys[2]:
                                        if option[1] < 1.0:
                                            transfer_value = min(int(option[1] * z.shape[1]), z.shape[1] - 1)
                                        else:
                                            max(min(int(option[1]), z.shape[1] - 1), 0)
                                    else:
                                        continue
                                    transfer_object.targets_positive.append(transfer_value)
                            case "ts":
                                if option[1] is not None:
                                    if multiplier.key == keys[0]:
                                        if option[1] < 1.0:
                                            transfer_value = min(int(option[1] * z.shape[1] * z.shape[3]), z.shape[1] * z.shape[3] - 1)
                                        else:
                                            max(min(int(option[1]), z.shape[1] * z.shape[3] - 1), 0)
                                    elif multiplier.key == keys[1]:
                                        if option[1] < 1.0:
                                            transfer_value = min(int(option[1] * z.shape[3]), z.shape[3] - 1)
                                        else:
                                            max(min(int(option[1]), z.shape[3] - 1), 0)
                                    elif multiplier.key == keys[2]:
                                        if option[1] < 1.0:
                                            transfer_value = min(int(option[1] * z.shape[1]), z.shape[1] - 1)
                                        else:
                                            max(min(int(option[1]), z.shape[1] - 1), 0)
                                    else:
                                        continue
                                    transfer_object.targets_negative.append(transfer_value)
                            case "taa":
                                if option[1] is not None:
                                    if multiplier.key == keys[0]:
                                        if option[1] < 1.0:
                                            transfer_value = min(int(option[1] * z.shape[1] * z.shape[3]), z.shape[1] * z.shape[3] - 1)
                                        else:
                                            max(min(int(option[1]), z.shape[1] * z.shape[3] - 1), 0)
                                    elif multiplier.key == keys[1]:
                                        if option[1] < 1.0:
                                            transfer_value = min(int(option[1] * z.shape[3]), z.shape[3] - 1)
                                        else:
                                            max(min(int(option[1]), z.shape[3] - 1), 0)
                                    elif multiplier.key == keys[2]:
                                        if option[1] < 1.0:
                                            transfer_value = min(int(option[1] * z.shape[1]), z.shape[1] - 1)
                                        else:
                                            max(min(int(option[1]), z.shape[1] - 1), 0)
                                    else:
                                        continue
                                    transfer_object.targets_absolute_positive.append(transfer_value)
                            case "tas":
                                if option[1] is not None:
                                    if multiplier.key == keys[0]:
                                        if option[1] < 1.0:
                                            transfer_value = min(int(option[1] * z.shape[1] * z.shape[3]), z.shape[1] * z.shape[3] - 1)
                                        else:
                                            max(min(int(option[1]), z.shape[1] * z.shape[3] - 1), 0)
                                    elif multiplier.key == keys[1]:
                                        if option[1] < 1.0:
                                            transfer_value = min(int(option[1] * z.shape[3]), z.shape[3] - 1)
                                        else:
                                            max(min(int(option[1]), z.shape[3] - 1), 0)
                                    elif multiplier.key == keys[2]:
                                        if option[1] < 1.0:
                                            transfer_value = min(int(option[1] * z.shape[1]), z.shape[1] - 1)
                                        else:
                                            max(min(int(option[1]), z.shape[1] - 1), 0)
                                    else:
                                        continue
                                    transfer_object.targets_absolute_negative.append(transfer_value)
                            case "tw":
                                if option[1] is not None:
                                    transfer_object.multiplier = torch.asarray(option[1], device=z.device)
                    if len(crossattentioncountertargets) != 0 and crossattentioncounter not in crossattentioncountertargets:
                        continue
                    if multiplier.key == keys[0]:
                        # "q", "s"
                        dim1_size = 1
                        z = einops.rearrange(z, "i h t l -> i t (h l)")
                        begin = 0 if pair.begin is None else int(pair.begin * z.shape[1]) if pair.begin < 1.0 else max(min(int(pair.begin) * dim1_size, z.shape[1]), 0)
                        end = z.shape[1] if pair.end is None else int(pair.end * z.shape[1]) if pair.end < 1.0 else max(min(int(pair.end) * dim1_size, z.shape[1]), 0)
                    elif multiplier.key == keys[1]:
                        # "ql", "sl"
                        dim1_size = z.shape[1]
                        z = einops.rearrange(z, "i h t l -> i (t h) l")
                        begin = 0 if pair.begin is None else int(pair.begin * z.shape[1]) if pair.begin < 1.0 else max(min(int(pair.begin) * dim1_size, z.shape[1]), 0)
                        end = z.shape[1] if pair.end is None else int(pair.end * z.shape[1]) if pair.end < 1.0 else max(min(int(pair.end) * dim1_size, z.shape[1]), 0)
                    elif multiplier.key == keys[2]:
                        # "qh", "sh"
                        dim1_size = z.shape[3]
                        z = einops.rearrange(z, "i h t l -> i (t l) h")
                        begin = 0 if pair.begin is None else int(pair.begin * z.shape[1]) if pair.begin < 1.0 else max(min(int(pair.begin) * dim1_size, z.shape[1]), 0)
                        end = z.shape[1] if pair.end is None else int(pair.end * z.shape[1]) if pair.end < 1.0 else max(min(int(pair.end) * dim1_size, z.shape[1]), 0)
                    else:
                        continue   
                    weight = torch.asarray([multiplier.weight], dtype=torch.float32, device=z.device)
                    preoffset = torch.asarray([preoffset], dtype=torch.float32, device=z.device)
                    postoffset = torch.asarray([postoffset], dtype=torch.float32, device=z.device)
                    zero = torch.asarray([0.0], dtype=torch.float32, device=z.device)
                    one = torch.asarray([1.0], dtype=torch.float32, device=z.device)
                    match mode:
                        case None:
                            thres_top = multiplier.threshold
                            thres_top = z.shape[2] if thres_top == 0 else thres_top
                            thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                            thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                            thres_top = thres_top - 1
                            thres_top = int(thres_top)
                            z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                            z_des_sel = z_des[:, thres_top]
                            z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                            target = z[i, begin:end, :] >= z_des_exp
                            if debug: print(target.to(device="cpu").nonzero().tolist())
                            apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                        case "b":
                            if multiplier.threshold == 0.0 and value == 0.0:
                                if debug: print("Emphasis will be skipped.")
                                pass
                            elif multiplier.threshold != 0.0 and value == 0.0:
                                thres_top = multiplier.threshold
                                #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                thres_top = thres_top - 1
                                thres_top = int(thres_top)
                                z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des[:, thres_top]
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                target = z[i, begin:end, :] >= z_des_exp
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                            elif multiplier.threshold == 0.0 and value != 0.0:
                                thres_top = value
                                #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                thres_top = thres_top - 1
                                thres_top = int(thres_top)
                                z_des = z[i, begin:end, :].sort(dim=-1, descending=False).values
                                z_des_sel = z_des[:, thres_top]
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                target = z[i, begin:end, :] <= z_des_exp
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                            elif multiplier.threshold != 0.0 and value != 0.0:
                                thres_top = multiplier.threshold
                                #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                thres_top = thres_top - 1
                                thres_top = int(thres_top)
                                thres_bottom = value
                                #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                thres_bottom = thres_bottom - 1
                                thres_bottom = int(thres_bottom)
                                z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des[:, thres_top]
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                z_asc = z_des.flip([-1])
                                z_asc_sel = z_asc[:, thres_bottom]
                                z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                target = (z[i, begin:end, :] >= z_des_exp) | (z[i, begin:end, :] <= z_asc_exp)
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                        case "o":
                            if multiplier.threshold == 0.0 and value == 0.0:
                                target = torch.ones_like(z[i, begin:end, :], dtype=torch.bool, device=z.device)
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                            elif multiplier.threshold != 0.0 and value == 0.0:
                                thres_top = multiplier.threshold
                                #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                thres_top = thres_top - 1
                                thres_top = int(thres_top)
                                z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des[:, thres_top]
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                target = z[i, begin:end, :] < z_des_exp
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                            elif multiplier.threshold == 0.0 and value != 0.0:
                                thres_top = value
                                #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                thres_top = thres_top - 1
                                thres_top = int(thres_top)
                                z_des = z[i, begin:end, :].sort(dim=-1, descending=False).values
                                z_des_sel = z_des[:, thres_top]
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                target = z[i, begin:end, :] > z_des_exp
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                            elif multiplier.threshold != 0.0 and value != 0.0:
                                thres_top = multiplier.threshold
                                #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                thres_top = thres_top - 1
                                thres_top = int(thres_top)
                                thres_bottom = value
                                #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                thres_bottom = thres_bottom - 1
                                thres_bottom = int(thres_bottom)
                                z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des[:, thres_top]
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                z_asc = z_des.flip([-1])
                                z_asc_sel = z_asc[:, thres_bottom]
                                z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                target = (z[i, begin:end, :] < z_des_exp) & (z[i, begin:end, :] > z_asc_exp)
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                        case "m":
                            thres_top = multiplier.threshold
                            thres_top = (z.shape[2] // 2) - thres_top if thres_top >= 1 else thres_top
                            thres_top = (z.shape[2] // 2) - thres_top * (z.shape[2] // 2) if thres_top < 1 else thres_top
                            thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                            thres_top = thres_top - 1
                            thres_top = int(thres_top)
                            thres_bottom = value
                            thres_bottom = (z.shape[2] // 2) - thres_bottom if thres_bottom >= 1 else thres_bottom
                            thres_bottom = (z.shape[2] // 2) - thres_bottom * (z.shape[2] // 2) if thres_bottom < 1 else thres_bottom
                            thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                            thres_bottom = thres_bottom - 1
                            thres_bottom = int(thres_bottom)
                            z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                            z_des_sel = z_des[:, thres_top]
                            z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                            z_asc = z_des.flip([-1])
                            z_asc_sel = z_asc[:, thres_bottom]
                            z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                            target = (z[i, begin:end, :] < z_des_exp) & (z[i, begin:end, :] > z_asc_exp)
                            if debug: print(target.to(device="cpu").nonzero().tolist())
                            apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                        case "r":
                            thres_top = multiplier.threshold
                            #thres_top = z.shape[2] if thres_top == 0 else thres_top
                            thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                            thres_top = z.shape[2] - 1 if thres_top > z.shape[2] - 1 else thres_top
                            #thres_top = thres_top - 1
                            thres_top = int(thres_top)
                            thres_bottom = value
                            #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                            thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                            thres_bottom = z.shape[2] - 1 if thres_bottom > z.shape[2] - 1 else thres_bottom
                            #thres_bottom = thres_bottom - 1
                            thres_bottom = int(thres_bottom)
                            z_des = z[i, begin:end, :].sort(dim=-1, descending=True).values
                            z_des_sel = z_des[:, thres_top]
                            z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                            z_asc = z_des
                            z_asc_sel = z_asc[:, thres_bottom]
                            z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                            target = (z[i, begin:end, :] <= z_des_exp) & (z[i, begin:end, :] >= z_asc_exp)
                            if debug: print(target.to(device="cpu").nonzero().tolist())
                            apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                        case "c":
                            thres_top = multiplier.threshold
                            #thres_top = z.shape[2] if thres_top == 0 else thres_top
                            thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                            thres_top = z.shape[2] - 1 if thres_top > z.shape[2] else thres_top
                            #thres_top = thres_top - 1
                            thres_top = int(thres_top)
                            thres_bottom = value
                            #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                            thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                            thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                            # thres_bottom = thres_bottom - 1
                            thres_bottom = int(thres_bottom)
                            target = torch.zeros_like(z[i, begin:end, :], dtype=torch.bool, device=z.device)
                            target[:, thres_top:thres_bottom] = True
                            if debug: print(target.to(device="cpu").nonzero().tolist())
                            apply_emphasis(z, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                    if multiplier.key == keys[0]:
                        # "q", "s"
                        z = einops.rearrange(z, "i t (h l) -> i h t l", l=latent_size)
                    elif multiplier.key == keys[1]:
                        # "ql", "sl"
                        z = einops.rearrange(z, "i (t h) l -> i h t l", h=heads)
                    elif multiplier.key == keys[2]:
                        # "qh", "sh"
                        z = einops.rearrange(z, "i (t l) h -> i h t l", l=latent_size)
                    else:
                        continue  
                
    else:
        for i, pairs in enumerate(multipliers):
            for pair in pairs:
                dim1_size_hl = 1
                z_hl = None
                begin_hl = 0 if pair.begin is None else int(pair.begin * z.shape[2] * dim1_size_hl) if pair.begin < 1.0 else max(min(int(pair.begin) * dim1_size_hl, z.shape[2] * dim1_size_hl), 0)
                end_hl = z.shape[2] if pair.end is None else int(pair.end * z.shape[2] * dim1_size_hl) if pair.end < 1.0 else max(min(int(pair.end) * dim1_size_hl, z.shape[2] * dim1_size_hl), 0)
                dim1_size_l = z.shape[1]
                z_l = None
                begin_l = 0 if pair.begin is None else int(pair.begin * z.shape[2] * dim1_size_l) if pair.begin < 1.0 else max(min(int(pair.begin) * dim1_size_l, z.shape[2] * dim1_size_l), 0)
                end_l = z.shape[2] * dim1_size_l if pair.end is None else int(pair.end * z.shape[2] * dim1_size_l) if pair.end < 1.0 else max(min(int(pair.end) * dim1_size_l, z.shape[2] * z.shape[1]), 0)
                dim1_size_h = z.shape[3]
                z_h = None
                begin_h = 0 if pair.begin is None else int(pair.begin * z.shape[2] * dim1_size_h) if pair.begin < 1.0 else max(min(int(pair.begin) * dim1_size_h, z.shape[2] * dim1_size_h), 0)
                end_h = z.shape[2] * dim1_size_h if pair.end is None else int(pair.end * z.shape[2] * dim1_size_h) if pair.end < 1.0 else max(min(int(pair.end) * dim1_size_h, z.shape[2] * dim1_size_h), 0)
                for multiplier in pair.multipliers:
                    if multiplier.key not in keys:
                        continue
                    mode = None
                    value = None
                    preoffset = 0.0
                    postoffset = 0.0
                    transfer_object = TransferObject(z.device)
                    crossattentioncountertargets = []
                    for option in multiplier.options:
                        match option[0]:
                            case "b" | "o" | "m" | "r":
                                if mode is None:
                                    mode = option[0]
                                    if option[1] is not None:
                                        value = option[1]
                                    else:
                                        value = multiplier.threshold
                            case "c":
                                if mode is None:
                                    mode = option[0]
                                    if option[1] is not None:
                                        value = option[1]
                                    else:
                                        value = multiplier.threshold + 1
                            case "n":
                                if option[1] is not None:
                                    crossattentioncountertargets.append(int(option[1]))
                            case "pa":
                                if option[1] is not None:
                                    preoffset += option[1]
                            case "ps":
                                if option[1] is not None:
                                    preoffset -= option[1]
                            case "a":
                                if option[1] is not None:
                                    postoffset += option[1]
                            case "s":
                                if option[1] is not None:
                                    postoffset -= option[1]
                            case "ta":
                                if option[1] is not None:
                                    if multiplier.key == keys[0]:
                                        if option[1] < 1.0:
                                            transfer_value = min(int(option[1] * z.shape[1] * z.shape[3]), z.shape[1] * z.shape[3] - 1)
                                        else:
                                            max(min(int(option[1]), z.shape[1] * z.shape[3] - 1), 0)
                                    elif multiplier.key == keys[1]:
                                        if option[1] < 1.0:
                                            transfer_value = min(int(option[1] * z.shape[3]), z.shape[3] - 1)
                                        else:
                                            max(min(int(option[1]), z.shape[3] - 1), 0)
                                    elif multiplier.key == keys[2]:
                                        if option[1] < 1.0:
                                            transfer_value = min(int(option[1] * z.shape[1]), z.shape[1] - 1)
                                        else:
                                            max(min(int(option[1]), z.shape[1] - 1), 0)
                                    else:
                                        continue
                                    transfer_object.targets_positive.append(transfer_value)
                            case "ts":
                                if option[1] is not None:
                                    if multiplier.key == keys[0]:
                                        if option[1] < 1.0:
                                            transfer_value = min(int(option[1] * z.shape[1] * z.shape[3]), z.shape[1] * z.shape[3] - 1)
                                        else:
                                            max(min(int(option[1]), z.shape[1] * z.shape[3] - 1), 0)
                                    elif multiplier.key == keys[1]:
                                        if option[1] < 1.0:
                                            transfer_value = min(int(option[1] * z.shape[3]), z.shape[3] - 1)
                                        else:
                                            max(min(int(option[1]), z.shape[3] - 1), 0)
                                    elif multiplier.key == keys[2]:
                                        if option[1] < 1.0:
                                            transfer_value = min(int(option[1] * z.shape[1]), z.shape[1] - 1)
                                        else:
                                            max(min(int(option[1]), z.shape[1] - 1), 0)
                                    else:
                                        continue
                                    transfer_object.targets_negative.append(transfer_value)
                            case "taa":
                                if option[1] is not None:
                                    if multiplier.key == keys[0]:
                                        if option[1] < 1.0:
                                            transfer_value = min(int(option[1] * z.shape[1] * z.shape[3]), z.shape[1] * z.shape[3] - 1)
                                        else:
                                            max(min(int(option[1]), z.shape[1] * z.shape[3] - 1), 0)
                                    elif multiplier.key == keys[1]:
                                        if option[1] < 1.0:
                                            transfer_value = min(int(option[1] * z.shape[3]), z.shape[3] - 1)
                                        else:
                                            max(min(int(option[1]), z.shape[3] - 1), 0)
                                    elif multiplier.key == keys[2]:
                                        if option[1] < 1.0:
                                            transfer_value = min(int(option[1] * z.shape[1]), z.shape[1] - 1)
                                        else:
                                            max(min(int(option[1]), z.shape[1] - 1), 0)
                                    else:
                                        continue
                                    transfer_object.targets_absolute_positive.append(transfer_value)
                            case "tas":
                                if option[1] is not None:
                                    if multiplier.key == keys[0]:
                                        if option[1] < 1.0:
                                            transfer_value = min(int(option[1] * z.shape[1] * z.shape[3]), z.shape[1] * z.shape[3] - 1)
                                        else:
                                            max(min(int(option[1]), z.shape[1] * z.shape[3] - 1), 0)
                                    elif multiplier.key == keys[1]:
                                        if option[1] < 1.0:
                                            transfer_value = min(int(option[1] * z.shape[3]), z.shape[3] - 1)
                                        else:
                                            max(min(int(option[1]), z.shape[3] - 1), 0)
                                    elif multiplier.key == keys[2]:
                                        if option[1] < 1.0:
                                            transfer_value = min(int(option[1] * z.shape[1]), z.shape[1] - 1)
                                        else:
                                            max(min(int(option[1]), z.shape[1] - 1), 0)
                                    else:
                                        continue
                                    transfer_object.targets_absolute_negative.append(transfer_value)
                            case "tw":
                                if option[1] is not None:
                                    transfer_object.multiplier = torch.asarray(option[1], device=z.device)
                    if len(crossattentioncountertargets) != 0 and crossattentioncounter not in crossattentioncountertargets:
                        continue
                    if multiplier.key == keys[0]:
                        # "q", "s"
                        if z_hl is None:
                            z_hl = einops.rearrange(z, "i h t l -> i t (h l)")
                            z_des_hl = z_hl[i, begin_hl:end_hl, :].sort(dim=-1, descending=True).values
                            z_asc_hl = z_des_hl.flip([-1])
                        dim1_size = dim1_size_hl
                        begin = begin_hl
                        end = end_hl
                        z_v = z_hl
                        z_des = z_des_hl
                        z_asc = z_asc_hl
                    elif multiplier.key == keys[1]:
                        # "ql", "sl"
                        if z_l is None:
                            z_l = einops.rearrange(z, "i h t l -> i (t h) l")
                            z_des_l = z_l[i, begin_l:end_l, :].sort(dim=-1, descending=True).values
                            z_asc_l = z_des_l.flip([-1])
                        dim1_size = dim1_size_l
                        begin = begin_l
                        end = end_l
                        z_v = z_l
                        z_des = z_des_l
                        z_asc = z_asc_l
                    elif multiplier.key == keys[2]:
                        # "qh", "sh"
                        if z_h is None:
                            z_h = einops.rearrange(z, "i h t l -> i (t l) h")
                            z_des_h = z_h[i, begin_h:end_h, :].sort(dim=-1, descending=True).values
                            z_asc_h = z_des_h.flip([-1])
                        dim1_size = dim1_size_h
                        begin = begin_h
                        end = end_h
                        z_v = z_h
                        z_des = z_des_h
                        z_asc = z_asc_h
                    else:
                        continue   
                    weight = torch.asarray([multiplier.weight], dtype=torch.float32, device=z.device)
                    preoffset = torch.asarray([preoffset], dtype=torch.float32, device=z.device)
                    postoffset = torch.asarray([postoffset], dtype=torch.float32, device=z.device)
                    zero = torch.asarray([0.0], dtype=torch.float32, device=z.device)
                    one = torch.asarray([1.0], dtype=torch.float32, device=z.device)
                    match mode:
                        case None:
                            thres_top = multiplier.threshold
                            thres_top = z_v.shape[2] if thres_top == 0 else thres_top
                            thres_top = thres_top * z_v.shape[2] if thres_top < 1 else thres_top
                            thres_top = z_v.shape[2] if thres_top > z_v.shape[2] else thres_top
                            thres_top = thres_top - 1
                            thres_top = int(thres_top)
                            #z_des = z_v[i, begin:end, :].sort(dim=-1, descending=True).values
                            z_des_sel = z_des[:, thres_top]
                            z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z_v.shape[2]))
                            target = z_v[i, begin:end, :] >= z_des_exp
                            if debug: print(target.to(device="cpu").nonzero().tolist())
                            apply_emphasis(z_v, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                        case "b":
                            if multiplier.threshold == 0.0 and value == 0.0:
                                if debug: print("Emphasis will be skipped.")
                                pass
                            elif multiplier.threshold != 0.0 and value == 0.0:
                                thres_top = multiplier.threshold
                                #thres_top = z_v.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z_v.shape[2] if thres_top < 1 else thres_top
                                thres_top = z_v.shape[2] if thres_top > z_v.shape[2] else thres_top
                                thres_top = thres_top - 1
                                thres_top = int(thres_top)
                                #z_des = z_v[i, begin:end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des[:, thres_top]
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z_v.shape[2]))
                                target = z_v[i, begin:end, :] >= z_des_exp
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                apply_emphasis(z_v, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                            elif multiplier.threshold == 0.0 and value != 0.0:
                                thres_top = value
                                #thres_top = z_v.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z_v.shape[2] if thres_top < 1 else thres_top
                                thres_top = z_v.shape[2] if thres_top > z_v.shape[2] else thres_top
                                thres_top = thres_top - 1
                                thres_top = int(thres_top)
                                #z_des = z_v[i, begin:end, :].sort(dim=-1, descending=False).values
                                z_des_sel = z_des[:, thres_top]
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z_v.shape[2]))
                                target = z_v[i, begin:end, :] <= z_des_exp
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                apply_emphasis(z_v, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                            elif multiplier.threshold != 0.0 and value != 0.0:
                                thres_top = multiplier.threshold
                                #thres_top = z_v.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z_v.shape[2] if thres_top < 1 else thres_top
                                thres_top = z_v.shape[2] if thres_top > z_v.shape[2] else thres_top
                                thres_top = thres_top - 1
                                thres_top = int(thres_top)
                                thres_bottom = value
                                #thres_bottom = z_v.shape[2] if thres_bottom == 0 else thres_bottom
                                thres_bottom = thres_bottom * z_v.shape[2] if thres_bottom < 1 else thres_bottom
                                thres_bottom = z_v.shape[2] if thres_bottom > z_v.shape[2] else thres_bottom
                                thres_bottom = thres_bottom - 1
                                thres_bottom = int(thres_bottom)
                                #z_des = z_v[i, begin:end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des[:, thres_top]
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z_v.shape[2]))
                                #z_asc = z_des.flip([-1])
                                z_asc_sel = z_asc[:, thres_bottom]
                                z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z_v.shape[2]))
                                target = (z_v[i, begin:end, :] >= z_des_exp) | (z_v[i, begin:end, :] <= z_asc_exp)
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                apply_emphasis(z_v, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                        case "o":
                            if multiplier.threshold == 0.0 and value == 0.0:
                                target = torch.ones_like(z_v[i, begin:end, :], dtype=torch.bool, device=z_v.device)
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                apply_emphasis(z_v, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                            elif multiplier.threshold != 0.0 and value == 0.0:
                                thres_top = multiplier.threshold
                                #thres_top = z_v.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z_v.shape[2] if thres_top < 1 else thres_top
                                thres_top = z_v.shape[2] if thres_top > z_v.shape[2] else thres_top
                                thres_top = thres_top - 1
                                thres_top = int(thres_top)
                                #z_des = z_v[i, begin:end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des[:, thres_top]
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z_v.shape[2]))
                                target = z_v[i, begin:end, :] < z_des_exp
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                apply_emphasis(z_v, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                            elif multiplier.threshold == 0.0 and value != 0.0:
                                thres_top = value
                                #thres_top = z_v.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z_v.shape[2] if thres_top < 1 else thres_top
                                thres_top = z_v.shape[2] if thres_top > z_v.shape[2] else thres_top
                                thres_top = thres_top - 1
                                thres_top = int(thres_top)
                                #z_des = z_v[i, begin:end, :].sort(dim=-1, descending=False).values
                                z_des_sel = z_des[:, thres_top]
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z_v.shape[2]))
                                target = z_v[i, begin:end, :] > z_des_exp
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                apply_emphasis(z_v, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                            elif multiplier.threshold != 0.0 and value != 0.0:
                                thres_top = multiplier.threshold
                                #thres_top = z_v.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z_v.shape[2] if thres_top < 1 else thres_top
                                thres_top = z_v.shape[2] if thres_top > z_v.shape[2] else thres_top
                                thres_top = thres_top - 1
                                thres_top = int(thres_top)
                                thres_bottom = value
                                #thres_bottom = z_v.shape[2] if thres_bottom == 0 else thres_bottom
                                thres_bottom = thres_bottom * z_v.shape[2] if thres_bottom < 1 else thres_bottom
                                thres_bottom = z_v.shape[2] if thres_bottom > z_v.shape[2] else thres_bottom
                                thres_bottom = thres_bottom - 1
                                thres_bottom = int(thres_bottom)
                                #z_des = z_v[i, begin:end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des[:, thres_top]
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z_v.shape[2]))
                                #z_asc = z_des.flip([-1])
                                z_asc_sel = z_asc[:, thres_bottom]
                                z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z_v.shape[2]))
                                target = (z_v[i, begin:end, :] < z_des_exp) & (z_v[i, begin:end, :] > z_asc_exp)
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                apply_emphasis(z_v, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                        case "m":
                            thres_top = multiplier.threshold
                            thres_top = (z_v.shape[2] // 2) - thres_top if thres_top >= 1 else thres_top
                            thres_top = (z_v.shape[2] // 2) - thres_top * (z_v.shape[2] // 2) if thres_top < 1 else thres_top
                            thres_top = z_v.shape[2] if thres_top > z_v.shape[2] else thres_top
                            thres_top = thres_top - 1
                            thres_top = int(thres_top)
                            thres_bottom = value
                            thres_bottom = (z_v.shape[2] // 2) - thres_bottom if thres_bottom >= 1 else thres_bottom
                            thres_bottom = (z_v.shape[2] // 2) - thres_bottom * (z_v.shape[2] // 2) if thres_bottom < 1 else thres_bottom
                            thres_bottom = z_v.shape[2] if thres_bottom > z_v.shape[2] else thres_bottom
                            thres_bottom = thres_bottom - 1
                            thres_bottom = int(thres_bottom)
                            #z_des = z_v[i, begin:end, :].sort(dim=-1, descending=True).values
                            z_des_sel = z_des[:, thres_top]
                            z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z_v.shape[2]))
                            #z_asc = z_des.flip([-1])
                            z_asc_sel = z_asc[:, thres_bottom]
                            z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z_v.shape[2]))
                            target = (z_v[i, begin:end, :] < z_des_exp) & (z_v[i, begin:end, :] > z_asc_exp)
                            if debug: print(target.to(device="cpu").nonzero().tolist())
                            apply_emphasis(z_v, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                        case "r":
                            thres_top = multiplier.threshold
                            #thres_top = z_v.shape[2] if thres_top == 0 else thres_top
                            thres_top = thres_top * z_v.shape[2] if thres_top < 1 else thres_top
                            thres_top = z_v.shape[2] - 1 if thres_top > z_v.shape[2] - 1 else thres_top
                            #thres_top = thres_top - 1
                            thres_top = int(thres_top)
                            thres_bottom = value
                            #thres_bottom = z_v.shape[2] if thres_bottom == 0 else thres_bottom
                            thres_bottom = thres_bottom * z_v.shape[2] if thres_bottom < 1 else thres_bottom
                            thres_bottom = z_v.shape[2] - 1 if thres_bottom > z_v.shape[2] - 1 else thres_bottom
                            #thres_bottom = thres_bottom - 1
                            thres_bottom = int(thres_bottom)
                            #z_des = z_v[i, begin:end, :].sort(dim=-1, descending=True).values
                            z_des_sel = z_des[:, thres_top]
                            z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z_v.shape[2]))
                            #z_asc = z_des
                            z_asc_sel = z_des.index_select(dim=-1, index=thres_bottom).diag()
                            z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z_v.shape[2]))
                            target = (z_v[i, begin:end, :] <= z_des_exp) & (z_v[i, begin:end, :] >= z_asc_exp)
                            if debug: print(target.to(device="cpu").nonzero().tolist())
                            apply_emphasis(z_v, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                        case "c":
                            thres_top = multiplier.threshold
                            #thres_top = z_v.shape[2] if thres_top == 0 else thres_top
                            thres_top = thres_top * z_v.shape[2] if thres_top < 1 else thres_top
                            thres_top = z_v.shape[2] - 1 if thres_top > z_v.shape[2] else thres_top
                            #thres_top = thres_top - 1
                            thres_top = int(thres_top)
                            thres_bottom = value
                            #thres_bottom = z_v.shape[2] if thres_bottom == 0 else thres_bottom
                            thres_bottom = thres_bottom * z_v.shape[2] if thres_bottom < 1 else thres_bottom
                            thres_bottom = z_v.shape[2] if thres_bottom > z_v.shape[2] else thres_bottom
                            # thres_bottom = thres_bottom - 1
                            thres_bottom = int(thres_bottom)
                            target = torch.zeros_like(z_v[i, begin:end, :], dtype=torch.bool, device=z_v.device)
                            target[:, thres_top:thres_bottom] = True
                            if debug: print(target.to(device="cpu").nonzero().tolist())
                            apply_emphasis(z_v, i, begin, end, target, preoffset, weight, postoffset, zero, one, transfer_object)
                    if multiplier.key == keys[0]:
                        # "q", "s"
                        z = einops.rearrange(z_v, "i t (h l) -> i h t l", l=latent_size)
                        z_hl = einops.rearrange(z, "i h t l -> i t (h l)")
                        z_l = einops.rearrange(z, "i h t l -> i (t h) l")
                        z_h = einops.rearrange(z, "i h t l -> i (t l) h")
                    elif multiplier.key == keys[1]:
                        # "ql", "sl"
                        z = einops.rearrange(z_v, "i (t h) l -> i h t l", h=heads)
                        z_hl = einops.rearrange(z, "i h t l -> i t (h l)")
                        z_l = einops.rearrange(z, "i h t l -> i (t h) l")
                        z_h = einops.rearrange(z, "i h t l -> i (t l) h")
                    elif multiplier.key == keys[2]:
                        # "qh", "sh"
                        z = einops.rearrange(z_v, "i (t l) h -> i h t l", l=latent_size)
                        z_hl = einops.rearrange(z, "i h t l -> i t (h l)")
                        z_l = einops.rearrange(z, "i h t l -> i (t h) l")
                        z_h = einops.rearrange(z, "i h t l -> i (t l) h")
                    else:
                        continue  
                z_hl = None
                z_l = None
                z_h = None
    z = einops.rearrange(z, "i h t l -> (i h) l t")
    return z